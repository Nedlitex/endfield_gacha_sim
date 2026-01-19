import asyncio
from typing import Optional

from pydantic import BaseModel, Field

from banner import Banner, DrawResult, Operator, RewardType


class BannerResource(BaseModel):
    """Tracks available draw resources for a banner simulation.

    Resources are categorized into three types:
    - normal_draws: Can be used at any banner, inherits across banners,
      receives inherited draw rewards and initial draws
    - current_banner_draws: Can only be used at the current banner,
      does NOT carry over to next banner, comes from previous banner's
      next_banner_draw rewards
    - special_draws: Can only be used at the current banner as special draws,
      does NOT carry over, comes from special_draw rewards

    The strategy should prioritize using non-carryover resources first
    (special_draws, then current_banner_draws) before using normal_draws.
    """

    normal_draws: int = Field(
        default=0,
        description="Draws that can be used at any banner and carry over",
    )
    current_banner_draws: int = Field(
        default=0,
        description="Draws only usable at current banner, don't carry over",
    )
    special_draws: int = Field(
        default=0,
        description="Special draws only usable at current banner, don't carry over",
    )

    def total_available(self) -> int:
        """Get total available draws (all types combined)."""
        return self.normal_draws + self.current_banner_draws + self.special_draws

    def has_special_draws(self) -> bool:
        """Check if there are special draws available."""
        return self.special_draws > 0

    def has_current_banner_draws(self) -> bool:
        """Check if there are current banner draws available."""
        return self.current_banner_draws > 0

    def has_normal_draws(self) -> bool:
        """Check if there are normal draws available."""
        return self.normal_draws > 0

    def consume_special_draws(self, amount: int) -> int:
        """Consume special draws up to the available amount.

        Args:
            amount: Number of special draws to consume.

        Returns:
            Actual number of special draws consumed.
        """
        consumed = min(amount, self.special_draws)
        self.special_draws -= consumed
        return consumed

    def consume_current_banner_draws(self, amount: int) -> int:
        """Consume current banner draws up to the available amount.

        Args:
            amount: Number of current banner draws to consume.

        Returns:
            Actual number of current banner draws consumed.
        """
        consumed = min(amount, self.current_banner_draws)
        self.current_banner_draws -= consumed
        return consumed

    def consume_normal_draws(self, amount: int) -> int:
        """Consume normal draws up to the available amount.

        Args:
            amount: Number of normal draws to consume.

        Returns:
            Actual number of normal draws consumed.
        """
        consumed = min(amount, self.normal_draws)
        self.normal_draws -= consumed
        return consumed

    def add_normal_draws(self, amount: int) -> None:
        """Add normal draws (from config gains, inherited rewards, etc.)."""
        self.normal_draws += amount

    def add_current_banner_draws(self, amount: int) -> None:
        """Add current banner draws (from previous banner's next_banner rewards)."""
        self.current_banner_draws += amount

    def add_special_draws(self, amount: int) -> None:
        """Add special draws (from special_draw rewards)."""
        self.special_draws += amount

    def prepare_for_next_banner(self, next_banner_reward: int) -> "BannerResource":
        """Create a new BannerResource for the next banner.

        - normal_draws carry over as-is
        - current_banner_draws are reset (don't carry over)
        - special_draws are reset (don't carry over)
        - next_banner_reward becomes the new current_banner_draws

        Args:
            next_banner_reward: Draws rewarded for the next banner.

        Returns:
            New BannerResource for the next banner.
        """
        return BannerResource(
            normal_draws=self.normal_draws,
            current_banner_draws=next_banner_reward,
            special_draws=0,
        )


class DrawStrategy(BaseModel):
    """Configuration for automated drawing strategy on banners.

    Defines rules for when to draw, how many draws to do, and when to stop.
    Used by the simulation to determine drawing behavior on each banner.
    """

    name: str = Field("Strategy_Placeholder", description="Name of the strategy")
    always_single_draw: bool = Field(
        default=False, description="Always do a single draw instead of 10 draws"
    )
    single_draw_after: int = Field(
        default=0, description="Start single draw after accumulated x draws"
    )
    skip_banner_threshold: int = Field(
        default=0, description="Minimum amount of draws left to skip the current banner"
    )
    min_draws_after_main: list[tuple[int, int]] = Field(
        default=[],
        description="After getting the main operator, if current draw count is less than threshold (first value) continue drawing until reaching this threshold (second value)",
    )
    min_draws_after_pity: list[tuple[int, int]] = Field(
        default=[],
        description="After reaching pity but missing the main operator, if current draw count is less than threshold (first value) continue drawing until reaching this threshold (second value)",
    )
    min_draws_per_banner: int = Field(
        default=0,
        description="Minimum draws to each banner, skip_banner_threshold wins over this",
    )
    max_draws_per_banner: int = Field(
        default=0,
        description="Maximum draws per banner (0 means no limit)",
    )
    stop_on_main: bool = Field(
        default=False,
        description="Stop drawing immediately after getting the main operator",
    )
    pay: bool = Field(
        default=False,
        description="Pay when the rules cannot be satisfied (i.e. gain extra draws to fulfill the rules above)",
    )


class Player(BaseModel):
    """Represents a player's operator collection (box) across simulation runs.

    Tracks all operators obtained, their potential (number of copies), and
    statistics about when each operator was first drawn. Used to aggregate
    results across multiple simulation iterations.
    """

    operators: dict[int, dict[str, Operator]] = Field(
        {}, description="Operators in the box by rarity"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for rarity in [4, 5, 6]:
            if rarity not in self.operators.keys():
                self.operators[rarity] = {}

    def _add_operator_to_box(
        self, operator: Operator, banner_draw: int, is_special: bool
    ):
        """Add operator to box.

        Args:
            operator: The operator drawn
            banner_draw: The draw count within the CURRENT banner (for fair comparison)
            is_special: Whether this was a special draw
        """
        if operator.name in self.operators[operator.rarity].keys():
            # Existing operator - just add potential, first draw already recorded
            self.operators[operator.rarity][operator.name].potential += 1
        else:
            # New operator - record first draw stats using per-banner draw count
            new_op = operator.model_copy(deep=True)
            new_op.potential = 1
            new_op.first_draw_total = banner_draw
            new_op.first_draw_count = 1
            # Record which bucket this first draw falls into
            # Use -1 as special bucket for special draws
            if is_special:
                new_op.draw_buckets = {-1: 1}
            else:
                bucket = (banner_draw // 10) * 10
                new_op.draw_buckets = {bucket: 1}
            self.operators[operator.rarity][operator.name] = new_op
        self.operators[operator.rarity][operator.name].drawn_by_special_count += (
            1 if is_special else 0
        )

    def draw(
        self,
        banner: Banner,
        repeat: int,
        is_special_draw: bool = False,
    ) -> tuple[int, int, bool, bool]:
        """Draw from banner and add operators to box.

        Args:
            banner: The banner to draw from
            repeat: Number of draws to perform
            is_special_draw: Whether these are special draws

        Returns:
            Tuple of [special draws rewarded, next banner draws rewarded, got main operator, got pity without main]
        """
        special_draw_reward_total = 0
        next_banner_reward_total = 0
        got_main = False
        got_pity_without_main = False
        while repeat > 0:
            repeat -= 1
            result: DrawResult = banner.draw(is_special_draw=is_special_draw)
            got_main = got_main or result.got_main
            # Track if pity was triggered but main was not obtained
            if result.triggered_pity and not result.got_main:
                got_pity_without_main = True
            # Use draws_accumulated (excludes special draws) for fair comparison
            # This matches the definitive draw guarantee which is based on non-special draws
            for operator in result.reward.operators:
                self._add_operator_to_box(
                    operator=operator,
                    banner_draw=banner.draws_accumulated,
                    is_special=is_special_draw,
                )
            # Add potential reward as extra copies of main operator
            if result.reward.potential > 0 and banner.main_operator:
                for _ in range(result.reward.potential):
                    self._add_operator_to_box(
                        operator=banner.main_operator,
                        banner_draw=banner.draws_accumulated,
                        is_special=is_special_draw,
                    )
            special_draw_reward_total += result.reward.special_draws
            next_banner_reward_total += result.reward.next_banner_draws
        return (
            special_draw_reward_total,
            next_banner_reward_total,
            got_main,
            got_pity_without_main,
        )

    def merge(self, other_player: "Player"):
        """Merge another player's operators into this player.

        All fields are summed:
        - potential: sum
        - first_draw_total: sum (for computing average later)
        - first_draw_count: sum
        - drawn_by_special_count: sum
        - draw_buckets: merge by summing counts per bucket
        """
        for rarity in other_player.operators:
            if rarity not in self.operators:
                self.operators[rarity] = {}
            for name, other_op in other_player.operators[rarity].items():
                if name in self.operators[rarity]:
                    existing_op = self.operators[rarity][name]
                    # Sum all counts
                    existing_op.potential += other_op.potential
                    existing_op.first_draw_total += other_op.first_draw_total
                    existing_op.first_draw_count += other_op.first_draw_count
                    existing_op.drawn_by_special_count += (
                        other_op.drawn_by_special_count
                    )
                    # Merge draw buckets
                    for bucket, count in other_op.draw_buckets.items():
                        existing_op.draw_buckets[bucket] = (
                            existing_op.draw_buckets.get(bucket, 0) + count
                        )
                else:
                    # New operator, just copy it
                    self.operators[rarity][name] = other_op.model_copy(deep=True)


class Config(BaseModel):
    """Configuration for the simulation's resource economy.

    Defines how many draws the player starts with and how many they gain
    per banner period.
    """

    initial_draws: int = Field(
        default=0, description="Number of draws player get initially"
    )
    draws_gain_per_banner: int = Field(
        default=0,
        description="Number of draws player get per each banner (carries over)",
    )
    draws_gain_this_banner: int = Field(
        default=0,
        description="Number of draws gained per banner that can only be used on that banner (does not carry over)",
    )


class Run(BaseModel):
    """Main simulation runner that executes gacha simulations across multiple banners.

    Manages the simulation state, applies drawing strategies to each banner,
    and aggregates results across multiple iterations.

    Attributes:
        paid_draws: Total number of paid pulls across all simulation runs.
        total_draws: Total number of all draws (paid + free) across all runs.
        config: Economy configuration for draws available.
        banner_strategies: Per-banner strategy overrides, keyed by banner name.
        banners: List of banners to simulate drawing on, in order.
        repeat: Number of simulation iterations to run.
    """

    paid_draws: int = Field(
        default=0, description="Total number of paid pulls in this run"
    )
    total_draws: int = Field(
        default=0, description="Total number of all draws in this run"
    )
    config: Optional[Config] = Field(
        default=None, description="Default run configuration"
    )
    banner_strategies: dict[str, DrawStrategy] = Field(
        default={},
        description="Per-banner draw strategy overrides, keyed by banner name",
    )
    banners: list[Banner] = Field(default=[], description="All banners to draw from")
    repeat: int = Field(default=1, description="Total number of simulations to run")

    def _get_draw_strategy_for_banner(self, banner: Banner) -> Optional[DrawStrategy]:
        """Get the draw strategy configured for a specific banner.

        Args:
            banner: The banner to get the strategy for.

        Returns:
            The DrawStrategy for this banner, or None if not configured.
        """
        if banner.name in self.banner_strategies:
            return self.banner_strategies[banner.name]
        return None

    def _should_continue_drawing(
        self,
        resource: BannerResource,
        banner: Banner,
        got_main: bool,
        got_pity_without_main: bool = False,
    ) -> bool:
        """Determine if we should continue drawing based on strategy rules.

        Args:
            resource: The current banner resource state.
            banner: The banner being drawn on.
            got_main: Whether the main operator has been obtained.
            got_pity_without_main: Whether pity was triggered without getting main.

        Returns:
            True if we should continue drawing, False to stop.
        """
        strategy = self._get_draw_strategy_for_banner(banner)
        if not strategy:
            return False

        draws_accumulated = banner.draws_accumulated
        # For threshold checks, only count normal_draws (carryover resources)
        # current_banner_draws and special_draws don't count for skip threshold
        normal_available = resource.normal_draws

        # Check max_draws_per_banner - if we've reached the max, stop drawing
        if (
            strategy.max_draws_per_banner > 0
            and draws_accumulated >= strategy.max_draws_per_banner
        ):
            return False

        # Check stop_on_main - if we got main and this flag is set, stop immediately
        if strategy.stop_on_main and got_main:
            return False

        # Check skip_banner_threshold - only considers normal_draws (carryover)
        # If normal draws < threshold and can't pay, skip this banner
        # But only if we've met min_draws_per_banner
        if not strategy.pay and normal_available < strategy.skip_banner_threshold:
            if draws_accumulated >= strategy.min_draws_per_banner:
                return False

        # If we got the main operator, check min_draws_after_main rules
        if got_main:
            target_draws = strategy.min_draws_per_banner
            for threshold, target in strategy.min_draws_after_main:
                if draws_accumulated >= threshold:
                    target_draws = max(target_draws, target)

            if draws_accumulated >= target_draws:
                return False

        # If we hit pity but missed the main operator, check min_draws_after_pity rules
        if got_pity_without_main and not got_main:
            target_draws = strategy.min_draws_per_banner
            for threshold, target in strategy.min_draws_after_pity:
                if draws_accumulated >= threshold:
                    target_draws = max(target_draws, target)

            if draws_accumulated >= target_draws:
                return False

        # Check if we have any draws available (or can pay)
        total_available = resource.total_available()
        if total_available <= 0 and not strategy.pay:
            return False

        return True

    def _get_draw_amount(
        self,
        resource: BannerResource,
        banner: Banner,
    ) -> int:
        """Determine how many draws to perform (1 or 10).

        Args:
            resource: The current banner resource state.
            banner: The banner being drawn on.

        Returns:
            Number of draws to perform (1 or 10).
        """
        strategy = self._get_draw_strategy_for_banner(banner)
        if not strategy:
            return 1

        draws_accumulated = banner.draws_accumulated
        total_available = resource.total_available()

        if strategy.always_single_draw:
            return 1
        elif (
            strategy.single_draw_after > 0
            and draws_accumulated >= strategy.single_draw_after
        ):
            return 1
        elif total_available >= 10 or strategy.pay:
            return 10
        else:
            return 1

    async def run_simulation_async(
        self,
        player: Optional[Player] = None,
        yield_every: int = 100,
        progress_callback=None,
    ) -> Player:
        """Run the gacha simulation asynchronously across all configured banners.

        Executes multiple simulation iterations, drawing on each banner according
        to its configured strategy. Yields control periodically to allow UI updates.

        The simulation uses BannerResource to track three types of draws:
        - normal_draws: Carry over between banners, used last
        - current_banner_draws: From previous banner's next_banner rewards, used second
        - special_draws: From special_draw rewards, used first (as special draws)

        The strategy prioritizes exhausting non-carryover resources first:
        1. Special draws (don't carry over, count as special draws)
        2. Current banner draws (don't carry over, count as normal draws)
        3. Normal draws (carry over to next banner)

        Args:
            player: Optional Player instance to merge results into. If None, creates new.
            yield_every: Number of iterations between yielding control to event loop.
            progress_callback: Optional callback(current, total) to report progress.

        Returns:
            Player instance with aggregated results from all simulation iterations.
        """
        if not player:
            player = Player()

        # Reset counters at the start of simulation
        self.paid_draws = 0
        self.total_draws = 0
        total = self.repeat
        repeat = total
        iteration = 0
        while repeat > 0:
            player_iter = Player()
            repeat -= 1
            iteration += 1

            # Initialize resource for the first banner
            resource = BannerResource(
                normal_draws=0 if not self.config else self.config.initial_draws,
                current_banner_draws=0,
                special_draws=0,
            )
            inherited_reward_states: dict[RewardType, int] = {}

            for banner in self.banners:
                # Reset banner state for this simulation iteration
                banner.reset(inherited_reward_states=inherited_reward_states)

                # Track rewards earned during this banner for next banner
                next_banner_reward_total = 0
                got_main = False
                got_pity_without_main = False

                # Add per-banner draw gains
                if self.config:
                    # draws_gain_per_banner goes to normal_draws (carries over)
                    resource.add_normal_draws(self.config.draws_gain_per_banner)
                    # draws_gain_this_banner goes to current_banner_draws (doesn't carry over)
                    resource.add_current_banner_draws(
                        self.config.draws_gain_this_banner
                    )

                # Main draw loop - prioritize non-carryover resources
                while True:
                    # Priority 1: Use special_draws first (they don't carry over)
                    # Special draws are used as is_special_draw=True
                    if resource.has_special_draws():
                        # Use all special draws at once (they come in multiples of reward count)
                        special_to_use = resource.special_draws
                        resource.consume_special_draws(special_to_use)
                        special, reward, drew_main, pity_miss = player_iter.draw(
                            banner=banner,
                            repeat=special_to_use,
                            is_special_draw=True,
                        )
                        # Special draws can reward more special draws and next banner draws
                        resource.add_special_draws(special)
                        next_banner_reward_total += reward
                        got_main = got_main or drew_main
                        got_pity_without_main = got_pity_without_main or pity_miss
                        continue

                    # Priority 2: Use current_banner_draws (they don't carry over)
                    # These are normal draws, not special draws
                    if resource.has_current_banner_draws():
                        # Use all current banner draws
                        current_to_use = resource.current_banner_draws
                        resource.consume_current_banner_draws(current_to_use)
                        special, reward, drew_main, pity_miss = player_iter.draw(
                            banner=banner,
                            repeat=current_to_use,
                            is_special_draw=False,
                        )
                        resource.add_special_draws(special)
                        next_banner_reward_total += reward
                        got_main = got_main or drew_main
                        got_pity_without_main = got_pity_without_main or pity_miss
                        continue

                    # Priority 3: Check if we should continue with normal draws
                    if not self._should_continue_drawing(
                        resource=resource,
                        banner=banner,
                        got_main=got_main,
                        got_pity_without_main=got_pity_without_main,
                    ):
                        break

                    # Determine draw amount (1 or 10)
                    draw_amount = self._get_draw_amount(
                        resource=resource, banner=banner
                    )

                    # Consume from normal_draws, pay if needed
                    consumed = resource.consume_normal_draws(draw_amount)
                    shortfall = draw_amount - consumed
                    if shortfall > 0:
                        # Need to pay for the shortfall
                        self.paid_draws += shortfall

                    # Draw the desired amount
                    special, reward, drew_main, pity_miss = player_iter.draw(
                        banner=banner,
                        repeat=draw_amount,
                        is_special_draw=False,
                    )
                    resource.add_special_draws(special)
                    next_banner_reward_total += reward
                    got_main = got_main or drew_main
                    got_pity_without_main = got_pity_without_main or pity_miss

                # Track total draws for this banner
                self.total_draws += banner.draws_accumulated_total

                # Prepare resource for next banner:
                # - normal_draws carry over
                # - current_banner_draws reset, replaced by this banner's next_banner_reward
                # - special_draws reset
                resource = resource.prepare_for_next_banner(next_banner_reward_total)
                inherited_reward_states = banner.get_inherited_reward_states()

            # Finish one iteration. Merge the box.
            player.merge(other_player=player_iter)

            # Yield control to event loop periodically and report progress
            if iteration % yield_every == 0:
                if progress_callback:
                    progress_callback(iteration, total)
                await asyncio.sleep(0)

        # Final progress update
        if progress_callback:
            progress_callback(total, total)

        return player
