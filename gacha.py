import asyncio
from typing import Optional

from pydantic import BaseModel, Field

from banner import Banner, DrawResult, Operator, RewardType


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
        default=0, description="Number of draws player get per each banner"
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

    def _get_next_draw_amount(
        self,
        current_available: int,
        banner: Banner,
        got_main: bool,
        got_pity_without_main: bool = False,
    ) -> int:
        """Determine the next draw amount based on the banner's strategy rules.

        Evaluates all strategy rules to decide whether to continue drawing and
        how many draws to perform (1 or 10).

        Args:
            current_available: Number of draws currently available to the player.
            banner: The banner being drawn on.
            got_main: Whether the main operator has been obtained.
            got_pity_without_main: Whether pity was triggered without getting main.

        Returns:
            Number of draws to perform (1, 10, or 0 to stop drawing).
        """
        strategy = self._get_draw_strategy_for_banner(banner)
        if not strategy:
            return 0

        draws_accumulated = banner.draws_accumulated

        # Check max_draws_per_banner - if we've reached the max, stop drawing
        if (
            strategy.max_draws_per_banner > 0
            and draws_accumulated >= strategy.max_draws_per_banner
        ):
            return 0

        # Check stop_on_main - if we got main and this flag is set, stop immediately
        if strategy.stop_on_main and got_main:
            return 0

        # Check skip_banner_threshold - if available draws < threshold, skip this banner
        # But only if we can't pay or haven't met min_draws_per_banner yet
        if not strategy.pay and current_available < strategy.skip_banner_threshold:
            # Check if we've met minimum draws requirement
            if draws_accumulated >= strategy.min_draws_per_banner:
                return 0

        # If we got the main operator, check min_draws_after_main rules
        if got_main:
            # Default: continue to min_draws_per_banner
            target_draws = strategy.min_draws_per_banner
            for threshold, target in strategy.min_draws_after_main:
                if draws_accumulated >= threshold:
                    target_draws = max(target_draws, target)

            # If we've reached the target, stop drawing
            if draws_accumulated >= target_draws:
                return 0

        # If we hit pity but missed the main operator, check min_draws_after_pity rules
        if got_pity_without_main and not got_main:
            # Default: continue to min_draws_per_banner
            target_draws = strategy.min_draws_per_banner
            for threshold, target in strategy.min_draws_after_pity:
                if draws_accumulated >= threshold:
                    target_draws = max(target_draws, target)

            # If we've reached the target, stop drawing
            if draws_accumulated >= target_draws:
                return 0

        # Check if we have draws available (or can pay)
        if current_available <= 0 and not strategy.pay:
            return 0

        # Determine draw amount (1 or 10)
        if strategy.always_single_draw:
            return 1
        elif (
            strategy.single_draw_after > 0
            and draws_accumulated >= strategy.single_draw_after
        ):
            return 1
        elif current_available >= 10 or strategy.pay:
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

        The simulation:
        1. Iterates through banners in order
        2. Applies drawing strategy to determine draw behavior
        3. Tracks pity/definitive/potential counters with inheritance between banners
        4. Aggregates operator statistics across all iterations

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
            special_draw = 0
            reward_draw = 0
            available_draws = 0 if not self.config else self.config.initial_draws
            repeat -= 1
            iteration += 1
            inherited_reward_states: dict[RewardType, int] = {}
            for banner in self.banners:
                # Reset banner state for this simulation iteration
                banner.reset(inherited_reward_states=inherited_reward_states)
                # For each banner, first clear out the special draw as they don't carry over.
                special_draw = 0
                reward_for_next = 0
                got_main = False
                got_pity_without_main = False
                # Also add the draws for each banner as available
                available_draws += (
                    0 if not self.config else self.config.draws_gain_per_banner
                )
                # Determine whether to draw or not.
                while True:
                    if special_draw > 0:
                        special, reward, drew_main, pity_miss = player_iter.draw(
                            banner=banner,
                            repeat=special_draw,
                            is_special_draw=True,
                        )
                        special_draw = 0
                        special_draw += special
                        reward_for_next += reward
                        got_main = got_main or drew_main
                        got_pity_without_main = got_pity_without_main or pity_miss
                        continue
                    if reward_draw > 0:
                        special, reward, drew_main, pity_miss = player_iter.draw(
                            banner=banner,
                            repeat=reward_draw,
                            is_special_draw=False,
                        )
                        reward_draw = 0
                        special_draw += special
                        reward_for_next += reward
                        got_main = got_main or drew_main
                        got_pity_without_main = got_pity_without_main or pity_miss
                        continue
                    draw_amount = self._get_next_draw_amount(
                        current_available=available_draws,
                        banner=banner,
                        got_main=got_main,
                        got_pity_without_main=got_pity_without_main,
                    )
                    if draw_amount <= 0:
                        break
                    # If the draw amount exceeds current available, pay for that.
                    available_draws -= draw_amount
                    if available_draws < 0:
                        self.paid_draws += -available_draws
                        available_draws = 0
                    # Draw the desired amount
                    special, reward, drew_main, pity_miss = player_iter.draw(
                        banner=banner,
                        repeat=draw_amount,
                        is_special_draw=False,
                    )
                    special_draw += special
                    reward_for_next += reward
                    got_main = got_main or drew_main
                    got_pity_without_main = got_pity_without_main or pity_miss
                    continue
                # Continue to next banner, track total draws for this banner
                self.total_draws += banner.draws_accumulated_total
                reward_draw = reward_for_next
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
