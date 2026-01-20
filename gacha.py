"""Gacha simulation engine."""

import asyncio
from typing import Optional

from pydantic import BaseModel, Field

from banner import (
    Banner,
    BannerTemplate,
    DrawResult,
    Operator,
    RewardType,
    create_next_banner,
)
from strategy import DrawStrategy, EvaluationContext


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
        new_non_up_counts_as_miss: bool = True,
    ) -> tuple[int, int, bool, bool, bool, bool, int]:
        """Draw from banner and add operators to box.

        Args:
            banner: The banner to draw from
            repeat: Number of draws to perform
            is_special_draw: Whether these are special draws
            new_non_up_counts_as_miss: Whether getting a NEW non-UP main operator
                (from another banner) counts as a miss. If False, getting a new
                main operator from another banner doesn't count as miss (only
                dupes count). Regular pool operators always count as miss.

        Returns:
            Tuple of [special draws rewarded, next banner draws rewarded,
                      got main operator, got pity without main,
                      got highest rarity, got highest rarity but not main,
                      main operator copies obtained]
        """
        special_draw_reward_total = 0
        next_banner_reward_total = 0
        got_main = False
        got_pity_without_main = False
        got_highest_rarity = False
        got_highest_rarity_but_not_main = False
        main_copies = 0
        highest_rarity = max(banner.template.rarities)
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
                # Check if operator is already owned BEFORE adding to box
                # This is needed to determine if it's a "new" operator for miss counting
                is_already_owned = operator.name in self.operators.get(
                    operator.rarity, {}
                )

                self._add_operator_to_box(
                    operator=operator,
                    banner_draw=banner.draws_accumulated,
                    is_special=is_special_draw,
                )
                # Track highest rarity obtained
                if operator.rarity == highest_rarity:
                    got_highest_rarity = True
                    # Track if highest rarity but not main (歪/miss)
                    if (
                        not banner.main_operator
                        or operator.name != banner.main_operator.name
                    ):
                        # Determine if this should count as a miss:
                        # - Regular pool operators (banner=None) always count as miss
                        # - Main operators from other banners:
                        #   - If new_non_up_counts_as_miss is True: always count as miss
                        #   - If False: only count as miss if already owned (dupe)
                        is_main_operator = operator.banner is not None
                        if not is_main_operator:
                            # Regular pool operator - always counts as miss
                            got_highest_rarity_but_not_main = True
                        elif new_non_up_counts_as_miss or is_already_owned:
                            # Main operator from another banner - check config
                            got_highest_rarity_but_not_main = True
                # Track main operator copies
                if banner.main_operator and operator.name == banner.main_operator.name:
                    main_copies += 1
            # Add potential reward as extra copies of main operator
            if result.reward.potential > 0 and banner.main_operator:
                for _ in range(result.reward.potential):
                    self._add_operator_to_box(
                        operator=banner.main_operator,
                        banner_draw=banner.draws_accumulated,
                        is_special=is_special_draw,
                    )
                main_copies += result.reward.potential
            special_draw_reward_total += result.reward.special_draws
            next_banner_reward_total += result.reward.next_banner_draws
        return (
            special_draw_reward_total,
            next_banner_reward_total,
            got_main,
            got_pity_without_main,
            got_highest_rarity,
            got_highest_rarity_but_not_main,
            main_copies,
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
    draws_gain_per_banner_start_at: int = Field(
        default=1,
        description="Banner index (1-based) from which draws_gain_per_banner starts applying",
    )
    draws_gain_this_banner: int = Field(
        default=0,
        description="Number of draws gained per banner that can only be used on that banner (does not carry over)",
    )
    new_non_up_counts_as_miss: bool = Field(
        default=True,
        description="Whether getting a NEW (not already owned) non-UP main operator (from another banner) counts as a miss (歪)",
    )


class Run(BaseModel):
    """Main simulation runner that executes gacha simulations across multiple banners.

    Manages the simulation state, applies drawing strategies to each banner,
    and aggregates results across multiple iterations.

    Attributes:
        paid_draws: Total number of paid pulls across all simulation runs.
        total_draws: Total number of all draws (paid + free) across all runs.
        total_main_copies: Total main operator copies obtained across all runs.
        total_highest_rarity_not_main: Total times highest rarity was obtained but not main (歪).
        config: Economy configuration for draws available.
        banner_strategies: Per-banner strategy overrides, keyed by banner name.
        banners: List of banners to simulate drawing on, in order.
        repeat: Number of simulation iterations to run.
        auto_banner_template: Template for auto-generated banners (default: last banner's template).
        auto_banner_strategy: Strategy for auto-generated banners (optional).
        auto_banner_count: Number of banners to auto-generate after exhausting given banners (0 = no auto).
    """

    paid_draws: int = Field(
        default=0, description="Total number of paid pulls in this run"
    )
    total_draws: int = Field(
        default=0, description="Total number of all draws in this run"
    )
    total_main_copies: int = Field(
        default=0, description="Total main operator copies obtained across all runs"
    )
    total_highest_rarity_not_main: int = Field(
        default=0,
        description="Total times highest rarity was obtained but not main (歪)",
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
    auto_banner_template: Optional[BannerTemplate] = Field(
        default=None,
        description="Template for auto-generated banners (default: last banner's template)",
    )
    auto_banner_strategy: Optional[DrawStrategy] = Field(
        default=None,
        description="Strategy for auto-generated banners",
    )
    auto_banner_count: int = Field(
        default=0,
        ge=0,
        description="Number of banners to auto-generate after exhausting given banners (0 = no auto)",
    )
    auto_banner_default_operators: list[Operator] = Field(
        default=[],
        description="Default operators to include in auto-generated banners",
    )

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

    def _create_auto_banner(
        self,
        index: int,
        template: BannerTemplate,
        previous_banners: list[Banner],
    ) -> Banner:
        """Create an auto-generated banner using the shared banner creation routine.

        Args:
            index: The 1-based index of the auto-generated banner.
            template: The template to use for the banner.
            previous_banners: List of previous banners to inherit main operators from.

        Returns:
            A new Banner instance with properly populated operators.
        """
        return create_next_banner(
            template=template,
            default_operators=self.auto_banner_default_operators,
            previous_banners=previous_banners,
            banner_name=f"自动池{index}",
        )

    def _should_continue_drawing(
        self,
        resource: BannerResource,
        banner: Banner,
        banner_index: int,
        got_main: bool,
        got_highest_rarity: bool = False,
        got_highest_rarity_but_not_main: bool = False,
        got_pity_without_main: bool = False,
        current_potential: int = 0,
        pity_counter: int = 0,
        definitive_draw_counter: int = 0,
    ) -> bool:
        """Determine if we should continue drawing based on strategy rules.

        Args:
            resource: The current banner resource state.
            banner: The banner being drawn on.
            banner_index: The 0-based index of the current banner.
            got_main: Whether the main operator has been obtained.
            got_highest_rarity: Whether any highest rarity has been obtained this banner.
            got_highest_rarity_but_not_main: Whether highest rarity obtained but not main.
            got_pity_without_main: Whether pity was triggered without getting main.
            current_potential: Current number of copies of main operator obtained.
            pity_counter: Current pity counter (draws since last highest rarity).
            definitive_draw_counter: Current definitive draw counter.

        Returns:
            True if we should continue drawing, False to stop.
        """
        strategy = self._get_draw_strategy_for_banner(banner)
        if not strategy:
            return False

        # Build evaluation context
        context = EvaluationContext(
            draws_accumulated=banner.draws_accumulated,
            normal_draws=resource.normal_draws,
            got_main=got_main,
            got_highest_rarity=got_highest_rarity,
            got_highest_rarity_but_not_main=got_highest_rarity_but_not_main,
            got_pity_without_main=got_pity_without_main,
            current_potential=current_potential,
            banner_index=banner_index,
            pity_counter=pity_counter,
            definitive_draw_counter=definitive_draw_counter,
        )

        # Build strategy registry for delegation lookup
        strategy_registry = {s.name: s for s in self.banner_strategies.values()}

        return strategy.should_continue_drawing(context, strategy_registry)

    def _get_draw_amount(
        self,
        resource: BannerResource,
        banner: Banner,
        banner_index: int,
        pity_counter: int = 0,
        definitive_draw_counter: int = 0,
    ) -> int:
        """Determine how many draws to perform (1 or 10).

        Args:
            resource: The current banner resource state.
            banner: The banner being drawn on.
            banner_index: The 0-based index of the current banner.
            pity_counter: Current pity counter (draws since last highest rarity).
            definitive_draw_counter: Current definitive draw counter.

        Returns:
            Number of draws to perform (1 or 10).
        """
        strategy = self._get_draw_strategy_for_banner(banner)
        if not strategy:
            return 1

        # Build evaluation context
        context = EvaluationContext(
            draws_accumulated=banner.draws_accumulated,
            normal_draws=resource.normal_draws,
            banner_index=banner_index,
            pity_counter=pity_counter,
            definitive_draw_counter=definitive_draw_counter,
        )

        # Build strategy registry for delegation lookup
        strategy_registry = {s.name: s for s in self.banner_strategies.values()}

        return strategy.get_draw_amount(
            context, resource.total_available(), strategy_registry
        )

    def _get_effective_pay(
        self,
        resource: BannerResource,
        banner: Banner,
        banner_index: int,
        pity_counter: int = 0,
        definitive_draw_counter: int = 0,
    ) -> bool:
        """Get effective pay setting considering action override.

        Args:
            resource: The current banner resource state.
            banner: The banner being drawn on.
            banner_index: The 0-based index of the current banner.
            pity_counter: Current pity counter (draws since last highest rarity).
            definitive_draw_counter: Current definitive draw counter.

        Returns:
            True if should pay for draws, False otherwise.
        """
        strategy = self._get_draw_strategy_for_banner(banner)
        if not strategy:
            return False

        # Build evaluation context
        context = EvaluationContext(
            draws_accumulated=banner.draws_accumulated,
            normal_draws=resource.normal_draws,
            banner_index=banner_index,
            pity_counter=pity_counter,
            definitive_draw_counter=definitive_draw_counter,
        )

        # Build strategy registry for delegation lookup
        strategy_registry = {s.name: s for s in self.banner_strategies.values()}

        return strategy.get_effective_pay(context, strategy_registry)

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
        self.total_main_copies = 0
        self.total_highest_rarity_not_main = 0
        repeat_count = self.repeat
        repeat_remaining = repeat_count

        # Determine template for auto-generated banners
        auto_template: Optional[BannerTemplate] = self.auto_banner_template
        if auto_template is None and self.auto_banner_count > 0 and self.banners:
            # Default to last banner's template
            auto_template = self.banners[-1].template

        # Total number of banners (original + auto-generated)
        total_banner_count = len(self.banners) + self.auto_banner_count

        # Progress tracking: count by banners across all repeats
        total_banners_to_process = repeat_count * total_banner_count
        banners_processed = 0

        while repeat_remaining > 0:
            player_iter = Player()
            repeat_remaining -= 1

            # Initialize resource for the first banner
            resource = BannerResource(
                normal_draws=0 if not self.config else self.config.initial_draws,
                current_banner_draws=0,
                special_draws=0,
            )
            inherited_reward_states: dict[RewardType, int] = {}

            # Track current auto banner for cleanup and previous banners for inheritance
            current_auto_banner: Optional[Banner] = None
            # Track banners processed in this iteration for main operator inheritance
            processed_banners: list[Banner] = []

            banner_index = 0
            while banner_index < total_banner_count:
                # Get or create the banner for this index
                if banner_index < len(self.banners):
                    # Use existing banner
                    banner = self.banners[banner_index]
                else:
                    # Create auto banner on-demand, passing all previous banners
                    # (original banners + previously created auto banners)
                    auto_index = banner_index - len(self.banners) + 1
                    previous_for_inherit = (
                        self.banners + processed_banners[len(self.banners) :]
                    )
                    current_auto_banner = self._create_auto_banner(
                        auto_index, auto_template, previous_for_inherit  # type: ignore
                    )
                    banner = current_auto_banner
                    # Register strategy for auto-generated banner
                    if self.auto_banner_strategy:
                        self.banner_strategies[banner.name] = self.auto_banner_strategy
                # Reset banner state for this simulation iteration
                banner.reset(inherited_reward_states=inherited_reward_states)

                # Track rewards earned during this banner for next banner
                next_banner_reward_total = 0
                got_main = False
                got_highest_rarity = False
                got_highest_rarity_but_not_main = False
                got_pity_without_main = False
                current_potential = 0  # Number of main operator copies obtained

                # Add per-banner draw gains
                if self.config:
                    # draws_gain_per_banner goes to normal_draws (carries over)
                    # Only apply if banner_index >= start_at (1-based)
                    if banner_index + 1 >= self.config.draws_gain_per_banner_start_at:
                        resource.add_normal_draws(self.config.draws_gain_per_banner)
                    # draws_gain_this_banner goes to current_banner_draws (doesn't carry over)
                    resource.add_current_banner_draws(
                        self.config.draws_gain_this_banner
                    )

                # Check strategy at banner entry (before any draws) to determine
                # if we should use normal_draws on this banner.
                # This check happens once at draws_accumulated=0 for check_once conditions.
                current_pity = banner.reward_states[RewardType.PITY].counter
                current_definitive = banner.reward_states[RewardType.DEFINITIVE].counter
                should_use_normal_draws = self._should_continue_drawing(
                    resource=resource,
                    banner=banner,
                    banner_index=banner_index,
                    got_main=False,
                    got_highest_rarity=False,
                    got_highest_rarity_but_not_main=False,
                    got_pity_without_main=False,
                    current_potential=0,
                    pity_counter=current_pity,
                    definitive_draw_counter=current_definitive,
                )

                # Main draw loop - prioritize non-carryover resources
                while True:
                    # Priority 1: Use special_draws first (they don't carry over)
                    # Special draws are used as is_special_draw=True
                    if resource.has_special_draws():
                        # Use all special draws at once (they come in multiples of reward count)
                        special_to_use = resource.special_draws
                        resource.consume_special_draws(special_to_use)
                        (
                            special,
                            reward,
                            drew_main,
                            pity_miss,
                            drew_hr,
                            drew_hr_not_main,
                            main_copies,
                        ) = player_iter.draw(
                            banner=banner,
                            repeat=special_to_use,
                            is_special_draw=True,
                            new_non_up_counts_as_miss=(
                                self.config.new_non_up_counts_as_miss
                                if self.config
                                else True
                            ),
                        )
                        # Special draws can reward more special draws and next banner draws
                        resource.add_special_draws(special)
                        next_banner_reward_total += reward
                        got_main = got_main or drew_main
                        got_highest_rarity = got_highest_rarity or drew_hr
                        got_highest_rarity_but_not_main = (
                            got_highest_rarity_but_not_main or drew_hr_not_main
                        )
                        got_pity_without_main = got_pity_without_main or pity_miss
                        current_potential += main_copies
                        continue

                    # Priority 2: Use current_banner_draws (they don't carry over)
                    # These are normal draws, not special draws
                    # ALWAYS use these - they would be wasted otherwise
                    if resource.has_current_banner_draws():
                        # Use all current banner draws
                        current_to_use = resource.current_banner_draws
                        resource.consume_current_banner_draws(current_to_use)
                        (
                            special,
                            reward,
                            drew_main,
                            pity_miss,
                            drew_hr,
                            drew_hr_not_main,
                            main_copies,
                        ) = player_iter.draw(
                            banner=banner,
                            repeat=current_to_use,
                            is_special_draw=False,
                            new_non_up_counts_as_miss=(
                                self.config.new_non_up_counts_as_miss
                                if self.config
                                else True
                            ),
                        )
                        resource.add_special_draws(special)
                        next_banner_reward_total += reward
                        got_main = got_main or drew_main
                        got_highest_rarity = got_highest_rarity or drew_hr
                        got_highest_rarity_but_not_main = (
                            got_highest_rarity_but_not_main or drew_hr_not_main
                        )
                        got_pity_without_main = got_pity_without_main or pity_miss
                        current_potential += main_copies
                        continue

                    # Priority 3: Check if we should continue with normal draws
                    # First check the decision made at banner entry (for check_once conditions)
                    if not should_use_normal_draws:
                        break

                    # Get current counters from banner for dynamic checks
                    current_pity = banner.reward_states[RewardType.PITY].counter
                    current_definitive = banner.reward_states[
                        RewardType.DEFINITIVE
                    ].counter
                    # Check dynamic conditions (stop_on_main, max_draws, etc.)
                    if not self._should_continue_drawing(
                        resource=resource,
                        banner=banner,
                        banner_index=banner_index,
                        got_main=got_main,
                        got_highest_rarity=got_highest_rarity,
                        got_highest_rarity_but_not_main=got_highest_rarity_but_not_main,
                        got_pity_without_main=got_pity_without_main,
                        current_potential=current_potential,
                        pity_counter=current_pity,
                        definitive_draw_counter=current_definitive,
                    ):
                        break

                    # Determine draw amount (1 or 10)
                    draw_amount = self._get_draw_amount(
                        resource=resource,
                        banner=banner,
                        banner_index=banner_index,
                        pity_counter=current_pity,
                        definitive_draw_counter=current_definitive,
                    )

                    # Get effective pay setting (considers action override)
                    effective_pay = self._get_effective_pay(
                        resource=resource,
                        banner=banner,
                        banner_index=banner_index,
                        pity_counter=current_pity,
                        definitive_draw_counter=current_definitive,
                    )

                    # Consume from normal_draws, pay if needed
                    consumed = resource.consume_normal_draws(draw_amount)
                    shortfall = draw_amount - consumed
                    if shortfall > 0 and effective_pay:
                        # Need to pay for the shortfall
                        self.paid_draws += shortfall

                    # Draw the desired amount
                    (
                        special,
                        reward,
                        drew_main,
                        pity_miss,
                        drew_hr,
                        drew_hr_not_main,
                        main_copies,
                    ) = player_iter.draw(
                        banner=banner,
                        repeat=draw_amount,
                        is_special_draw=False,
                        new_non_up_counts_as_miss=(
                            self.config.new_non_up_counts_as_miss
                            if self.config
                            else True
                        ),
                    )
                    resource.add_special_draws(special)
                    next_banner_reward_total += reward
                    got_main = got_main or drew_main
                    got_highest_rarity = got_highest_rarity or drew_hr
                    got_highest_rarity_but_not_main = (
                        got_highest_rarity_but_not_main or drew_hr_not_main
                    )
                    got_pity_without_main = got_pity_without_main or pity_miss
                    current_potential += main_copies

                # Track total draws for this banner
                self.total_draws += banner.draws_accumulated_total

                # Track main copies and highest rarity not main (歪) for this banner
                self.total_main_copies += current_potential
                if got_highest_rarity_but_not_main:
                    self.total_highest_rarity_not_main += 1

                # Prepare resource for next banner:
                # - normal_draws carry over
                # - current_banner_draws reset, replaced by this banner's next_banner_reward
                # - special_draws reset
                resource = resource.prepare_for_next_banner(next_banner_reward_total)
                inherited_reward_states = banner.get_inherited_reward_states()

                # Track this banner for main operator inheritance in future auto banners
                processed_banners.append(banner)

                # Clean up auto banner strategy registration to avoid memory buildup
                if current_auto_banner is not None:
                    self.banner_strategies.pop(current_auto_banner.name, None)
                    current_auto_banner = None

                # Move to next banner
                banner_index += 1
                banners_processed += 1

                # Yield control to event loop periodically and report progress
                if banners_processed % yield_every == 0:
                    if progress_callback:
                        progress_callback(banners_processed, total_banners_to_process)
                    await asyncio.sleep(0)

            # Finish one iteration. Merge the box.
            player.merge(other_player=player_iter)

        # Final progress update
        if progress_callback:
            progress_callback(total_banners_to_process, total_banners_to_process)

        return player
