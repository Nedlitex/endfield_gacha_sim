from enum import Enum
from random import random
from typing import Optional

from pydantic import BaseModel, Field


class InheritPolicy(str, Enum):
    """Policy for inheriting state to the next banner"""

    NO_INHERIT = "no_inherit"  # State resets each banner
    INHERIT_TO_NEXT = "inherit_to_next"  # State carries to next banner only
    ALWAYS_INHERIT = "always_inherit"  # State persists across all banners


class RepeatPolicy(str, Enum):
    """Policy for repeating rewards after they've been triggered"""

    NO_REPEAT = "no_repeat"  # Reward can only be obtained once
    ALWAYS_REPEAT = "always_repeat"  # Reward can be obtained multiple times


class ResetCondition(str, Enum):
    """Conditions that can trigger a reward counter reset"""

    NONE = "none"  # Never reset on draw outcome
    ON_HIGHEST_RARITY = "on_highest_rarity"  # Reset when highest rarity is drawn
    ON_MAIN = "on_main"  # Reset when main operator is obtained


class RewardType(str, Enum):
    """Types of rewards that can be triggered during draws"""

    PITY = "pity"  # Pity draw (small guarantee for highest rarity)
    DEFINITIVE = "definitive"  # Definitive draw (big guarantee for main operator)
    POTENTIAL = "potential"  # Potential reward (extra potential for main operator)
    SPECIAL_DRAW = "special_draw"  # Special draw reward for this banner
    NEXT_BANNER_DRAW = "next_banner_draw"  # Draw reward for the next banner


class RewardState(BaseModel):
    """State for tracking a reward counter"""

    counter: int = Field(default=0, description="Current counter value")
    triggered: bool = Field(
        default=False, description="Whether the reward has been triggered"
    )


class RarityProbability(BaseModel):
    """Probability distribution for a single rarity level"""

    rarity: int = Field(..., description="Rarity level (e.g., 4, 5, 6)")
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of drawing this rarity"
    )


class BannerTemplate(BaseModel):
    """Template configuration for a gacha banner system.

    This class defines all the configurable parameters for a gacha banner,
    including probability distributions, pity systems, and reward mechanisms.
    """

    # Basic info
    name: str = Field("Default", description="Name of the banner template")

    # Rarity configuration
    rarities: list[int] = Field(
        default=[4, 5, 6], description="List of available rarity levels"
    )
    default_distribution: list[RarityProbability] = Field(
        default=[
            RarityProbability(rarity=4, probability=0.912),
            RarityProbability(rarity=5, probability=0.08),
            RarityProbability(rarity=6, probability=0.008),
        ],
        description="Default probability distribution for each rarity",
    )

    # Main operator configuration
    main_probability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of drawing the main/featured operator when drawing its rarity",
    )
    inherit_main_from_previous_banners: int = Field(
        default=0,
        ge=0,
        description="Number of previous banners from which main operators are inherited into the pool",
    )

    # Pity draw configuration
    has_pity_draw: bool = Field(
        default=True, description="Whether the pity system is enabled"
    )
    pity_draw_start: int = Field(
        default=66,
        ge=0,
        description="Draw count at which pity probability boost begins",
    )
    pity_draw_limit: int = Field(
        default=80,
        ge=0,
        description="Draw count at which guaranteed highest rarity is obtained",
    )
    pity_rarity_boost_per_draw: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Additional probability added to highest rarity for each draw after pity_draw_start",
    )
    pity_draw_inherit_policy: InheritPolicy = Field(
        default=InheritPolicy.INHERIT_TO_NEXT,
        description="Policy for inheriting pity counter to subsequent banners",
    )
    pity_draw_repeat_policy: RepeatPolicy = Field(
        default=RepeatPolicy.NO_REPEAT,
        description="Policy for whether pity can trigger multiple times per banner",
    )
    pity_reset_condition: ResetCondition = Field(
        default=ResetCondition.ON_HIGHEST_RARITY,
        description="Condition that triggers pity reset (resets counter and marks as triggered)",
    )

    # Definitive draw configuration
    has_definitive_draw: bool = Field(
        default=True,
        description="Whether a guaranteed main operator draw exists at a specific count",
    )
    definitive_draw_count: int = Field(
        default=120,
        ge=0,
        description="Draw count at which the main operator is guaranteed",
    )
    definitive_draw_inherit_policy: InheritPolicy = Field(
        default=InheritPolicy.NO_INHERIT,
        description="Policy for inheriting definitive draw progress to subsequent banners",
    )
    definitive_draw_repeat_policy: RepeatPolicy = Field(
        default=RepeatPolicy.NO_REPEAT,
        description="Policy for whether definitive draw can trigger multiple times",
    )
    definitive_reset_condition: ResetCondition = Field(
        default=ResetCondition.NONE,
        description="Condition that triggers definitive reset (resets counter and marks as triggered)",
    )

    # Potential reward configuration
    has_potential_reward: bool = Field(
        default=True,
        description="Whether extra potential rewards are given at specific draw counts",
    )
    potential_reward_draw: int = Field(
        default=240,
        ge=0,
        description="Draw count interval at which extra potential for main operator is rewarded",
    )
    potential_reward_inherit_policy: InheritPolicy = Field(
        default=InheritPolicy.NO_INHERIT,
        description="Policy for inheriting potential reward progress to subsequent banners",
    )
    potential_reward_repeat_policy: RepeatPolicy = Field(
        default=RepeatPolicy.ALWAYS_REPEAT,
        description="Policy for whether potential rewards can be obtained multiple times",
    )

    # Special draw reward configuration
    special_draw_reward_at: int = Field(
        30, description="Draw count at which special draw reward is given"
    )
    special_draw_reward_count: int = Field(
        10, description="Number of special draws rewarded"
    )
    special_draw_repeat: bool = Field(
        False,
        description="Whether special draw reward can trigger multiple times per banner",
    )

    # Next banner draw reward configuration
    next_banner_draw_reward_at: int = Field(
        60, description="Draw count at which next banner draw reward is given"
    )
    next_banner_draw_reward_count: int = Field(
        10, description="Number of next banner draws rewarded"
    )
    next_banner_draw_repeat: bool = Field(
        False,
        description="Whether next banner draw reward can trigger multiple times per banner",
    )

    def validate_distribution(self) -> bool:
        """Validate that the default distribution sums to 1.0 and covers all rarities."""
        total = sum(rp.probability for rp in self.default_distribution)
        if abs(total - 1.0) > 1e-9:
            return False
        distribution_rarities = {rp.rarity for rp in self.default_distribution}
        return distribution_rarities == set(self.rarities)

    def _get_inherit_policy_description(self, policy: InheritPolicy) -> str:
        """Get Chinese description for inherit policy."""
        descriptions = {
            InheritPolicy.NO_INHERIT: "不继承",
            InheritPolicy.INHERIT_TO_NEXT: "继承至下一期卡池",
            InheritPolicy.ALWAYS_INHERIT: "永久继承",
        }
        return descriptions.get(policy, str(policy))

    def _get_repeat_policy_description(self, policy: RepeatPolicy) -> str:
        """Get Chinese description for repeat policy."""
        descriptions = {
            RepeatPolicy.NO_REPEAT: "不可重复触发",
            RepeatPolicy.ALWAYS_REPEAT: "可重复触发",
        }
        return descriptions.get(policy, str(policy))

    def get_description(self) -> str:
        """Generate a human-readable markdown description of the banner policy in Chinese."""
        lines = []
        lines.append(f"## {self.name}")
        lines.append("")

        # Rarity distribution
        lines.append("### 基础概率分布")
        lines.append(
            f"**稀有度等级:** {', '.join(str(r) + '星' for r in sorted(self.rarities))}"
        )
        lines.append("")
        lines.append("| 稀有度 | 概率 |")
        lines.append("|--------|------|")
        for rp in sorted(self.default_distribution, key=lambda x: x.rarity):
            lines.append(f"| {rp.rarity}星 | {rp.probability * 100:.2f}% |")
        lines.append("")

        # Main operator
        lines.append("### UP干员机制")
        lines.append(
            f"- **UP干员概率:** 抽到对应稀有度时，{self.main_probability * 100:.0f}%为UP干员"
        )
        if self.inherit_main_from_previous_banners > 0:
            lines.append(
                f"- **历史UP继承:** 继承前{self.inherit_main_from_previous_banners}期UP干员进入卡池"
            )
        else:
            lines.append("- **历史UP继承:** 不继承历史UP干员")
        lines.append("")

        # Pity system (小保底)
        highest_rarity = max(self.rarities)
        lines.append("### 小保底机制")
        if self.has_pity_draw:
            lines.append(
                f"- **概率提升起始:** 第{self.pity_draw_start}抽开始提升{highest_rarity}星概率"
            )
            lines.append(
                f"- **每抽提升幅度:** +{self.pity_rarity_boost_per_draw * 100:.0f}%"
            )
            lines.append(
                f"- **小保底:** 第{self.pity_draw_limit}抽必出{highest_rarity}星"
            )
            lines.append(
                f"- **计数继承:** {self._get_inherit_policy_description(self.pity_draw_inherit_policy)}"
            )
            lines.append(
                f"- **重复触发:** {self._get_repeat_policy_description(self.pity_draw_repeat_policy)}"
            )
            if self.pity_reset_condition == ResetCondition.ON_HIGHEST_RARITY:
                lines.append(f"- **重置条件:** 抽到{highest_rarity}星时重置计数")
            elif self.pity_reset_condition == ResetCondition.ON_MAIN:
                lines.append("- **重置条件:** 抽到UP干员时重置计数")
            else:
                lines.append("- **重置条件:** 无")
        else:
            lines.append("*无小保底机制*")
        lines.append("")

        # Definitive draw (大保底)
        lines.append("### 大保底机制")
        if self.has_definitive_draw:
            lines.append(
                f"- **大保底抽数:** 第{self.definitive_draw_count}抽必得UP干员"
            )
            lines.append(
                f"- **计数继承:** {self._get_inherit_policy_description(self.definitive_draw_inherit_policy)}"
            )
            lines.append(
                f"- **重复触发:** {self._get_repeat_policy_description(self.definitive_draw_repeat_policy)}"
            )
            if self.definitive_reset_condition == ResetCondition.ON_MAIN:
                lines.append("- **重置条件:** 抽到UP干员时重置计数")
            elif self.definitive_reset_condition == ResetCondition.ON_HIGHEST_RARITY:
                lines.append(f"- **重置条件:** 抽到{highest_rarity}星时重置计数")
            else:
                lines.append("- **重置条件:** 无")
        else:
            lines.append("*无大保底机制*")
        lines.append("")

        # Potential reward
        lines.append("### 潜能奖励")
        if self.has_potential_reward:
            lines.append(
                f"- **奖励间隔:** 每{self.potential_reward_draw}抽赠送UP干员潜能×1"
            )
            lines.append(
                f"- **计数继承:** {self._get_inherit_policy_description(self.potential_reward_inherit_policy)}"
            )
            lines.append(
                f"- **重复触发:** {self._get_repeat_policy_description(self.potential_reward_repeat_policy)}"
            )
        else:
            lines.append("*无潜能奖励*")
        lines.append("")

        # Special draw reward
        lines.append("### 特殊抽奖励")
        if self.special_draw_reward_at > 0 and self.special_draw_reward_count > 0:
            lines.append(f"- **触发条件:** 累计抽数达到{self.special_draw_reward_at}抽")
            lines.append(f"- **奖励数量:** {self.special_draw_reward_count}次特殊抽")
            lines.append(
                f"- **重复触发:** {'可重复触发' if self.special_draw_repeat else '不可重复触发'}"
            )
        else:
            lines.append("*无特殊抽奖励*")
        lines.append("")

        # Next banner draw reward
        lines.append("### 下期卡池抽奖励")
        if (
            self.next_banner_draw_reward_at > 0
            and self.next_banner_draw_reward_count > 0
        ):
            lines.append(
                f"- **触发条件:** 累计抽数达到{self.next_banner_draw_reward_at}抽"
            )
            lines.append(
                f"- **奖励数量:** {self.next_banner_draw_reward_count}次下期卡池抽"
            )
            lines.append(
                f"- **重复触发:** {'可重复触发' if self.next_banner_draw_repeat else '不可重复触发'}"
            )
        else:
            lines.append("*无下期卡池抽奖励*")

        return "\n".join(lines)


# Default banner template for EndField (终末地)
# Based on the current Banner implementation
EndFieldBannerTemplate = BannerTemplate(
    name="终末地卡池模板",
    # Rarity configuration - 4/5/6 star system
    rarities=[4, 5, 6],
    default_distribution=[
        RarityProbability(rarity=4, probability=0.912),
        RarityProbability(rarity=5, probability=0.08),
        RarityProbability(rarity=6, probability=0.008),
    ],
    # Main operator gets 50% when drawing its rarity
    main_probability=0.5,
    # Historical main operator inheritance from previous 2
    inherit_main_from_previous_banners=2,
    # Pity system: starts at draw 66, guaranteed at 80, +5% per draw
    has_pity_draw=True,
    pity_draw_start=66,
    pity_draw_limit=80,
    pity_rarity_boost_per_draw=0.05,
    pity_draw_inherit_policy=InheritPolicy.ALWAYS_INHERIT,
    pity_draw_repeat_policy=RepeatPolicy.ALWAYS_REPEAT,
    pity_reset_condition=ResetCondition.ON_HIGHEST_RARITY,
    # Definitive draw: guaranteed main operator at 120 draws
    has_definitive_draw=True,
    definitive_draw_count=120,
    definitive_draw_inherit_policy=InheritPolicy.NO_INHERIT,
    definitive_draw_repeat_policy=RepeatPolicy.NO_REPEAT,
    definitive_reset_condition=ResetCondition.ON_MAIN,
    # Potential reward: extra potential every 240 draws
    has_potential_reward=True,
    potential_reward_draw=240,
    potential_reward_inherit_policy=InheritPolicy.NO_INHERIT,
    potential_reward_repeat_policy=RepeatPolicy.ALWAYS_REPEAT,
    # Special draw reward configuration
    special_draw_reward_at=30,
    special_draw_reward_count=10,
    special_draw_repeat=False,
    # Next banner draw reward configuration
    next_banner_draw_reward_at=60,
    next_banner_draw_reward_count=10,
    next_banner_draw_repeat=False,
)


def create_next_banner(
    template: "BannerTemplate",
    default_operators: list["Operator"],
    previous_banners: list["Banner"],
    banner_name: Optional[str] = None,
    banner_index: Optional[int] = None,
    main_operator: Optional["Operator"] = None,
) -> "Banner":
    """Create a new banner using the template and inheriting main operators from previous banners.

    This is the shared banner creation routine used by both UI and simulation.

    Args:
        template: The banner template to use for the new banner.
        default_operators: List of default operators to include in the banner.
        previous_banners: List of previous banners to inherit main operators from.
        banner_name: Optional name for the banner. If not provided, generates one.
        banner_index: Optional index for naming auto-generated banners.
        main_operator: Optional main operator. If not provided, creates a dummy one.

    Returns:
        A new Banner instance with properly populated operators.
    """
    # Group default operators by rarity
    banner_operators: dict[int, list[Operator]] = {}
    for op in default_operators:
        if op.rarity not in banner_operators:
            banner_operators[op.rarity] = []
        banner_operators[op.rarity].append(op.model_copy(deep=True))

    # Determine banner name
    if banner_name is None:
        if banner_index is not None:
            banner_name = f"卡池_{banner_index}"
        else:
            banner_name = f"卡池_{len(previous_banners) + 1}"

    # Create or use provided main operator
    if main_operator is None:
        idx = banner_index if banner_index is not None else len(previous_banners) + 1
        highest_rarity = max(template.rarities)
        main_operator = Operator(name=f"新干员_{idx}", rarity=highest_rarity, banner=banner_name)  # type: ignore

    # Add main operator to the pool
    main_rarity = main_operator.rarity
    if main_rarity not in banner_operators:
        banner_operators[main_rarity] = []
    # Insert at the beginning
    banner_operators[main_rarity].insert(0, main_operator.model_copy(deep=True))

    # Inherit main operators from previous banners based on template policy
    inherit_count = template.inherit_main_from_previous_banners
    if inherit_count > 0:
        # Get the last N banners' main operators
        prev_mains: list[Operator] = []
        for banner in previous_banners[-inherit_count:]:
            if banner.main_operator:
                prev_mains.append(banner.main_operator)

        # Add them to the pool, avoiding duplicates
        for prev_main in prev_mains:
            rarity = prev_main.rarity
            if rarity not in banner_operators:
                banner_operators[rarity] = []
            existing_names = [op.name for op in banner_operators[rarity]]
            if prev_main.name not in existing_names:
                banner_operators[rarity].insert(0, prev_main.model_copy(deep=True))

    return Banner(  # type: ignore
        name=banner_name,
        operators=banner_operators,
        main_operator=main_operator,
        template=template.model_copy(deep=True),
    )


def prepare_banner_for_next(
    current_banner: "Banner",
    next_banner: "Banner",
) -> None:
    """Prepare the next banner by inheriting reward states from the current banner.

    This is the shared banner transition routine used by both trial draw UI and simulation.
    It handles inheriting reward counters (pity, definitive, potential, etc.) based on
    each reward type's inherit policy defined in the banner template.

    Args:
        current_banner: The banner that was just completed.
        next_banner: The banner to transition to (will be modified in place).
    """
    # Get inherited reward states from current banner based on template policies
    inherited_states = current_banner.get_inherited_reward_states()
    # Reset next banner with inherited states
    next_banner.reset(inherited_states)


class Operator(BaseModel):
    """Represents a gacha operator/character with their stats across simulations."""

    rarity: int = Field(6, description="Rarity of the operator")
    name: str = Field("Operator_Placeholder", description="Name of the operator")
    potential: int = Field(
        0, description="Total potential / number of times drawn across all experiments"
    )
    banner: Optional[str] = Field(
        None, description="Which banner does this operator belong to"
    )
    # Stats for first copy (首抽期望) - uses per-banner draw count for fair comparison
    first_draw_total: int = Field(
        0, description="Sum of per-banner draw counts when first copy obtained"
    )
    first_draw_count: int = Field(
        0, description="Number of experiments where this operator was drawn"
    )
    # Stats for special draws
    drawn_by_special_count: int = Field(
        0, description="Total times drawn by special draw"
    )
    # Draw bucket distribution: count of first draws in each 10-draw bucket
    # Key is bucket start (0, 10, 20, ...), value is count
    # Uses per-banner draw count for fair comparison
    draw_buckets: dict[int, int] = Field(
        default_factory=dict,
        description="Distribution of first draw counts by 10-draw buckets",
    )


class DrawReward(BaseModel):
    """Represents a reward from a draw."""

    operators: list["Operator"] = Field(
        default_factory=list, description="Operators obtained from the draw"
    )
    potential: int = Field(
        default=0, description="Extra potential rewarded for main operator"
    )
    special_draws: int = Field(
        default=0, description="Special draws rewarded (for current banner)"
    )
    next_banner_draws: int = Field(
        default=0, description="Draws rewarded for next banner"
    )


class DrawResult(BaseModel):
    """Result of a single draw from a banner."""

    reward: DrawReward = Field(
        default_factory=lambda: DrawReward(), description="Rewards from this draw"
    )
    got_main: bool = Field(
        default=False, description="Whether the main operator has been obtained"
    )
    triggered_pity: bool = Field(
        default=False, description="Whether this draw triggered the pity guarantee"
    )
    triggered_definitive: bool = Field(
        default=False,
        description="Whether this draw triggered the definitive guarantee",
    )


class DrawRarityDistribution(BaseModel):
    """Probability distribution for drawing rarities. Supports arbitrary rarity levels."""

    probabilities: dict[int, float] = Field(
        ...,
        description="Probability for each rarity level",
    )

    # Pre-sorted list of (rarity, cumulative_probability) for fast lookup
    _sorted_cumulative: list[tuple[int, float]] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Validate probabilities
        for rarity, prob in self.probabilities.items():
            if prob < 0:
                raise ValueError(
                    f"Probability for rarity {rarity} cannot be less than 0"
                )

        total = sum(self.probabilities.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Rarity distribution must sum to 1, got {total}")

        # Pre-compute sorted cumulative probabilities for fast lookup
        cumulative = 0.0
        self._sorted_cumulative = []
        for rarity in sorted(self.probabilities.keys()):
            cumulative += self.probabilities[rarity]
            self._sorted_cumulative.append((rarity, cumulative))

    def get_rarity(self) -> int:
        """Get a random rarity based on the probability distribution."""
        rand = random()
        for rarity, cumulative in self._sorted_cumulative:
            if rand < cumulative:
                return rarity
        # Return highest rarity if we somehow get here (due to floating point)
        return self._sorted_cumulative[-1][0] if self._sorted_cumulative else 0


class Banner(BaseModel):
    operators: dict[int, list[Operator]] = Field(
        {}, description="Operators in the banner by rarity"
    )
    main_operator: Optional[Operator] = Field(
        None, description="Main operator of this banner"
    )
    name: str = Field("Banner_Placeholder", description="Name of the banner")
    expanded: bool = Field(True, description="Whether the banner is expanded in UI")

    # Banner template - defines the draw mechanics
    template: BannerTemplate = Field(
        default_factory=lambda: EndFieldBannerTemplate.model_copy(deep=True),
        description="Template that defines draw mechanics for this banner",
    )

    # Below are runtime fields
    draws_accumulated: int = Field(0, description="Draws spent in this banner")
    draws_accumulated_total: int = Field(
        0, description="Draws spent in this banner, include special draws"
    )
    got_main: bool = Field(False, description="Whether main operator has been drawn")

    # Reward states tracked by RewardType
    reward_states: dict[RewardType, RewardState] = Field(
        default_factory=lambda: {rt: RewardState() for rt in RewardType},
        description="State for each reward type (counter and triggered flag)",
    )

    def _get_reward_config(
        self, reward_type: RewardType
    ) -> tuple[bool, int, InheritPolicy, RepeatPolicy, ResetCondition]:
        """Get reward configuration from template.

        Returns:
            Tuple of (enabled, threshold, inherit_policy, repeat_policy, reset_condition) for the reward type.
        """
        template = self.template
        if reward_type == RewardType.PITY:
            return (
                template.has_pity_draw,
                template.pity_draw_limit,
                template.pity_draw_inherit_policy,
                template.pity_draw_repeat_policy,
                template.pity_reset_condition,
            )
        elif reward_type == RewardType.DEFINITIVE:
            return (
                template.has_definitive_draw,
                template.definitive_draw_count,
                template.definitive_draw_inherit_policy,
                template.definitive_draw_repeat_policy,
                template.definitive_reset_condition,
            )
        elif reward_type == RewardType.POTENTIAL:
            return (
                template.has_potential_reward,
                template.potential_reward_draw,
                template.potential_reward_inherit_policy,
                template.potential_reward_repeat_policy,
                ResetCondition.NONE,
            )
        elif reward_type == RewardType.SPECIAL_DRAW:
            # Special draw uses simple bool for repeat, convert to policy
            repeat_policy = (
                RepeatPolicy.ALWAYS_REPEAT
                if template.special_draw_repeat
                else RepeatPolicy.NO_REPEAT
            )
            return (
                template.special_draw_reward_at > 0,
                template.special_draw_reward_at,
                InheritPolicy.NO_INHERIT,  # special_draw never inherits
                repeat_policy,
                ResetCondition.NONE,
            )
        elif reward_type == RewardType.NEXT_BANNER_DRAW:
            repeat_policy = (
                RepeatPolicy.ALWAYS_REPEAT
                if template.next_banner_draw_repeat
                else RepeatPolicy.NO_REPEAT
            )
            return (
                template.next_banner_draw_reward_at > 0,
                template.next_banner_draw_reward_at,
                InheritPolicy.NO_INHERIT,  # next_banner_draw never inherits
                repeat_policy,
                ResetCondition.NONE,
            )
        return (
            False,
            0,
            InheritPolicy.NO_INHERIT,
            RepeatPolicy.NO_REPEAT,
            ResetCondition.NONE,
        )

    def _get_reward_count(self, reward_type: RewardType) -> int:
        """Get the reward count for a reward type."""
        template = self.template
        if reward_type == RewardType.POTENTIAL:
            return 1  # Potential reward gives 1 potential
        elif reward_type == RewardType.SPECIAL_DRAW:
            return template.special_draw_reward_count
        elif reward_type == RewardType.NEXT_BANNER_DRAW:
            return template.next_banner_draw_reward_count
        return 0  # PITY and DEFINITIVE don't have counts, they affect draw mechanics

    def _is_reward_exhausted(self, reward_type: RewardType) -> bool:
        """Check if a reward is exhausted (triggered and cannot repeat)."""
        state = self.reward_states[reward_type]
        enabled, _, _, repeat_policy, _ = self._get_reward_config(reward_type)
        if not enabled:
            return True
        return state.triggered and repeat_policy == RepeatPolicy.NO_REPEAT

    def _increment_reward_counter(self, reward_type: RewardType) -> None:
        """Increment the counter for a reward type if not exhausted."""
        if not self._is_reward_exhausted(reward_type):
            self.reward_states[reward_type].counter += 1

    def _check_and_trigger_reward(self, reward_type: RewardType) -> bool:
        """Check if a reward should be triggered.

        Args:
            reward_type: The type of reward to check.

        Returns:
            True if the reward was triggered, False otherwise.
        """
        state = self.reward_states[reward_type]
        enabled, threshold, _, repeat_policy, _ = self._get_reward_config(reward_type)

        # If not enabled or no threshold, no reward
        if not enabled or threshold <= 0:
            return False

        # If already triggered and can't repeat, no reward
        if state.triggered and repeat_policy == RepeatPolicy.NO_REPEAT:
            return False

        # Check if counter has reached threshold
        if state.counter >= threshold:
            # For repeating rewards, check modulo; for non-repeating, just check >=
            if repeat_policy == RepeatPolicy.ALWAYS_REPEAT:
                if state.counter % threshold == 0:
                    state.triggered = True
                    return True
            else:
                state.triggered = True
                return True

        return False

    def _reset_reward_counter(self, reward_type: RewardType) -> None:
        """Reset the counter for a reward type to 0."""
        self.reward_states[reward_type].counter = 0

    def _apply_reset_conditions(self, drew_highest_rarity: bool) -> None:
        """Apply reset conditions to all reward counters based on draw outcome.

        Loops through all reward types and checks their reset conditions.
        If the condition is met, marks the reward as triggered and resets its counter.

        Args:
            drew_highest_rarity: Whether the highest rarity was drawn this draw.
        """
        for reward_type in RewardType:
            enabled, _, _, _, reset_condition = self._get_reward_config(reward_type)
            if not enabled:
                continue

            should_reset = False
            if (
                reset_condition == ResetCondition.ON_HIGHEST_RARITY
                and drew_highest_rarity
            ):
                should_reset = True
            elif reset_condition == ResetCondition.ON_MAIN and self.got_main:
                should_reset = True

            if should_reset:
                self.reward_states[reward_type].triggered = True
                self._reset_reward_counter(reward_type)

    def reset(self, inherited_reward_states: Optional[dict[RewardType, int]] = None):
        """Reset banner state for a new simulation run.

        Args:
            inherited_reward_states: Inherited reward counters by type from previous banner
        """
        self.draws_accumulated = 0
        self.draws_accumulated_total = 0
        self.got_main = False

        # Reset reward states, inheriting counters where applicable
        inherited = inherited_reward_states or {}
        for reward_type in RewardType:
            self.reward_states[reward_type] = RewardState(
                counter=inherited.get(reward_type, 0),
                triggered=False,
            )

    def get_inherited_reward_states(self) -> dict[RewardType, int]:
        """Get reward counters to inherit to next banner based on template policies."""
        inherited = {}
        for reward_type in RewardType:
            enabled, _, inherit_policy, _, _ = self._get_reward_config(reward_type)
            if not enabled:
                continue

            state = self.reward_states[reward_type]

            if inherit_policy == InheritPolicy.NO_INHERIT:
                continue
            elif inherit_policy == InheritPolicy.INHERIT_TO_NEXT:
                # Only inherit if not triggered (for INHERIT_TO_NEXT policy)
                if not state.triggered:
                    inherited[reward_type] = state.counter
            elif inherit_policy == InheritPolicy.ALWAYS_INHERIT:
                inherited[reward_type] = state.counter

        return inherited

    def _get_default_distribution(self) -> DrawRarityDistribution:
        """Get the default probability distribution from the banner's template.

        Returns:
            DrawRarityDistribution with base probabilities for each rarity level.
        """
        dist = self.template.default_distribution
        # Build distribution dict from template
        probs = {rp.rarity: rp.probability for rp in dist}
        return DrawRarityDistribution(probabilities=probs)

    def _get_draw_rarity_distribution(
        self, is_special_draw: bool = False
    ) -> DrawRarityDistribution:
        """Get the rarity distribution for the current draw based on pity state.

        Takes into account:
        - Whether this is a special draw (always uses default distribution)
        - Current pity counter and whether pity has been exhausted
        - Pity boost when in the soft pity range (pity_draw_start to pity_draw_limit)
        - Guaranteed highest rarity at pity_draw_limit

        Args:
            is_special_draw: If True, returns default distribution without pity effects.

        Returns:
            DrawRarityDistribution with adjusted probabilities based on pity state.
        """
        template = self.template

        # Special draws always use default distribution (no pity boost)
        if is_special_draw:
            return self._get_default_distribution()

        # Check if pity is exhausted
        if self._is_reward_exhausted(RewardType.PITY):
            return self._get_default_distribution()

        if not template.has_pity_draw:
            return self._get_default_distribution()

        pity_counter = self.reward_states[RewardType.PITY].counter

        # Check if we've reached the pity limit
        if pity_counter >= template.pity_draw_limit:
            # This is the pity draw, 100% highest rarity
            self.reward_states[RewardType.PITY].triggered = True
            highest_rarity = max(template.rarities)
            return DrawRarityDistribution(probabilities={highest_rarity: 1.0})
        elif pity_counter >= template.pity_draw_start:
            # In pity draw range, add probability to highest rarity
            highest_rarity = max(template.rarities)
            base_probs = {
                rp.rarity: rp.probability for rp in template.default_distribution
            }
            base_highest = base_probs.get(highest_rarity, 0.008)

            # Calculate boosted probability for highest rarity
            boost = (
                pity_counter - template.pity_draw_start + 1
            ) * template.pity_rarity_boost_per_draw
            probability_highest = min(base_highest + boost, 1.0)

            # Keep the ratio between other rarities the same
            remaining_prob = 1 - probability_highest
            original_remaining = 1 - base_highest

            new_probs = {}
            if original_remaining > 0:
                for rarity, prob in base_probs.items():
                    if rarity != highest_rarity:
                        new_probs[rarity] = prob * remaining_prob / original_remaining
            new_probs[highest_rarity] = probability_highest

            return DrawRarityDistribution(probabilities=new_probs)
        else:
            return self._get_default_distribution()

    def draw(self, is_special_draw: bool = False) -> DrawResult:
        """Perform a single draw from the banner.

        Args:
            is_special_draw: Whether this is a special (bonus) draw that doesn't count toward counters.

        Returns:
            DrawResult containing the operator drawn and reward information.
        """
        template = self.template

        # Special draw does not count as draw counts of this banner.
        self.draws_accumulated_total += 1
        if not is_special_draw:
            self.draws_accumulated += 1
            # Increment all reward counters that are not exhausted
            for reward_type in RewardType:
                self._increment_reward_counter(reward_type)

        # Check rewards that give items (potential, special draw, next banner draw)
        reward_potential = False
        reward_special_draw = 0
        reward_next_banner_draw = 0

        if not is_special_draw:
            if self._check_and_trigger_reward(RewardType.POTENTIAL):
                reward_potential = True
            if self._check_and_trigger_reward(RewardType.SPECIAL_DRAW):
                reward_special_draw = self._get_reward_count(RewardType.SPECIAL_DRAW)
            if self._check_and_trigger_reward(RewardType.NEXT_BANNER_DRAW):
                reward_next_banner_draw = self._get_reward_count(
                    RewardType.NEXT_BANNER_DRAW
                )

        # Check if definitive draw is triggered - guarantees the main operator
        is_definitive_draw = (
            not is_special_draw
            and self.main_operator
            and not self._is_reward_exhausted(RewardType.DEFINITIVE)
            and self._check_and_trigger_reward(RewardType.DEFINITIVE)
        )

        triggered_pity = False
        drawn_operator: Operator
        drew_highest_rarity = False
        highest_rarity = max(template.rarities)

        if is_definitive_draw:
            # Definitive draw guarantees the main operator (which is highest rarity)
            self.got_main = True
            drawn_operator = self.main_operator  # type: ignore
            drew_highest_rarity = True
        else:
            # Get the current rarity distribution and get the rarity of this draw
            pity_triggered_before = self.reward_states[RewardType.PITY].triggered
            distribution = self._get_draw_rarity_distribution(
                is_special_draw=is_special_draw
            )
            triggered_pity = (
                self.reward_states[RewardType.PITY].triggered
                and not pity_triggered_before
            )
            rarity = distribution.get_rarity()

            if rarity not in self.operators.keys():
                raise RuntimeError(f"Rarity {rarity} does not exist in the banner!")

            drew_highest_rarity = rarity == highest_rarity

            # Build operator probability distribution
            operators_to_draw = self.operators[rarity]
            if not operators_to_draw:
                raise RuntimeError(
                    f"No operator with rarity {rarity} exists in the banner!"
                )

            main_prob = template.main_probability
            if self.main_operator in operators_to_draw:
                operators_to_draw_distribution = [
                    (
                        op,
                        (
                            main_prob
                            if op.name == self.main_operator.name
                            else (
                                0
                                if len(operators_to_draw) == 1
                                else (1 - main_prob) / (len(operators_to_draw) - 1)
                            )
                        ),
                    )
                    for op in operators_to_draw
                ]
            else:
                operators_to_draw_distribution = [
                    (op, 1 / len(operators_to_draw)) for op in operators_to_draw
                ]
            operators_to_draw_distribution = sorted(
                operators_to_draw_distribution, key=lambda x: x[1], reverse=True
            )

            # Perform the actual draw
            rand = random()
            probability = 0
            drawn_operator = operators_to_draw_distribution[-1][0]
            for op, op_prob in operators_to_draw_distribution:
                probability += op_prob
                if rand < probability:
                    drawn_operator = op
                    break

            # Check if we drew the main operator
            if self.main_operator and drawn_operator.name == self.main_operator.name:
                self.got_main = True

        # Apply reset conditions for all rewards based on draw outcome
        if not is_special_draw:
            self._apply_reset_conditions(drew_highest_rarity)

        return DrawResult(
            reward=DrawReward(
                operators=[drawn_operator],
                potential=1 if reward_potential else 0,
                special_draws=reward_special_draw,
                next_banner_draws=reward_next_banner_draw,
            ),
            got_main=self.got_main,
            triggered_pity=triggered_pity,
            triggered_definitive=bool(is_definitive_draw),
        )
