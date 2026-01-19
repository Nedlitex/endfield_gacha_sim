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
    pity_draw_reset_on_highest_rarity: bool = Field(
        default=True,
        description="Whether pity counter resets when drawing the highest rarity",
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
    definitive_draw_reset_on_highest_rarity: bool = Field(
        default=False,
        description="Whether definitive draw counter resets when drawing the highest rarity",
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

    def validate_distribution(self) -> bool:
        """Validate that the default distribution sums to 1.0 and covers all rarities."""
        total = sum(rp.probability for rp in self.default_distribution)
        if abs(total - 1.0) > 1e-9:
            return False
        distribution_rarities = {rp.rarity for rp in self.default_distribution}
        return distribution_rarities == set(self.rarities)


class Operator(BaseModel):
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


class DrawRarityDistribution(BaseModel):
    probability_rarity4: float = Field(0, description="Probability of rarity 4")
    probability_rarity5: float = Field(0, description="Probability of rarity 5")
    probability_rarity6: float = Field(0, description="Probability of rarity 6")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if (
            self.probability_rarity4 < 0
            or self.probability_rarity5 < 0
            or self.probability_rarity6 < 0
        ):
            raise ValueError("Rarity cannot be less than 0")
        if (
            self.probability_rarity4
            + self.probability_rarity5
            + self.probability_rarity6
            != 1
        ):
            raise ValueError("Rarity distribution must sum to 1")

    def get_rarity(self) -> int:
        rand = random()
        if rand < self.probability_rarity4:
            return 4
        elif rand < self.probability_rarity4 + self.probability_rarity5:
            return 5
        else:
            return 6


class Banner(BaseModel):
    operators: dict[int, list[Operator]] = Field(
        {}, description="Operators in the banner by rarity"
    )
    main_operator: Optional[Operator] = Field(
        None, description="Main operator of this banner"
    )
    name: str = Field("Banner_Placeholder", description="Name of the banner")
    expanded: bool = Field(True, description="Whether the banner is expanded in UI")
    special_draw_reward_at: int = Field(
        30, description="Draws accumulated to get special draw reward"
    )
    special_draw_reward_count: int = Field(10, description="Special draw reward")
    next_banner_draw_reward_at: int = Field(
        60, description="Draws accumulated to get next banner's draw reward"
    )
    next_banner_draw_reward_count: int = Field(
        10, description="Next banner draw reward"
    )
    pity_draw_begin: int = Field(
        66, description="Draws accumulated when pity draw begin"
    )
    pity_draw_max: int = Field(80, description="Pity draw max count")
    definitive_draw_number: int = Field(
        120, description="The draw number that must return the main operator"
    )
    main_operator_potential_reward_draw: int = Field(
        240,
        description="The draw number that rewards extra potential for main operator of this banner",
    )

    # Below are runtime fields
    draws_accumulated: int = Field(0, description="Draws spent in this banner")
    draws_accumulated_total: int = Field(
        0, description="Draws spent in this banner, include special draws"
    )
    pity_draw_counter: int = Field(0, description="Pity draw counter")
    got_pity: bool = Field(False, description="Whether pity draw has been drawn")
    got_main: bool = Field(False, description="Whether main operator has been drawn")

    def reset(self, pity_draw_counter: int):
        self.draws_accumulated = 0
        self.draws_accumulated_total = 0
        self.pity_draw_counter = pity_draw_counter
        self.got_main = False
        self.got_pity = False

    def get_pity_draw_for_next_banner(self) -> int:
        if self.got_main or self.got_pity:
            return 0
        return self.pity_draw_counter

    def _get_draw_rarity_distribution(
        self, is_special_draw: bool = False
    ) -> DrawRarityDistribution:
        if is_special_draw or self.got_pity or self.got_main:
            # Special draw, already got pity draw or already got main operator: just use the default distribution
            return DrawRarityDistribution(
                probability_rarity4=0.912,
                probability_rarity5=0.08,
                probability_rarity6=0.008,
            )
        elif self.pity_draw_counter == self.pity_draw_max:
            # This is the pity draw, 100% rarity 6
            self.got_pity = True
            return DrawRarityDistribution(
                probability_rarity4=0, probability_rarity5=0, probability_rarity6=1
            )
        elif (
            self.pity_draw_counter >= self.pity_draw_begin
            and self.pity_draw_counter < self.pity_draw_max
        ):
            # In pity draw range, add probability to rarity 6 according.
            probability_rarity6 = (
                0.008 + (self.pity_draw_counter - self.pity_draw_begin + 1) * 0.05
            )
            # Keep the ration between r4 and r5 the same while increasing the r6 probability.
            probability_rarity5 = (1 - probability_rarity6) * 0.08 / 0.992
            probability_rarity4 = 1 - probability_rarity5 - probability_rarity6
            return DrawRarityDistribution(
                probability_rarity4=probability_rarity4,
                probability_rarity5=probability_rarity5,
                probability_rarity6=probability_rarity6,
            )
        else:
            return DrawRarityDistribution(
                probability_rarity4=0.912,
                probability_rarity5=0.08,
                probability_rarity6=0.008,
            )

    def draw(
        self, is_special_draw: bool = False
    ) -> tuple[Operator, bool, int, int, bool, bool]:
        """Returns the tuple of [operator drawn, extra potential rewarded, special draws rewarded, next banner draws rewarded, whether got main, whether this draw triggered pity]"""
        # Special draw does not count as draw counts of this banner.
        self.draws_accumulated_total += 1
        if not is_special_draw:
            self.draws_accumulated += 1
            # Pity draw counter +1 if we haven't got main operator
            if not self.got_main:
                self.pity_draw_counter += 1

        # A reward of the potential of the main operator is given if the current draw is a multiply of the reward draw count.
        reward_potential = False
        if (
            not is_special_draw
            and self.main_operator_potential_reward_draw > 0
            and self.draws_accumulated >= self.main_operator_potential_reward_draw
            and self.draws_accumulated % self.main_operator_potential_reward_draw == 0
        ):
            reward_potential = True

        # A reward of special draws of this banner is given if accumulated draw reaches the target.
        reward_special_draw = 0
        if (
            not is_special_draw
            and self.special_draw_reward_at > 0
            and self.draws_accumulated == self.special_draw_reward_at
        ):
            reward_special_draw = self.special_draw_reward_count

        # A reward of draws of next banner is given if accumulated draw reaches the target.
        reward_next_banner_draw = 0
        if (
            not is_special_draw
            and self.next_banner_draw_reward_at > 0
            and self.draws_accumulated == self.next_banner_draw_reward_at
        ):
            reward_next_banner_draw = self.next_banner_draw_reward_count

        # Short-cut: if the draw is definitive draw, just return the main operator if any. Special draw
        # cannot trigger this. Also if already got main, this is void.
        if (
            not self.got_main
            and not is_special_draw
            and self.draws_accumulated == self.definitive_draw_number
            and self.main_operator
        ):
            self.got_main = True
            return (
                self.main_operator,
                reward_potential,
                reward_special_draw,
                reward_next_banner_draw,
                self.got_main,
                False,  # definitive draw is not pity
            )

        # Get the current rarity distribution and get the rarity of this draw
        # Track if this draw triggers pity
        pity_before = self.got_pity
        distribution = self._get_draw_rarity_distribution(
            is_special_draw=is_special_draw
        )
        triggered_pity = self.got_pity and not pity_before
        rarity = distribution.get_rarity()
        if rarity not in self.operators.keys():
            raise RuntimeError(f"Rarity {rarity} does not exist in the banner!")

        # For the operators to draw from, create their distribution, the main operator gets 50% while others share the
        # remaining, sort the distribution by probability from high to low.
        operators_to_draw = self.operators[rarity]
        if not operators_to_draw:
            raise RuntimeError(
                f"No operation with rarity {rarity} exists in the banner!"
            )
        if self.main_operator in operators_to_draw:
            operators_to_draw_distribution = [
                (
                    op,
                    (
                        0.5
                        if op.name == self.main_operator.name
                        else (
                            0
                            if len(operators_to_draw) == 1
                            else (0.5 / (len(operators_to_draw) - 1))
                        )
                    ),
                )
                for op in operators_to_draw
            ]
        else:
            operators_to_draw_distribution = [
                (op, (1 / len(operators_to_draw))) for op in operators_to_draw
            ]
        operators_to_draw_distribution = sorted(
            operators_to_draw_distribution, key=lambda x: x[1], reverse=True
        )

        # Now do the actually draw.
        rand = random()
        probability = 0
        for op, op_prob in operators_to_draw_distribution:
            probability += op_prob
            if rand < probability:
                if self.main_operator and op.name == self.main_operator.name:
                    self.got_main = True
                return (
                    op,
                    reward_potential,
                    reward_special_draw,
                    reward_next_banner_draw,
                    self.got_main,
                    triggered_pity,
                )

        raise RuntimeError(
            f"Operator distribution has problem: {operators_to_draw_distribution}"
        )
