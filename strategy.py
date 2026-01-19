"""Strategy module for gacha simulation.

This module contains all strategy-related classes including:
- DrawBehavior: How draws are performed (single/multi, pay)
- Condition types: When rules apply
- Action types: What to do when conditions match
- StrategyRule: Combines conditions + action
- DrawStrategy: Main strategy class
- EvaluationContext: Context for strategy evaluation
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

# =============================================================================
# Evaluation Context
# =============================================================================


class EvaluationContext(BaseModel):
    """Context passed to condition and strategy evaluation.

    Contains the current state of the simulation that conditions can check against.
    """

    draws_accumulated: int = Field(
        default=0, description="Non-special draws accumulated on current banner"
    )
    normal_draws: int = Field(
        default=0, description="Available normal draws (carryover resources)"
    )
    got_main: bool = Field(
        default=False, description="Whether main operator has been obtained"
    )
    got_highest_rarity: bool = Field(
        default=False,
        description="Whether any highest rarity operator has been obtained this banner",
    )
    got_highest_rarity_but_not_main: bool = Field(
        default=False,
        description="Whether highest rarity obtained but not the main operator",
    )
    got_pity_without_main: bool = Field(
        default=False, description="Whether pity triggered without getting main"
    )
    current_potential: int = Field(
        default=0, description="Current number of copies of main operator obtained"
    )
    banner_index: int = Field(default=0, description="Current banner index (0-based)")
    pity_counter: int = Field(
        default=0, description="Current pity counter (draws since last highest rarity)"
    )
    definitive_draw_counter: int = Field(
        default=0,
        description="Current definitive draw counter (draws toward guaranteed main)",
    )

    model_config = {"frozen": True}


# =============================================================================
# Behavior
# =============================================================================


class DrawBehavior(BaseModel):
    """Determines HOW draws are performed (single vs multi-draw, paying).

    These settings apply globally regardless of which rule matches.
    """

    always_single_draw: bool = Field(
        default=False, description="Always perform single draws instead of 10-draws"
    )
    single_draw_after: int = Field(
        default=0,
        ge=0,
        description="Switch to single draws after X accumulated draws (0 = never)",
    )
    pay: bool = Field(
        default=False,
        description="Pay for extra draws when resources are insufficient",
    )


# =============================================================================
# Condition Types
# =============================================================================


class DrawCountCondition(BaseModel):
    """Condition based on current draw count (non-special draws)."""

    type: Literal["draw_count"] = "draw_count"
    min_draws: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum draw count (inclusive). None means no minimum.",
    )
    max_draws: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum draw count (inclusive). None means no maximum.",
    )

    def evaluate(self, context: EvaluationContext) -> bool:
        """Evaluate if the current draw count satisfies this condition."""
        if self.min_draws is not None and context.draws_accumulated < self.min_draws:
            return False
        if self.max_draws is not None and context.draws_accumulated > self.max_draws:
            return False
        return True


class GotMainCondition(BaseModel):
    """Condition based on whether the main operator has been obtained."""

    type: Literal["got_main"] = "got_main"
    value: bool = Field(
        default=True, description="True = condition matches when main is obtained"
    )

    def evaluate(self, context: EvaluationContext) -> bool:
        return context.got_main == self.value


class GotHighestRarityCondition(BaseModel):
    """Condition based on whether any highest rarity operator has been obtained."""

    type: Literal["got_highest_rarity"] = "got_highest_rarity"
    value: bool = Field(
        default=True,
        description="True = condition matches when highest rarity is obtained",
    )

    def evaluate(self, context: EvaluationContext) -> bool:
        return context.got_highest_rarity == self.value


class GotHighestRarityButNotMainCondition(BaseModel):
    """Condition based on whether highest rarity was obtained but not main operator."""

    type: Literal["got_highest_rarity_but_not_main"] = "got_highest_rarity_but_not_main"
    value: bool = Field(
        default=True,
        description="True = condition matches when highest rarity obtained but not main",
    )

    def evaluate(self, context: EvaluationContext) -> bool:
        return context.got_highest_rarity_but_not_main == self.value


class ResourceThresholdCondition(BaseModel):
    """Condition based on remaining normal draws.

    Replaces the old skip_banner_threshold functionality.
    """

    type: Literal["resource_threshold"] = "resource_threshold"
    min_normal_draws: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum normal draws required. None means no minimum.",
    )
    max_normal_draws: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum normal draws. None means no maximum.",
    )
    check_once: bool = Field(
        default=False,
        description="If True, only check at banner entry (draws_accumulated == 0). "
        "If False, check continuously.",
    )

    def evaluate(self, context: EvaluationContext) -> bool:
        # If check_once is True, only evaluate when at banner entry
        if self.check_once and context.draws_accumulated > 0:
            return False  # Condition doesn't match after entry
        if (
            self.min_normal_draws is not None
            and context.normal_draws < self.min_normal_draws
        ):
            return False
        if (
            self.max_normal_draws is not None
            and context.normal_draws > self.max_normal_draws
        ):
            return False
        return True


class GotPityWithoutMainCondition(BaseModel):
    """Condition based on whether pity triggered without getting main."""

    type: Literal["got_pity_without_main"] = "got_pity_without_main"
    value: bool = Field(
        default=True,
        description="True = condition matches when pity triggered but main not obtained",
    )

    def evaluate(self, context: EvaluationContext) -> bool:
        return context.got_pity_without_main == self.value


class BannerIndexCondition(BaseModel):
    """Condition based on banner index (every Nth banner starting at a specific banner).

    The banner_index in context is 0-based, but we use 1-based counting for start_at:
    - Banner 1 has index 0, Banner 2 has index 1, etc.

    Examples:
    - every_n=0: matches all banners (no filtering)
    - every_n=1: matches banner 1, 2, 3, ... (same as 0, every banner)
    - every_n=2, start_at=1: matches banner 1, 3, 5, ... (odd banners)
    - every_n=2, start_at=2: matches banner 2, 4, 6, ... (even banners)
    - every_n=3, start_at=1: matches banner 1, 4, 7, ...
    - every_n=3, start_at=2: matches banner 2, 5, 8, ...
    - every_n=3, start_at=3: matches banner 3, 6, 9, ...
    """

    type: Literal["banner_index"] = "banner_index"
    every_n: int = Field(
        default=0,
        ge=0,
        description="Apply on every Nth banner (0 = every banner)",
    )
    start_at: int = Field(
        default=1,
        ge=1,
        description="First banner in the cycle (1-based). E.g., start_at=1 means banner 1, 1+every_n, 1+2*every_n, ...",
    )

    def evaluate(self, context: EvaluationContext) -> bool:
        # 0 or 1 means every banner
        if self.every_n <= 1:
            return True
        # Convert start_at (1-based) to offset (0-based)
        # start_at=1 -> offset=0, start_at=2 -> offset=1, etc.
        offset = (self.start_at - 1) % self.every_n
        # Check if banner index matches the pattern
        # index % every_n == offset means:
        #   start_at=1: banners 1, every_n+1, 2*every_n+1, ... (indices 0, every_n, 2*every_n)
        #   start_at=2: banners 2, every_n+2, 2*every_n+2, ... (indices 1, every_n+1, 2*every_n+1)
        return context.banner_index % self.every_n == offset


class PityCounterCondition(BaseModel):
    """Condition based on current pity counter (draws since last highest rarity).

    The pity counter tracks how many draws have been made since the last
    highest rarity operator was obtained. This is useful for strategies
    that want to stop at a certain pity level or continue until pity.

    Examples:
    - min_pity=60: matches when pity counter >= 60 (in soft pity range)
    - max_pity=30: matches when pity counter <= 30 (early draws)
    - min_pity=0, max_pity=0: matches when pity counter is exactly 0 (just got highest rarity)
    """

    type: Literal["pity_counter"] = "pity_counter"
    min_pity: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum pity counter (inclusive). None means no minimum.",
    )
    max_pity: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum pity counter (inclusive). None means no maximum.",
    )

    def evaluate(self, context: EvaluationContext) -> bool:
        if self.min_pity is not None and context.pity_counter < self.min_pity:
            return False
        if self.max_pity is not None and context.pity_counter > self.max_pity:
            return False
        return True


class DefinitiveDrawCounterCondition(BaseModel):
    """Condition based on current definitive draw counter (draws toward guaranteed main).

    The definitive draw counter tracks progress toward the guaranteed main operator
    (big pity/大保底). This is useful for strategies that want to check how close
    the player is to hitting the definitive draw guarantee.

    Examples:
    - min_definitive=100: matches when definitive counter >= 100 (close to guarantee)
    - max_definitive=60: matches when definitive counter <= 60 (early in cycle)
    """

    type: Literal["definitive_draw_counter"] = "definitive_draw_counter"
    min_definitive: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum definitive draw counter (inclusive). None means no minimum.",
    )
    max_definitive: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum definitive draw counter (inclusive). None means no maximum.",
    )

    def evaluate(self, context: EvaluationContext) -> bool:
        if (
            self.min_definitive is not None
            and context.definitive_draw_counter < self.min_definitive
        ):
            return False
        if (
            self.max_definitive is not None
            and context.definitive_draw_counter > self.max_definitive
        ):
            return False
        return True


# Union of all condition types using discriminated union
StrategyCondition = Annotated[
    Union[
        DrawCountCondition,
        GotMainCondition,
        GotHighestRarityCondition,
        GotHighestRarityButNotMainCondition,
        ResourceThresholdCondition,
        GotPityWithoutMainCondition,
        BannerIndexCondition,
        PityCounterCondition,
        DefinitiveDrawCounterCondition,
    ],
    Field(discriminator="type"),
]


# =============================================================================
# Action Types
# =============================================================================


class StopAction(BaseModel):
    """Action to stop drawing."""

    type: Literal["stop"] = "stop"
    pay_override: Optional[bool] = Field(
        default=None,
        description="Override pay setting. None = use strategy's behavior.pay",
    )


class ContinueAction(BaseModel):
    """Action to continue drawing with constraints."""

    type: Literal["continue"] = "continue"
    min_draws_per_banner: int = Field(
        default=0, ge=0, description="Minimum draws to perform on this banner"
    )
    max_draws_per_banner: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum draws per banner (None or 0 = unlimited)",
    )
    stop_on_main: bool = Field(
        default=False,
        description="Stop immediately after getting the main operator",
    )
    stop_on_highest_rarity: bool = Field(
        default=False,
        description="Stop immediately after getting any highest rarity operator",
    )
    target_potential: Optional[int] = Field(
        default=None,
        ge=1,
        description="Target number of main operator copies to obtain. None = don't target potential.",
    )
    target_pity: Optional[int] = Field(
        default=None,
        ge=1,
        description="Target pity counter to reach before stopping. None = don't target pity.",
    )
    target_definitive_draw: Optional[int] = Field(
        default=None,
        ge=1,
        description="Target definitive draw counter to reach before stopping. None = don't target.",
    )
    pay_override: Optional[bool] = Field(
        default=None,
        description="Override pay setting. None = use strategy's behavior.pay",
    )


class DelegateAction(BaseModel):
    """Action that delegates to another strategy by name."""

    type: Literal["delegate"] = "delegate"
    strategy_name: str = Field(..., description="Name of the strategy to delegate to")


# Union of all action types using discriminated union
StrategyAction = Annotated[
    Union[StopAction, ContinueAction, DelegateAction],
    Field(discriminator="type"),
]


# =============================================================================
# Strategy Rule
# =============================================================================


class StrategyRule(BaseModel):
    """A single rule within a strategy: conditions -> action.

    All conditions are combined with AND logic - all must match for the rule to apply.
    """

    conditions: list[StrategyCondition] = Field(
        default_factory=list,
        description="List of conditions to check (AND logic). Empty = always matches.",
    )
    action: StrategyAction = Field(
        ..., description="Action to take when all conditions match"
    )
    priority: int = Field(
        default=0, description="Higher priority rules are evaluated first"
    )

    def matches(self, context: EvaluationContext) -> bool:
        """Check if all conditions match the given context."""
        if not self.conditions:
            return True  # Empty conditions = always matches
        return all(cond.evaluate(context) for cond in self.conditions)


# =============================================================================
# Main DrawStrategy Class
# =============================================================================


class DrawStrategy(BaseModel):
    """Configuration for automated drawing strategy on banners.

    The strategy is evaluated as follows:
    1. Rules are sorted by priority (descending)
    2. For each rule, if all conditions match, the action is executed
    3. If action is DelegateAction, recursively evaluate the target strategy
    4. First matching rule wins (no further rules evaluated)
    5. If no rules match, default_action is used

    The behavior settings (always_single_draw, single_draw_after, pay) are
    applied globally regardless of which rule matches.
    """

    name: str = Field(
        default="Strategy_Placeholder", description="Name of the strategy"
    )
    behavior: DrawBehavior = Field(
        default_factory=DrawBehavior, description="Global drawing behavior settings"
    )
    rules: list[StrategyRule] = Field(
        default_factory=list, description="List of condition-action rules"
    )
    default_action: StrategyAction = Field(
        default_factory=lambda: ContinueAction(),
        description="Default action when no rules match",
    )

    @model_validator(mode="after")
    def validate_no_self_reference(self) -> "DrawStrategy":
        """Validate that strategy doesn't delegate to itself."""
        for rule in self.rules:
            if isinstance(rule.action, DelegateAction):
                if rule.action.strategy_name == self.name:
                    raise ValueError(
                        f"Strategy '{self.name}' cannot delegate to itself"
                    )
        if isinstance(self.default_action, DelegateAction):
            if self.default_action.strategy_name == self.name:
                raise ValueError(f"Strategy '{self.name}' cannot delegate to itself")
        return self

    def evaluate(
        self,
        context: EvaluationContext,
        strategy_registry: Optional[dict[str, "DrawStrategy"]] = None,
        visited: Optional[set[str]] = None,
    ) -> tuple[StrategyAction, DrawBehavior]:
        """Evaluate the strategy and return the action to take.

        Args:
            context: Current evaluation context
            strategy_registry: Dict of strategy name -> strategy for delegation
            visited: Set of visited strategy names (for cycle detection)

        Returns:
            Tuple of (resolved action, behavior to apply)

        Raises:
            ValueError: If cyclic delegation is detected
        """
        visited = visited or set()

        if self.name in visited:
            raise ValueError(
                f"Cyclic strategy reference detected: {' -> '.join(visited)} -> {self.name}"
            )
        visited = visited | {self.name}  # Create new set to avoid mutation

        # Sort rules by priority (descending)
        sorted_rules = sorted(self.rules, key=lambda r: -r.priority)

        # Find first matching rule
        matched_action: Optional[StrategyAction] = None
        for rule in sorted_rules:
            if rule.matches(context):
                matched_action = rule.action
                break

        # Use default action if no rule matched
        if matched_action is None:
            matched_action = self.default_action

        # Handle delegation
        if isinstance(matched_action, DelegateAction):
            if strategy_registry is None:
                raise ValueError(
                    f"Cannot delegate to '{matched_action.strategy_name}': no strategy registry provided"
                )
            if matched_action.strategy_name not in strategy_registry:
                raise ValueError(
                    f"Cannot delegate to '{matched_action.strategy_name}': strategy not found"
                )
            target = strategy_registry[matched_action.strategy_name]
            # Recursively evaluate, combining behaviors
            action, target_behavior = target.evaluate(
                context, strategy_registry, visited
            )
            # Merge behaviors: current strategy's behavior takes precedence for non-default values
            merged_behavior = DrawBehavior(
                always_single_draw=self.behavior.always_single_draw
                or target_behavior.always_single_draw,
                single_draw_after=self.behavior.single_draw_after
                or target_behavior.single_draw_after,
                pay=self.behavior.pay or target_behavior.pay,
            )
            return action, merged_behavior

        return matched_action, self.behavior

    def should_continue_drawing(
        self,
        context: EvaluationContext,
        strategy_registry: Optional[dict[str, "DrawStrategy"]] = None,
    ) -> bool:
        """Determine if drawing should continue.

        Returns:
            True if should continue drawing, False to stop
        """
        try:
            action, behavior = self.evaluate(context, strategy_registry)
        except ValueError:
            return False  # Stop on any error

        if isinstance(action, StopAction):
            return False
        elif isinstance(action, ContinueAction):
            # Check max draws constraint
            if action.max_draws_per_banner and action.max_draws_per_banner > 0:
                if context.draws_accumulated >= action.max_draws_per_banner:
                    return False

            # Check stop_on_main constraint
            if action.stop_on_main and context.got_main:
                return False

            # Check stop_on_highest_rarity constraint
            if action.stop_on_highest_rarity and context.got_highest_rarity:
                return False

            # Check target_potential constraint
            if action.target_potential is not None:
                if context.current_potential >= action.target_potential:
                    return False

            # Check target_pity constraint (stop when reached target pity)
            if action.target_pity is not None:
                if context.pity_counter >= action.target_pity:
                    return False

            # Check target_definitive_draw constraint
            if action.target_definitive_draw is not None:
                if context.definitive_draw_counter >= action.target_definitive_draw:
                    return False

            # Determine effective pay setting (action override takes precedence)
            effective_pay = (
                action.pay_override if action.pay_override is not None else behavior.pay
            )

            # Check resource availability
            if context.normal_draws <= 0 and not effective_pay:
                # No draws available and can't pay
                # But still continue if we haven't met min_draws and can pay
                return False

            return True

        return False

    def get_effective_pay(
        self,
        context: EvaluationContext,
        strategy_registry: Optional[dict[str, "DrawStrategy"]] = None,
    ) -> bool:
        """Get the effective pay setting considering action override.

        Args:
            context: Current evaluation context
            strategy_registry: Dict for delegation lookup

        Returns:
            True if should pay for draws, False otherwise
        """
        try:
            action, behavior = self.evaluate(context, strategy_registry)
        except ValueError:
            return False

        # Check for pay_override in the action
        if isinstance(action, (StopAction, ContinueAction)):
            if action.pay_override is not None:
                return action.pay_override
        return behavior.pay

    def get_draw_amount(
        self,
        context: EvaluationContext,
        total_available: int,
        strategy_registry: Optional[dict[str, "DrawStrategy"]] = None,
    ) -> int:
        """Determine how many draws to perform (1 or 10).

        Args:
            context: Current evaluation context
            total_available: Total draws available
            strategy_registry: Dict for delegation lookup

        Returns:
            Number of draws to perform (1 or 10)
        """
        try:
            action, behavior = self.evaluate(context, strategy_registry)
        except ValueError:
            return 1

        # Determine effective pay setting
        effective_pay = behavior.pay
        if isinstance(action, (StopAction, ContinueAction)):
            if action.pay_override is not None:
                effective_pay = action.pay_override

        if behavior.always_single_draw:
            return 1
        if (
            behavior.single_draw_after > 0
            and context.draws_accumulated >= behavior.single_draw_after
        ):
            return 1
        if total_available >= 10 or effective_pay:
            return 10
        return 1

    def get_description(
        self,
        strategy_registry: Optional[dict[str, "DrawStrategy"]] = None,
        visited: Optional[set[str]] = None,
        indent: int = 0,
    ) -> str:
        """Generate a human-readable Chinese description of the strategy.

        Args:
            strategy_registry: Dict of strategy name -> strategy for delegation lookup
            visited: Set of visited strategy names (for cycle detection)
            indent: Indentation level for nested strategies

        Returns:
            Human-readable Chinese description string
        """
        visited = visited or set()
        prefix = "  " * indent

        # Cycle detection
        if self.name in visited:
            return f"{prefix}⚠️ 循环引用: {self.name}"
        visited = visited | {self.name}

        lines = [f"{prefix}【{self.name}】"]

        # Behavior section
        behavior_parts = []
        if self.behavior.always_single_draw:
            behavior_parts.append("始终单抽")
        if self.behavior.single_draw_after > 0:
            behavior_parts.append(f"累计{self.behavior.single_draw_after}抽后单抽")
        if self.behavior.pay:
            behavior_parts.append("氪金(抽数不足时补充)")

        if behavior_parts:
            lines.append(f"{prefix}抽卡行为: {', '.join(behavior_parts)}")

        # Rules section using 如果/否则如果/否则 format
        if self.rules:
            sorted_rules = sorted(self.rules, key=lambda r: -r.priority)
            for i, rule in enumerate(sorted_rules):
                cond_str, action_str = self._rule_to_chinese_parts(
                    rule, strategy_registry, visited, indent + 1
                )
                if i == 0:
                    lines.append(f"{prefix}如果 {cond_str}，那么 {action_str}")
                else:
                    lines.append(f"{prefix}否则如果 {cond_str}，那么 {action_str}")

        # Default action
        default_desc = self._action_to_chinese(
            self.default_action, strategy_registry, visited, indent + 1
        )
        if self.rules:
            lines.append(f"{prefix}否则，{default_desc}")
        else:
            lines.append(f"{prefix}默认行为: {default_desc}")

        return "\n".join(lines)

    def _condition_to_chinese(self, cond: "StrategyCondition") -> str:
        """Convert a condition to Chinese text."""
        if isinstance(cond, DrawCountCondition):
            parts = []
            if cond.min_draws is not None:
                parts.append(f"已用抽数(不含特殊抽)≥{cond.min_draws}")
            if cond.max_draws is not None:
                parts.append(f"已用抽数(不含特殊抽)≤{cond.max_draws}")
            return " 且 ".join(parts) if parts else "任意已用抽数"
        elif isinstance(cond, GotMainCondition):
            return "已获得UP" if cond.value else "未获得UP"
        elif isinstance(cond, GotHighestRarityCondition):
            return "已获得最高星级" if cond.value else "未获得最高星级"
        elif isinstance(cond, GotHighestRarityButNotMainCondition):
            return "歪了(出最高星级但非UP)" if cond.value else "未歪最高星级"
        elif isinstance(cond, ResourceThresholdCondition):
            parts = []
            if cond.min_normal_draws is not None:
                parts.append(f"可用抽数≥{cond.min_normal_draws}")
            if cond.max_normal_draws is not None:
                parts.append(f"可用抽数≤{cond.max_normal_draws}")
            base = " 且 ".join(parts) if parts else "任意可用抽数"
            if cond.check_once:
                return f"{base}(仅入池时检查)"
            return base
        elif isinstance(cond, GotPityWithoutMainCondition):
            return "歪了(保底未出UP)" if cond.value else "未歪"
        elif isinstance(cond, BannerIndexCondition):
            if cond.every_n <= 1:
                return "每个池子"
            return f"每{cond.every_n}个池子的第{cond.start_at}个"
        elif isinstance(cond, PityCounterCondition):
            parts = []
            if cond.min_pity is not None:
                parts.append(f"保底计数≥{cond.min_pity}")
            if cond.max_pity is not None:
                parts.append(f"保底计数≤{cond.max_pity}")
            return " 且 ".join(parts) if parts else "任意保底计数"
        elif isinstance(cond, DefinitiveDrawCounterCondition):
            parts = []
            if cond.min_definitive is not None:
                parts.append(f"大保底计数≥{cond.min_definitive}")
            if cond.max_definitive is not None:
                parts.append(f"大保底计数≤{cond.max_definitive}")
            return " 且 ".join(parts) if parts else "任意大保底计数"
        return str(cond)

    def _action_to_chinese(
        self,
        action: "StrategyAction",
        strategy_registry: Optional[dict[str, "DrawStrategy"]],
        visited: set[str],
        indent: int,
    ) -> str:
        """Convert an action to Chinese text, handling delegation recursively."""
        if isinstance(action, StopAction):
            base = "停止抽卡"
            if action.pay_override is not None:
                base += f" (氪金: {'是' if action.pay_override else '否'})"
            return base
        elif isinstance(action, ContinueAction):
            parts = []
            if action.min_draws_per_banner > 0:
                parts.append(f"至少抽到{action.min_draws_per_banner}抽")
            if action.max_draws_per_banner and action.max_draws_per_banner > 0:
                parts.append(f"最多抽到{action.max_draws_per_banner}抽")
            if action.stop_on_main:
                parts.append("抽到UP后停止")
            if action.stop_on_highest_rarity:
                parts.append("抽到最高星级后停止")
            if action.target_potential:
                parts.append(f"目标{action.target_potential}潜能")
            if action.target_pity:
                parts.append(f"抽到{action.target_pity}保底计数")
            if action.target_definitive_draw:
                parts.append(f"抽到{action.target_definitive_draw}大保底计数")
            if action.pay_override is not None:
                parts.append(f"氪金: {'是' if action.pay_override else '否'}")
            return "继续抽卡" + (f" ({', '.join(parts)})" if parts else "")
        elif isinstance(action, DelegateAction):
            desc = f"执行策略「{action.strategy_name}」"
            # Recursively describe the delegated strategy
            if strategy_registry and action.strategy_name in strategy_registry:
                target = strategy_registry[action.strategy_name]
                nested_desc = target.get_description(strategy_registry, visited, indent)
                desc += f"\n{nested_desc}"
            elif action.strategy_name in visited:
                desc += " (循环引用)"
            return desc
        return str(action)

    def _rule_to_chinese_parts(
        self,
        rule: "StrategyRule",
        strategy_registry: Optional[dict[str, "DrawStrategy"]],
        visited: set[str],
        indent: int,
    ) -> tuple[str, str]:
        """Convert a rule to Chinese text parts (condition, action).

        Returns:
            Tuple of (conditions_str, action_str)
        """
        # Conditions
        if rule.conditions:
            cond_texts = [self._condition_to_chinese(c) for c in rule.conditions]
            conditions_str = " 且 ".join(cond_texts)
        else:
            conditions_str = "始终"

        # Action
        action_str = self._action_to_chinese(
            rule.action, strategy_registry, visited, indent
        )

        return conditions_str, action_str


# =============================================================================
# Deserialization Helper
# =============================================================================


def is_legacy_strategy(data: dict) -> bool:
    """Check if a strategy dict is in legacy format.

    Args:
        data: Dictionary representation of a strategy

    Returns:
        True if legacy format, False if new format
    """
    # New format has 'behavior' or 'rules' keys
    return "behavior" not in data and "rules" not in data


def deserialize_strategy(data: dict) -> DrawStrategy:
    """Deserialize a strategy from dict.

    Args:
        data: Dictionary representation of a strategy (must be new format)

    Returns:
        DrawStrategy instance

    Raises:
        ValueError: If data is in legacy format
    """
    if is_legacy_strategy(data):
        raise ValueError(
            f"Legacy strategy format not supported: {data.get('name', 'unknown')}"
        )
    return DrawStrategy(**data)
