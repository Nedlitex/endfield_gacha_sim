"""Strategy selector and editor component with advanced rule-based UI."""

import streamlit as st

from strategy import (
    BannerIndexCondition,
    ContinueAction,
    DefinitiveDrawCounterCondition,
    DelegateAction,
    DrawBehavior,
    DrawCountCondition,
    DrawStrategy,
    GotHighestRarityButNotMainCondition,
    GotHighestRarityCondition,
    GotMainCondition,
    GotPityWithoutMainCondition,
    PityCounterCondition,
    ResourceThresholdCondition,
    StopAction,
    StrategyCondition,
    StrategyRule,
)
from ui.components.st_horizontal import st_horizontal
from ui.state import update_url


def _get_current_strategy() -> DrawStrategy:
    """Get the currently selected strategy."""
    return st.session_state.strategies[st.session_state.current_strategy_idx]


def _get_other_strategy_names() -> list[str]:
    """Get names of other strategies (for delegation)."""
    current = _get_current_strategy()
    return [s.name for s in st.session_state.strategies if s.name != current.name]


def _reassign_rule_priorities(strategy: DrawStrategy):
    """Reassign priorities to rules based on their current order.

    Rules are ordered from highest to lowest priority in the list,
    so the first rule gets the highest priority.
    """
    num_rules = len(strategy.rules)
    for i, rule in enumerate(strategy.rules):
        # First rule gets highest priority, last gets lowest
        rule.priority = (num_rules - i) * 10


def render_strategy_section():
    """Render the strategy selector and editor."""
    # Strategy selector
    st.subheader("抽卡策略")
    strategy_names = [s.name for s in st.session_state.strategies]
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_strategy_idx = st.selectbox(
            "选择策略",
            range(len(strategy_names)),
            index=st.session_state.current_strategy_idx,
            format_func=lambda x: strategy_names[x],
        )
        if selected_strategy_idx != st.session_state.current_strategy_idx:
            st.session_state.current_strategy_idx = selected_strategy_idx
            update_url()
            st.rerun()
    with col2:
        pass  # Delete button moved to strategy creation section

    current_strategy = _get_current_strategy()

    # Check if this is the default strategy (read-only)
    is_default_strategy = (
        st.session_state.current_strategy_idx == 0
        and current_strategy.name.startswith("默认策略")
    )

    # Strategy settings inside an expandable container
    with st.expander(f"策略配置: {current_strategy.name}", expanded=False):
        if is_default_strategy:
            # Show read-only view for default strategy
            st.info(
                "这是默认策略，不可编辑。点击下方「创建新策略」按钮创建自定义策略。"
            )
            st.markdown("**策略说明:** 氪金抽到UP（抽数不足时额外获得抽数以满足规则）")
        else:
            _render_strategy_editor(current_strategy)

    # New strategy creation (after the expander)
    _render_strategy_creation()


def _render_strategy_editor(current_strategy: DrawStrategy):
    """Render the advanced strategy editor."""
    strategy_key_prefix = f"strategy_{st.session_state.current_strategy_idx}_"

    # === Name Section ===
    st.markdown("### 策略名称")
    existing_names = [
        s.name
        for i, s in enumerate(st.session_state.strategies)
        if i != st.session_state.current_strategy_idx
    ]
    new_name = st.text_input(
        "名称",
        value=current_strategy.name,
        key=f"{strategy_key_prefix}name",
        help="策略名称，用于区分不同策略",
    )
    if new_name and new_name != current_strategy.name:
        if new_name in existing_names:
            st.error(f"策略名称「{new_name}」已存在，请使用其他名称")
        else:
            current_strategy.name = new_name
            update_url()
            st.rerun()

    # === Behavior Section ===
    st.markdown("### 抽卡行为")
    always_single = st.checkbox(
        "始终单抽",
        value=current_strategy.behavior.always_single_draw,
        key=f"{strategy_key_prefix}always_single_draw",
        help="始终单抽(特殊10连除外)",
    )
    single_after = st.number_input(
        "累计抽数后单抽",
        min_value=0,
        value=current_strategy.behavior.single_draw_after,
        step=1,
        key=f"{strategy_key_prefix}single_draw_after",
        help="累计抽数达到此值后开始单抽(0=不启用)",
    )
    pay = st.checkbox(
        "氪金",
        value=current_strategy.behavior.pay,
        key=f"{strategy_key_prefix}pay",
        help="抽数不足时额外获得抽数以满足规则",
    )

    # Update behavior if changed
    new_behavior = DrawBehavior(
        always_single_draw=always_single,
        single_draw_after=single_after,
        pay=pay,
    )
    if new_behavior != current_strategy.behavior:
        current_strategy.behavior = new_behavior
        update_url()

    # === Rules Section ===
    st.markdown("### 策略规则")
    st.caption("规则按优先级从高到低执行，第一个匹配的规则生效")

    # Render existing rules
    rules_to_remove = []
    rule_to_move_up = None
    rule_to_move_down = None

    for idx, rule in enumerate(current_strategy.rules):
        with st.container():
            st.markdown(f"**规则 {idx + 1}** (优先级: {rule.priority})")
            _render_rule_summary(rule)

            with st_horizontal():
                if idx > 0:
                    if st.button(
                        "↑", key=f"{strategy_key_prefix}move_up_{idx}", help="上移"
                    ):
                        rule_to_move_up = idx

                if idx < len(current_strategy.rules) - 1:
                    if st.button(
                        "↓", key=f"{strategy_key_prefix}move_down_{idx}", help="下移"
                    ):
                        rule_to_move_down = idx

                edit_key = f"{strategy_key_prefix}editing_rule_{idx}"
                if st.button("编辑", key=f"{strategy_key_prefix}edit_rule_{idx}"):
                    st.session_state[edit_key] = True

                if st.button("删除", key=f"{strategy_key_prefix}delete_rule_{idx}"):
                    rules_to_remove.append(idx)

            # Show editor if editing this rule
            if st.session_state.get(f"{strategy_key_prefix}editing_rule_{idx}", False):
                _render_existing_rule_editor(
                    current_strategy, idx, f"{strategy_key_prefix}rule_{idx}_"
                )

            st.divider()

    # Handle move up - swap with previous rule and adjust priorities
    if rule_to_move_up is not None:
        idx = rule_to_move_up
        # Swap positions in list
        current_strategy.rules[idx], current_strategy.rules[idx - 1] = (
            current_strategy.rules[idx - 1],
            current_strategy.rules[idx],
        )
        # Update priorities to match new order (higher index = lower priority)
        _reassign_rule_priorities(current_strategy)
        update_url()
        st.rerun()

    # Handle move down - swap with next rule and adjust priorities
    if rule_to_move_down is not None:
        idx = rule_to_move_down
        # Swap positions in list
        current_strategy.rules[idx], current_strategy.rules[idx + 1] = (
            current_strategy.rules[idx + 1],
            current_strategy.rules[idx],
        )
        # Update priorities to match new order
        _reassign_rule_priorities(current_strategy)
        update_url()
        st.rerun()

    # Remove rules marked for deletion
    if rules_to_remove:
        for idx in sorted(rules_to_remove, reverse=True):
            current_strategy.rules.pop(idx)
        update_url()
        st.rerun()

    # Add new rule button
    if st.button("添加规则", key=f"{strategy_key_prefix}add_rule"):
        st.session_state[f"{strategy_key_prefix}adding_rule"] = True

    # New rule editor
    if st.session_state.get(f"{strategy_key_prefix}adding_rule", False):
        _render_new_rule_editor(current_strategy, strategy_key_prefix)

    # === Default Action Section ===
    st.markdown("### 默认行为")
    st.caption("当没有规则匹配时执行此行为")
    _render_default_action_editor(current_strategy, strategy_key_prefix)


def _render_rule_summary(rule: StrategyRule):
    """Render a summary of a rule."""
    # Conditions
    if rule.conditions:
        cond_texts = []
        for cond in rule.conditions:
            cond_texts.append(_condition_to_text(cond))
        conditions_str = " 且 ".join(cond_texts)
    else:
        conditions_str = "始终"

    # Action
    action_str = _action_to_text(rule.action)

    st.markdown(f"**条件:** {conditions_str}")
    st.markdown(f"**动作:** {action_str}")


def _condition_to_text(cond: StrategyCondition) -> str:
    """Convert a condition to human-readable text."""
    if isinstance(cond, DrawCountCondition):
        parts = []
        if cond.min_draws is not None:
            parts.append(f"已用抽数(不含特殊抽)>={cond.min_draws}")
        if cond.max_draws is not None:
            parts.append(f"已用抽数(不含特殊抽)<={cond.max_draws}")
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
            parts.append(f"可用抽数>={cond.min_normal_draws}")
        if cond.max_normal_draws is not None:
            parts.append(f"可用抽数<={cond.max_normal_draws}")
        base = " 且 ".join(parts) if parts else "任意可用抽数"
        if cond.check_once:
            return f"{base}(仅入池时检查)"
        return base
    elif isinstance(cond, GotPityWithoutMainCondition):
        return "歪了(保底未出UP)" if cond.value else "未歪"
    elif isinstance(cond, BannerIndexCondition):
        if cond.every_n == 0:
            return f"仅第{cond.start_at}个池子"
        if cond.every_n == 1:
            return "每个池子"
        return f"每{cond.every_n}个池子的第{cond.start_at}个"
    elif isinstance(cond, PityCounterCondition):
        parts = []
        if cond.min_pity is not None:
            parts.append(f"小保底计数>={cond.min_pity}")
        if cond.max_pity is not None:
            parts.append(f"小保底计数<={cond.max_pity}")
        return " 且 ".join(parts) if parts else "任意小保底计数"
    elif isinstance(cond, DefinitiveDrawCounterCondition):
        parts = []
        if cond.min_definitive is not None:
            parts.append(f"大保底计数>={cond.min_definitive}")
        if cond.max_definitive is not None:
            parts.append(f"大保底计数<={cond.max_definitive}")
        return " 且 ".join(parts) if parts else "任意大保底计数"
    return str(cond)


def _action_to_text(action) -> str:
    """Convert an action to human-readable text."""
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
        return f"执行策略「{action.strategy_name}」"
    return str(action)


def _render_new_rule_editor(current_strategy: DrawStrategy, prefix: str):
    """Render the new rule editor."""
    st.markdown("#### 添加新规则")

    # Priority
    priority = st.number_input(
        "优先级",
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        key=f"{prefix}new_rule_priority",
        help="数值越大优先级越高",
    )

    # Conditions
    st.markdown("**条件 (全部满足时触发)**")

    # Draw count condition (draws used on current banner, excluding special draws)
    use_draw_count = st.checkbox(
        "已用抽数条件",
        key=f"{prefix}new_rule_use_draw_count",
        help="当前池已消耗的抽数(不含特殊抽)",
    )
    draw_count_min = None
    draw_count_max = None
    if use_draw_count:
        col1, col2 = st.columns(2)
        with col1:
            draw_count_min = st.number_input(
                "最小已用抽数",
                min_value=0,
                value=0,
                step=1,
                key=f"{prefix}new_rule_draw_count_min",
            )
            if draw_count_min == 0:
                draw_count_min = None
        with col2:
            draw_count_max = st.number_input(
                "最大已用抽数",
                min_value=0,
                value=0,
                step=1,
                key=f"{prefix}new_rule_draw_count_max",
                help="0表示不限制",
            )
            if draw_count_max == 0:
                draw_count_max = None

    # Got main condition
    use_got_main = st.checkbox("UP获取条件", key=f"{prefix}new_rule_use_got_main")
    got_main_value = True
    if use_got_main:
        got_main_value = st.radio(
            "UP状态",
            [True, False],
            format_func=lambda x: "已获得UP" if x else "未获得UP",
            key=f"{prefix}new_rule_got_main_value",
            horizontal=True,
        )

    # Got highest rarity condition
    use_got_highest_rarity = st.checkbox(
        "最高星级获取条件", key=f"{prefix}new_rule_use_got_highest_rarity"
    )
    got_highest_rarity_value = True
    if use_got_highest_rarity:
        got_highest_rarity_value = st.radio(
            "最高星级状态",
            [True, False],
            format_func=lambda x: "已获得最高星级" if x else "未获得最高星级",
            key=f"{prefix}new_rule_got_highest_rarity_value",
            horizontal=True,
        )

    # Got highest rarity but not main condition
    use_got_hr_not_main = st.checkbox(
        "歪最高星级条件", key=f"{prefix}new_rule_use_got_hr_not_main"
    )
    got_hr_not_main_value = True
    if use_got_hr_not_main:
        got_hr_not_main_value = st.radio(
            "歪最高星级状态",
            [True, False],
            format_func=lambda x: "歪了(出最高星级但非UP)" if x else "未歪最高星级",
            key=f"{prefix}new_rule_got_hr_not_main_value",
            horizontal=True,
        )

    # Resource threshold condition (available draws that carry over)
    use_resource = st.checkbox(
        "可用抽数条件",
        key=f"{prefix}new_rule_use_resource",
        help="可跨池继承的剩余抽数",
    )
    resource_min = None
    resource_max = None
    resource_check_once = False
    if use_resource:
        col1, col2 = st.columns(2)
        with col1:
            resource_min = st.number_input(
                "最小可用抽数",
                min_value=0,
                value=0,
                step=1,
                key=f"{prefix}new_rule_resource_min",
            )
            if resource_min == 0:
                resource_min = None
        with col2:
            resource_max = st.number_input(
                "最大可用抽数",
                min_value=0,
                value=0,
                step=1,
                key=f"{prefix}new_rule_resource_max",
                help="0表示不限制",
            )
            if resource_max == 0:
                resource_max = None
        resource_check_once = st.checkbox(
            "仅入池时检查",
            key=f"{prefix}new_rule_resource_check_once",
            help="勾选后仅在进入池子时检查一次，之后不再检查",
        )

    # Pity without main condition
    use_pity = st.checkbox("歪了条件", key=f"{prefix}new_rule_use_pity")
    pity_value = True
    if use_pity:
        pity_value = st.radio(
            "歪了状态",
            [True, False],
            format_func=lambda x: "歪了(保底未出UP)" if x else "未歪",
            key=f"{prefix}new_rule_pity_value",
            horizontal=True,
        )

    # Banner index condition
    use_banner_index = st.checkbox(
        "池子序号条件",
        key=f"{prefix}new_rule_use_banner_index",
        help="仅在特定序号的池子生效(从第1个池子开始计数)",
    )
    banner_every_n = 0
    banner_start_at = 1
    if use_banner_index:
        col1, col2 = st.columns(2)
        with col1:
            banner_every_n = st.number_input(
                "每N个池子",
                min_value=0,
                value=2,
                step=1,
                key=f"{prefix}new_rule_banner_every_n",
                help="0=仅指定池子, 1=每个池子, 2=每2个池子中选1个",
            )
        with col2:
            # When every_n=0, allow any start_at (specific banner)
            # When every_n=1, start_at doesn't matter (every banner)
            # When every_n>=2, limit start_at to every_n
            max_start = (
                999
                if banner_every_n == 0
                else (banner_every_n if banner_every_n > 1 else 1)
            )
            banner_start_at = st.number_input(
                "第几个池子" if banner_every_n == 0 else "从第几个池子开始",
                min_value=1,
                max_value=max(1, max_start),
                value=1,
                step=1,
                key=f"{prefix}new_rule_banner_start_at",
                help=(
                    "指定池子序号"
                    if banner_every_n == 0
                    else "1=第1,3,5...个, 2=第2,4,6...个(每2池时)"
                ),
            )

    # Pity counter condition
    use_pity_counter = st.checkbox(
        "保底计数条件",
        key=f"{prefix}new_rule_use_pity_counter",
        help="距离上次出最高星级的抽数",
    )
    pity_counter_min = None
    pity_counter_max = None
    if use_pity_counter:
        col1, col2 = st.columns(2)
        with col1:
            pity_counter_min = st.number_input(
                "最小保底计数",
                min_value=0,
                value=0,
                step=1,
                key=f"{prefix}new_rule_pity_counter_min",
            )
            if pity_counter_min == 0:
                pity_counter_min = None
        with col2:
            pity_counter_max = st.number_input(
                "最大保底计数",
                min_value=0,
                value=0,
                step=1,
                key=f"{prefix}new_rule_pity_counter_max",
                help="0表示不限制",
            )
            if pity_counter_max == 0:
                pity_counter_max = None

    # Definitive draw counter condition
    use_definitive_counter = st.checkbox(
        "大保底计数条件",
        key=f"{prefix}new_rule_use_definitive_counter",
        help="距离大保底(必得UP)的抽数",
    )
    definitive_counter_min = None
    definitive_counter_max = None
    if use_definitive_counter:
        col1, col2 = st.columns(2)
        with col1:
            definitive_counter_min = st.number_input(
                "最小大保底计数",
                min_value=0,
                value=0,
                step=1,
                key=f"{prefix}new_rule_definitive_counter_min",
            )
            if definitive_counter_min == 0:
                definitive_counter_min = None
        with col2:
            definitive_counter_max = st.number_input(
                "最大大保底计数",
                min_value=0,
                value=0,
                step=1,
                key=f"{prefix}new_rule_definitive_counter_max",
                help="0表示不限制",
            )
            if definitive_counter_max == 0:
                definitive_counter_max = None

    # Action
    st.markdown("**动作**")
    action_type = st.selectbox(
        "动作类型",
        ["stop", "continue", "delegate"],
        format_func=lambda x: {
            "stop": "停止抽卡",
            "continue": "继续抽卡",
            "delegate": "执行其他策略",
        }[x],
        key=f"{prefix}new_rule_action_type",
    )

    action = None
    if action_type == "stop":
        stop_pay_override = st.selectbox(
            "氪金覆盖",
            [None, True, False],
            format_func=lambda x: {
                None: "使用策略默认",
                True: "强制氪金",
                False: "强制不氪金",
            }[x],
            key=f"{prefix}new_rule_stop_pay_override",
            help="覆盖策略的默认氪金设置",
        )
        action = StopAction(pay_override=stop_pay_override)
    elif action_type == "continue":
        col1, col2 = st.columns(2)
        with col1:
            min_draws = st.number_input(
                "至少抽到",
                min_value=0,
                value=0,
                step=1,
                key=f"{prefix}new_rule_action_min_draws",
                help="抽到此数量前不会停止",
            )
        with col2:
            max_draws = st.number_input(
                "最多抽到",
                min_value=0,
                value=0,
                step=1,
                key=f"{prefix}new_rule_action_max_draws",
                help="抽到此数量后停止(0=不限制)",
            )

        col3, col4 = st.columns(2)
        with col3:
            stop_on_main = st.checkbox(
                "抽到UP后停止",
                key=f"{prefix}new_rule_action_stop_on_main",
            )
        with col4:
            stop_on_highest_rarity = st.checkbox(
                "抽到最高星级后停止",
                key=f"{prefix}new_rule_action_stop_on_highest_rarity",
            )

        col5, col6 = st.columns(2)
        with col5:
            target_potential = st.number_input(
                "目标潜能",
                min_value=0,
                max_value=6,
                value=0,
                step=1,
                key=f"{prefix}new_rule_action_target_potential",
                help="0表示不限制潜能",
            )
        with col6:
            target_pity = st.number_input(
                "目标保底计数",
                min_value=0,
                value=0,
                step=1,
                key=f"{prefix}new_rule_action_target_pity",
                help="抽到此保底计数后停止(0=不限制)",
            )

        col7, col8 = st.columns(2)
        with col7:
            target_definitive = st.number_input(
                "目标大保底计数",
                min_value=0,
                value=0,
                step=1,
                key=f"{prefix}new_rule_action_target_definitive",
                help="抽到此大保底计数后停止(0=不限制)",
            )
        with col8:
            continue_pay_override = st.selectbox(
                "氪金覆盖",
                [None, True, False],
                format_func=lambda x: {
                    None: "使用策略默认",
                    True: "强制氪金",
                    False: "强制不氪金",
                }[x],
                key=f"{prefix}new_rule_continue_pay_override",
                help="覆盖策略的默认氪金设置",
            )

        action = ContinueAction(
            min_draws_per_banner=min_draws,
            max_draws_per_banner=max_draws if max_draws > 0 else None,
            stop_on_main=stop_on_main,
            stop_on_highest_rarity=stop_on_highest_rarity,
            target_potential=target_potential if target_potential > 0 else None,
            target_pity=target_pity if target_pity > 0 else None,
            target_definitive_draw=target_definitive if target_definitive > 0 else None,
            pay_override=continue_pay_override,
        )
    elif action_type == "delegate":
        other_strategies = _get_other_strategy_names()
        if other_strategies:
            delegate_to = st.selectbox(
                "执行策略",
                other_strategies,
                key=f"{prefix}new_rule_action_delegate_to",
            )
            action = DelegateAction(strategy_name=delegate_to)
        else:
            st.warning("没有其他可用策略")
            action = None

    # Buttons
    with st_horizontal():
        if st.button("确认添加", key=f"{prefix}confirm_add_rule"):
            if action is not None:
                # Build conditions
                conditions: list[StrategyCondition] = []
                if use_draw_count and (
                    draw_count_min is not None or draw_count_max is not None
                ):
                    conditions.append(
                        DrawCountCondition(
                            min_draws=draw_count_min,
                            max_draws=draw_count_max,
                        )
                    )
                if use_got_main:
                    conditions.append(GotMainCondition(value=got_main_value))
                if use_got_highest_rarity:
                    conditions.append(
                        GotHighestRarityCondition(value=got_highest_rarity_value)
                    )
                if use_got_hr_not_main:
                    conditions.append(
                        GotHighestRarityButNotMainCondition(value=got_hr_not_main_value)
                    )
                if use_resource and (
                    resource_min is not None or resource_max is not None
                ):
                    conditions.append(
                        ResourceThresholdCondition(
                            min_normal_draws=resource_min,
                            max_normal_draws=resource_max,
                            check_once=resource_check_once,
                        )
                    )
                if use_pity:
                    conditions.append(GotPityWithoutMainCondition(value=pity_value))
                if use_banner_index:
                    conditions.append(
                        BannerIndexCondition(
                            every_n=banner_every_n, start_at=banner_start_at
                        )
                    )
                if use_pity_counter and (
                    pity_counter_min is not None or pity_counter_max is not None
                ):
                    conditions.append(
                        PityCounterCondition(
                            min_pity=pity_counter_min,
                            max_pity=pity_counter_max,
                        )
                    )
                if use_definitive_counter and (
                    definitive_counter_min is not None
                    or definitive_counter_max is not None
                ):
                    conditions.append(
                        DefinitiveDrawCounterCondition(
                            min_definitive=definitive_counter_min,
                            max_definitive=definitive_counter_max,
                        )
                    )

                # Create rule
                new_rule = StrategyRule(
                    conditions=conditions,
                    action=action,
                    priority=priority,
                )
                current_strategy.rules.append(new_rule)
                # Sort by priority
                current_strategy.rules.sort(key=lambda r: -r.priority)
                st.session_state[f"{prefix}adding_rule"] = False
                update_url()
                st.rerun()
        if st.button("取消", key=f"{prefix}cancel_add_rule"):
            st.session_state[f"{prefix}adding_rule"] = False
            st.rerun()


def _render_existing_rule_editor(
    current_strategy: DrawStrategy, rule_idx: int, rule_prefix: str
):
    """Render editor for an existing rule."""
    rule = current_strategy.rules[rule_idx]
    st.markdown("#### 编辑规则")

    # Extract current condition values
    current_draw_count = next(
        (c for c in rule.conditions if isinstance(c, DrawCountCondition)), None
    )
    current_got_main = next(
        (c for c in rule.conditions if isinstance(c, GotMainCondition)), None
    )
    current_got_hr = next(
        (c for c in rule.conditions if isinstance(c, GotHighestRarityCondition)), None
    )
    current_got_hr_not_main = next(
        (
            c
            for c in rule.conditions
            if isinstance(c, GotHighestRarityButNotMainCondition)
        ),
        None,
    )
    current_resource = next(
        (c for c in rule.conditions if isinstance(c, ResourceThresholdCondition)), None
    )
    current_pity_without_main = next(
        (c for c in rule.conditions if isinstance(c, GotPityWithoutMainCondition)), None
    )
    current_banner_index = next(
        (c for c in rule.conditions if isinstance(c, BannerIndexCondition)), None
    )
    current_pity_counter = next(
        (c for c in rule.conditions if isinstance(c, PityCounterCondition)), None
    )
    current_definitive_counter = next(
        (c for c in rule.conditions if isinstance(c, DefinitiveDrawCounterCondition)),
        None,
    )

    # Priority
    priority = st.number_input(
        "优先级",
        min_value=0,
        max_value=100,
        value=rule.priority,
        step=1,
        key=f"{rule_prefix}priority",
        help="数值越大优先级越高",
    )

    # Conditions
    st.markdown("**条件 (全部满足时触发)**")

    # Draw count condition
    use_draw_count = st.checkbox(
        "已用抽数条件",
        value=current_draw_count is not None,
        key=f"{rule_prefix}use_draw_count",
        help="当前池已消耗的抽数(不含特殊抽)",
    )
    draw_count_min = None
    draw_count_max = None
    if use_draw_count:
        col1, col2 = st.columns(2)
        with col1:
            draw_count_min = st.number_input(
                "最小已用抽数",
                min_value=0,
                value=(
                    current_draw_count.min_draws
                    if current_draw_count and current_draw_count.min_draws
                    else 0
                ),
                step=1,
                key=f"{rule_prefix}draw_count_min",
            )
            if draw_count_min == 0:
                draw_count_min = None
        with col2:
            draw_count_max = st.number_input(
                "最大已用抽数",
                min_value=0,
                value=(
                    current_draw_count.max_draws
                    if current_draw_count and current_draw_count.max_draws
                    else 0
                ),
                step=1,
                key=f"{rule_prefix}draw_count_max",
                help="0表示不限制",
            )
            if draw_count_max == 0:
                draw_count_max = None

    # Got main condition
    use_got_main = st.checkbox(
        "UP获取条件",
        value=current_got_main is not None,
        key=f"{rule_prefix}use_got_main",
    )
    got_main_value = True
    if use_got_main:
        got_main_value = st.radio(
            "UP状态",
            [True, False],
            index=0 if (current_got_main is None or current_got_main.value) else 1,
            format_func=lambda x: "已获得UP" if x else "未获得UP",
            key=f"{rule_prefix}got_main_value",
            horizontal=True,
        )

    # Got highest rarity condition
    use_got_highest_rarity = st.checkbox(
        "最高星级获取条件",
        value=current_got_hr is not None,
        key=f"{rule_prefix}use_got_highest_rarity",
    )
    got_highest_rarity_value = True
    if use_got_highest_rarity:
        got_highest_rarity_value = st.radio(
            "最高星级状态",
            [True, False],
            index=0 if (current_got_hr is None or current_got_hr.value) else 1,
            format_func=lambda x: "已获得最高星级" if x else "未获得最高星级",
            key=f"{rule_prefix}got_highest_rarity_value",
            horizontal=True,
        )

    # Got highest rarity but not main condition
    use_got_hr_not_main = st.checkbox(
        "歪最高星级条件",
        value=current_got_hr_not_main is not None,
        key=f"{rule_prefix}use_got_hr_not_main",
    )
    got_hr_not_main_value = True
    if use_got_hr_not_main:
        got_hr_not_main_value = st.radio(
            "歪最高星级状态",
            [True, False],
            index=(
                0
                if (current_got_hr_not_main is None or current_got_hr_not_main.value)
                else 1
            ),
            format_func=lambda x: "歪了(出最高星级但非UP)" if x else "未歪最高星级",
            key=f"{rule_prefix}got_hr_not_main_value",
            horizontal=True,
        )

    # Resource threshold condition
    use_resource = st.checkbox(
        "可用抽数条件",
        value=current_resource is not None,
        key=f"{rule_prefix}use_resource",
        help="可跨池继承的剩余抽数",
    )
    resource_min = None
    resource_max = None
    resource_check_once = False
    if use_resource:
        col1, col2 = st.columns(2)
        with col1:
            resource_min = st.number_input(
                "最小可用抽数",
                min_value=0,
                value=(
                    current_resource.min_normal_draws
                    if current_resource and current_resource.min_normal_draws
                    else 0
                ),
                step=1,
                key=f"{rule_prefix}resource_min",
            )
            if resource_min == 0:
                resource_min = None
        with col2:
            resource_max = st.number_input(
                "最大可用抽数",
                min_value=0,
                value=(
                    current_resource.max_normal_draws
                    if current_resource and current_resource.max_normal_draws
                    else 0
                ),
                step=1,
                key=f"{rule_prefix}resource_max",
                help="0表示不限制",
            )
            if resource_max == 0:
                resource_max = None
        resource_check_once = st.checkbox(
            "仅入池时检查",
            value=current_resource.check_once if current_resource else False,
            key=f"{rule_prefix}resource_check_once",
            help="勾选后仅在进入池子时检查一次，之后不再检查",
        )

    # Pity without main condition
    use_pity = st.checkbox(
        "歪了条件",
        value=current_pity_without_main is not None,
        key=f"{rule_prefix}use_pity",
    )
    pity_value = True
    if use_pity:
        pity_value = st.radio(
            "歪了状态",
            [True, False],
            index=(
                0
                if (
                    current_pity_without_main is None or current_pity_without_main.value
                )
                else 1
            ),
            format_func=lambda x: "歪了(保底未出UP)" if x else "未歪",
            key=f"{rule_prefix}pity_value",
            horizontal=True,
        )

    # Banner index condition
    use_banner_index = st.checkbox(
        "池子序号条件",
        value=current_banner_index is not None,
        key=f"{rule_prefix}use_banner_index",
        help="仅在特定序号的池子生效(从第1个池子开始计数)",
    )
    banner_every_n = 0
    banner_start_at = 1
    if use_banner_index:
        col1, col2 = st.columns(2)
        with col1:
            banner_every_n = st.number_input(
                "每N个池子",
                min_value=0,
                value=current_banner_index.every_n if current_banner_index else 2,
                step=1,
                key=f"{rule_prefix}banner_every_n",
                help="0=仅指定池子, 1=每个池子, 2=每2个池子中选1个",
            )
        with col2:
            # When every_n=0, allow any start_at (specific banner)
            # When every_n=1, start_at doesn't matter (every banner)
            # When every_n>=2, limit start_at to every_n
            max_start = (
                999
                if banner_every_n == 0
                else (banner_every_n if banner_every_n > 1 else 1)
            )
            banner_start_at = st.number_input(
                "第几个池子" if banner_every_n == 0 else "从第几个池子开始",
                min_value=1,
                max_value=max(1, max_start),
                value=min(
                    current_banner_index.start_at if current_banner_index else 1,
                    max(1, max_start),
                ),
                step=1,
                key=f"{rule_prefix}banner_start_at",
                help=(
                    "指定池子序号"
                    if banner_every_n == 0
                    else "1=第1,3,5...个, 2=第2,4,6...个(每2池时)"
                ),
            )

    # Pity counter condition
    use_pity_counter = st.checkbox(
        "保底计数条件",
        value=current_pity_counter is not None,
        key=f"{rule_prefix}use_pity_counter",
        help="距离上次出最高星级的抽数",
    )
    pity_counter_min = None
    pity_counter_max = None
    if use_pity_counter:
        col1, col2 = st.columns(2)
        with col1:
            pity_counter_min = st.number_input(
                "最小保底计数",
                min_value=0,
                value=(
                    current_pity_counter.min_pity
                    if current_pity_counter and current_pity_counter.min_pity
                    else 0
                ),
                step=1,
                key=f"{rule_prefix}pity_counter_min",
            )
            if pity_counter_min == 0:
                pity_counter_min = None
        with col2:
            pity_counter_max = st.number_input(
                "最大保底计数",
                min_value=0,
                value=(
                    current_pity_counter.max_pity
                    if current_pity_counter and current_pity_counter.max_pity
                    else 0
                ),
                step=1,
                key=f"{rule_prefix}pity_counter_max",
                help="0表示不限制",
            )
            if pity_counter_max == 0:
                pity_counter_max = None

    # Definitive draw counter condition
    use_definitive_counter = st.checkbox(
        "大保底计数条件",
        value=current_definitive_counter is not None,
        key=f"{rule_prefix}use_definitive_counter",
        help="距离大保底(必得UP)的抽数",
    )
    definitive_counter_min = None
    definitive_counter_max = None
    if use_definitive_counter:
        col1, col2 = st.columns(2)
        with col1:
            definitive_counter_min = st.number_input(
                "最小大保底计数",
                min_value=0,
                value=(
                    current_definitive_counter.min_definitive
                    if current_definitive_counter
                    and current_definitive_counter.min_definitive
                    else 0
                ),
                step=1,
                key=f"{rule_prefix}definitive_counter_min",
            )
            if definitive_counter_min == 0:
                definitive_counter_min = None
        with col2:
            definitive_counter_max = st.number_input(
                "最大大保底计数",
                min_value=0,
                value=(
                    current_definitive_counter.max_definitive
                    if current_definitive_counter
                    and current_definitive_counter.max_definitive
                    else 0
                ),
                step=1,
                key=f"{rule_prefix}definitive_counter_max",
                help="0表示不限制",
            )
            if definitive_counter_max == 0:
                definitive_counter_max = None

    # Action
    st.markdown("**动作**")
    current_action = rule.action
    action_type_index = (
        0
        if isinstance(current_action, StopAction)
        else (1 if isinstance(current_action, ContinueAction) else 2)
    )
    action_type = st.selectbox(
        "动作类型",
        ["stop", "continue", "delegate"],
        index=action_type_index,
        format_func=lambda x: {
            "stop": "停止抽卡",
            "continue": "继续抽卡",
            "delegate": "执行其他策略",
        }[x],
        key=f"{rule_prefix}action_type",
    )

    new_action = None
    if action_type == "stop":
        current_pay_override = (
            current_action.pay_override
            if isinstance(current_action, StopAction)
            else None
        )
        pay_override_options = [None, True, False]
        stop_pay_override = st.selectbox(
            "氪金覆盖",
            pay_override_options,
            index=(
                pay_override_options.index(current_pay_override)
                if current_pay_override in pay_override_options
                else 0
            ),
            format_func=lambda x: {
                None: "使用策略默认",
                True: "强制氪金",
                False: "强制不氪金",
            }[x],
            key=f"{rule_prefix}stop_pay_override",
            help="覆盖策略的默认氪金设置",
        )
        new_action = StopAction(pay_override=stop_pay_override)
    elif action_type == "continue":
        # Get current values if action is ContinueAction
        curr_min = (
            current_action.min_draws_per_banner
            if isinstance(current_action, ContinueAction)
            else 0
        )
        curr_max = (
            current_action.max_draws_per_banner
            if isinstance(current_action, ContinueAction)
            else None
        )
        curr_stop_main = (
            current_action.stop_on_main
            if isinstance(current_action, ContinueAction)
            else False
        )
        curr_stop_hr = (
            current_action.stop_on_highest_rarity
            if isinstance(current_action, ContinueAction)
            else False
        )
        curr_potential = (
            current_action.target_potential
            if isinstance(current_action, ContinueAction)
            else None
        )
        curr_target_pity = (
            current_action.target_pity
            if isinstance(current_action, ContinueAction)
            else None
        )
        curr_target_def = (
            current_action.target_definitive_draw
            if isinstance(current_action, ContinueAction)
            else None
        )
        curr_pay = (
            current_action.pay_override
            if isinstance(current_action, ContinueAction)
            else None
        )

        col1, col2 = st.columns(2)
        with col1:
            min_draws = st.number_input(
                "至少抽到",
                min_value=0,
                value=curr_min,
                step=1,
                key=f"{rule_prefix}action_min_draws",
                help="抽到此数量前不会停止",
            )
        with col2:
            max_draws = st.number_input(
                "最多抽到",
                min_value=0,
                value=curr_max if curr_max else 0,
                step=1,
                key=f"{rule_prefix}action_max_draws",
                help="抽到此数量后停止(0=不限制)",
            )

        col3, col4 = st.columns(2)
        with col3:
            stop_on_main = st.checkbox(
                "抽到UP后停止",
                value=curr_stop_main,
                key=f"{rule_prefix}action_stop_on_main",
            )
        with col4:
            stop_on_highest_rarity = st.checkbox(
                "抽到最高星级后停止",
                value=curr_stop_hr,
                key=f"{rule_prefix}action_stop_on_highest_rarity",
            )

        col5, col6 = st.columns(2)
        with col5:
            target_potential = st.number_input(
                "目标潜能",
                min_value=0,
                max_value=6,
                value=curr_potential if curr_potential else 0,
                step=1,
                key=f"{rule_prefix}action_target_potential",
                help="0表示不限制潜能",
            )
        with col6:
            target_pity = st.number_input(
                "目标保底计数",
                min_value=0,
                value=curr_target_pity if curr_target_pity else 0,
                step=1,
                key=f"{rule_prefix}action_target_pity",
                help="抽到此保底计数后停止(0=不限制)",
            )

        col7, col8 = st.columns(2)
        with col7:
            target_definitive = st.number_input(
                "目标大保底计数",
                min_value=0,
                value=curr_target_def if curr_target_def else 0,
                step=1,
                key=f"{rule_prefix}action_target_definitive",
                help="抽到此大保底计数后停止(0=不限制)",
            )
        with col8:
            pay_override_options = [None, True, False]
            continue_pay_override = st.selectbox(
                "氪金覆盖",
                pay_override_options,
                index=(
                    pay_override_options.index(curr_pay)
                    if curr_pay in pay_override_options
                    else 0
                ),
                format_func=lambda x: {
                    None: "使用策略默认",
                    True: "强制氪金",
                    False: "强制不氪金",
                }[x],
                key=f"{rule_prefix}continue_pay_override",
                help="覆盖策略的默认氪金设置",
            )

        new_action = ContinueAction(
            min_draws_per_banner=min_draws,
            max_draws_per_banner=max_draws if max_draws > 0 else None,
            stop_on_main=stop_on_main,
            stop_on_highest_rarity=stop_on_highest_rarity,
            target_potential=target_potential if target_potential > 0 else None,
            target_pity=target_pity if target_pity > 0 else None,
            target_definitive_draw=target_definitive if target_definitive > 0 else None,
            pay_override=continue_pay_override,
        )
    elif action_type == "delegate":
        # Get all strategy names except current
        other_strategies = _get_other_strategy_names()
        if other_strategies:
            current_delegate = (
                current_action.strategy_name
                if isinstance(current_action, DelegateAction)
                else other_strategies[0]
            )
            delegate_idx = (
                other_strategies.index(current_delegate)
                if current_delegate in other_strategies
                else 0
            )
            delegate_to = st.selectbox(
                "执行策略",
                other_strategies,
                index=delegate_idx,
                key=f"{rule_prefix}action_delegate_to",
            )
            new_action = DelegateAction(strategy_name=delegate_to)
        else:
            st.warning("没有其他可用策略")
            new_action = None

    # Buttons
    with st_horizontal():
        if st.button("保存修改", key=f"{rule_prefix}save_rule"):
            if new_action is not None:
                # Build conditions
                conditions: list[StrategyCondition] = []
                if use_draw_count and (
                    draw_count_min is not None or draw_count_max is not None
                ):
                    conditions.append(
                        DrawCountCondition(
                            min_draws=draw_count_min,
                            max_draws=draw_count_max,
                        )
                    )
                if use_got_main:
                    conditions.append(GotMainCondition(value=got_main_value))
                if use_got_highest_rarity:
                    conditions.append(
                        GotHighestRarityCondition(value=got_highest_rarity_value)
                    )
                if use_got_hr_not_main:
                    conditions.append(
                        GotHighestRarityButNotMainCondition(value=got_hr_not_main_value)
                    )
                if use_resource and (
                    resource_min is not None or resource_max is not None
                ):
                    conditions.append(
                        ResourceThresholdCondition(
                            min_normal_draws=resource_min,
                            max_normal_draws=resource_max,
                            check_once=resource_check_once,
                        )
                    )
                if use_pity:
                    conditions.append(GotPityWithoutMainCondition(value=pity_value))
                if use_banner_index:
                    conditions.append(
                        BannerIndexCondition(
                            every_n=banner_every_n, start_at=banner_start_at
                        )
                    )
                if use_pity_counter and (
                    pity_counter_min is not None or pity_counter_max is not None
                ):
                    conditions.append(
                        PityCounterCondition(
                            min_pity=pity_counter_min,
                            max_pity=pity_counter_max,
                        )
                    )
                if use_definitive_counter and (
                    definitive_counter_min is not None
                    or definitive_counter_max is not None
                ):
                    conditions.append(
                        DefinitiveDrawCounterCondition(
                            min_definitive=definitive_counter_min,
                            max_definitive=definitive_counter_max,
                        )
                    )

                # Update rule
                current_strategy.rules[rule_idx] = StrategyRule(
                    conditions=conditions,
                    action=new_action,
                    priority=priority,
                )
                # Sort by priority
                current_strategy.rules.sort(key=lambda r: -r.priority)
                # Clear editing state
                strategy_key_prefix = (
                    f"strategy_{st.session_state.current_strategy_idx}_"
                )
                st.session_state[f"{strategy_key_prefix}editing_rule_{rule_idx}"] = (
                    False
                )
                update_url()
                st.rerun()
        if st.button("取消", key=f"{rule_prefix}cancel_edit"):
            strategy_key_prefix = f"strategy_{st.session_state.current_strategy_idx}_"
            st.session_state[f"{strategy_key_prefix}editing_rule_{rule_idx}"] = False
            st.rerun()


def _render_default_action_editor(current_strategy: DrawStrategy, prefix: str):
    """Render the default action editor."""
    action = current_strategy.default_action

    action_type = st.selectbox(
        "默认动作类型",
        ["stop", "continue", "delegate"],
        index=(
            0
            if isinstance(action, StopAction)
            else (1 if isinstance(action, ContinueAction) else 2)
        ),
        format_func=lambda x: {
            "stop": "停止抽卡",
            "continue": "继续抽卡",
            "delegate": "执行其他策略",
        }[x],
        key=f"{prefix}default_action_type",
    )

    new_action = None
    if action_type == "stop":
        current_pay_override = (
            action.pay_override if isinstance(action, StopAction) else None
        )
        pay_override_options = [None, True, False]
        stop_pay_override = st.selectbox(
            "氪金覆盖",
            pay_override_options,
            index=pay_override_options.index(current_pay_override),
            format_func=lambda x: {
                None: "使用策略默认",
                True: "强制氪金",
                False: "强制不氪金",
            }[x],
            key=f"{prefix}default_stop_pay_override",
            help="覆盖策略的默认氪金设置",
        )
        new_action = StopAction(pay_override=stop_pay_override)
    elif action_type == "continue":
        # Get current values if action is ContinueAction
        current_min = (
            action.min_draws_per_banner if isinstance(action, ContinueAction) else 0
        )
        current_max = (
            action.max_draws_per_banner if isinstance(action, ContinueAction) else None
        )
        current_stop = (
            action.stop_on_main if isinstance(action, ContinueAction) else False
        )
        current_stop_hr = (
            action.stop_on_highest_rarity
            if isinstance(action, ContinueAction)
            else False
        )
        current_potential = (
            action.target_potential if isinstance(action, ContinueAction) else None
        )
        current_target_pity = (
            action.target_pity if isinstance(action, ContinueAction) else None
        )
        current_target_definitive = (
            action.target_definitive_draw
            if isinstance(action, ContinueAction)
            else None
        )
        current_pay_override = (
            action.pay_override if isinstance(action, ContinueAction) else None
        )

        col1, col2 = st.columns(2)
        with col1:
            min_draws = st.number_input(
                "至少抽到",
                min_value=0,
                value=current_min,
                step=1,
                key=f"{prefix}default_action_min_draws",
                help="抽到此数量前不会停止",
            )
        with col2:
            max_draws = st.number_input(
                "最多抽到",
                min_value=0,
                value=current_max if current_max else 0,
                step=1,
                key=f"{prefix}default_action_max_draws",
                help="抽到此数量后停止(0=不限制)",
            )

        col3, col4 = st.columns(2)
        with col3:
            stop_on_main = st.checkbox(
                "抽到UP后停止",
                value=current_stop,
                key=f"{prefix}default_action_stop_on_main",
            )
        with col4:
            stop_on_highest_rarity = st.checkbox(
                "抽到最高星级后停止",
                value=current_stop_hr,
                key=f"{prefix}default_action_stop_on_highest_rarity",
            )

        col5, col6 = st.columns(2)
        with col5:
            target_potential = st.number_input(
                "目标潜能",
                min_value=0,
                max_value=6,
                value=current_potential if current_potential else 0,
                step=1,
                key=f"{prefix}default_action_target_potential",
                help="0表示不限制潜能",
            )
        with col6:
            target_pity = st.number_input(
                "目标保底计数",
                min_value=0,
                value=current_target_pity if current_target_pity else 0,
                step=1,
                key=f"{prefix}default_action_target_pity",
                help="抽到此保底计数后停止(0=不限制)",
            )

        col7, col8 = st.columns(2)
        with col7:
            target_definitive = st.number_input(
                "目标大保底计数",
                min_value=0,
                value=current_target_definitive if current_target_definitive else 0,
                step=1,
                key=f"{prefix}default_action_target_definitive",
                help="抽到此大保底计数后停止(0=不限制)",
            )
        with col8:
            pay_override_options = [None, True, False]
            continue_pay_override = st.selectbox(
                "氪金覆盖",
                pay_override_options,
                index=pay_override_options.index(current_pay_override),
                format_func=lambda x: {
                    None: "使用策略默认",
                    True: "强制氪金",
                    False: "强制不氪金",
                }[x],
                key=f"{prefix}default_continue_pay_override",
                help="覆盖策略的默认氪金设置",
            )

        new_action = ContinueAction(
            min_draws_per_banner=min_draws,
            max_draws_per_banner=max_draws if max_draws > 0 else None,
            stop_on_main=stop_on_main,
            stop_on_highest_rarity=stop_on_highest_rarity,
            target_potential=target_potential if target_potential > 0 else None,
            target_pity=target_pity if target_pity > 0 else None,
            target_definitive_draw=target_definitive if target_definitive > 0 else None,
            pay_override=continue_pay_override,
        )
    elif action_type == "delegate":
        other_strategies = _get_other_strategy_names()
        if other_strategies:
            current_delegate = (
                action.strategy_name
                if isinstance(action, DelegateAction)
                else other_strategies[0]
            )
            if current_delegate not in other_strategies:
                current_delegate = other_strategies[0]
            delegate_to = st.selectbox(
                "执行策略",
                other_strategies,
                index=(
                    other_strategies.index(current_delegate)
                    if current_delegate in other_strategies
                    else 0
                ),
                key=f"{prefix}default_action_delegate_to",
            )
            new_action = DelegateAction(strategy_name=delegate_to)
        else:
            st.warning("没有其他可用策略，无法使用委托动作")
            new_action = StopAction()

    # Update if changed
    if new_action is not None and new_action != current_strategy.default_action:
        current_strategy.default_action = new_action
        update_url()


def _render_strategy_creation():
    """Render the strategy creation and deletion section."""
    with st_horizontal():
        with st.popover("创建新策略"):
            _render_strategy_creation_dialog()

        # Delete strategy button (only if more than one strategy exists and not default)
        if (
            len(st.session_state.strategies) > 1
            and st.session_state.current_strategy_idx > 0
        ):
            if st.button("删除当前策略"):
                st.session_state.strategies.pop(st.session_state.current_strategy_idx)
                st.session_state.current_strategy_idx = min(
                    st.session_state.current_strategy_idx,
                    len(st.session_state.strategies) - 1,
                )
                update_url()
                st.rerun()

        # Apply current strategy to all banners button
        if st.button(
            "应用到所有卡池", help="将当前策略应用到所有卡池（包括自动添加的卡池）"
        ):
            current_strategy = _get_current_strategy()
            # Apply to all existing banners
            for banner in st.session_state.banners:
                st.session_state.run_banner_strategies[banner.name] = (
                    current_strategy.name
                )
                # Also update the widget key so the UI reflects the change
                config_key = f"run_config_{banner.name}"
                st.session_state[config_key] = current_strategy.name
            # Apply to auto banner strategy
            strategy_names = [s.name for s in st.session_state.strategies]
            if current_strategy.name in strategy_names:
                st.session_state.auto_banner_strategy_idx = strategy_names.index(
                    current_strategy.name
                )
            update_url()
            st.rerun()

        # Show strategy description button
        with st.popover("查看策略说明"):
            current_strategy = _get_current_strategy()
            strategy_registry = {s.name: s for s in st.session_state.strategies}
            description = current_strategy.get_description(strategy_registry)
            st.code(description, language=None)


def _render_strategy_creation_dialog():
    """Render the strategy creation dialog with full condition support."""
    prefix = "new_strategy_"

    st.subheader("创建抽卡策略")

    new_strategy_name = st.text_input(
        "策略名称",
        value="自定义策略",
        key=f"{prefix}name",
    )

    # === Behavior Section ===
    st.markdown("**抽卡行为**")
    new_always_single = st.checkbox(
        "始终单抽",
        value=False,
        key=f"{prefix}always_single",
    )
    new_single_after = st.number_input(
        "累计抽数后单抽",
        min_value=0,
        value=0,
        step=1,
        key=f"{prefix}single_after",
    )
    new_pay = st.checkbox(
        "氪金",
        value=False,
        key=f"{prefix}pay",
    )

    # === Rules Section ===
    st.markdown("**策略规则**")
    st.caption("规则按优先级从高到低执行，第一个匹配的规则生效")

    # Initialize rules list in session state if not present
    if f"{prefix}rules" not in st.session_state:
        st.session_state[f"{prefix}rules"] = []

    # Display existing rules
    rules_to_remove = []
    for idx, rule in enumerate(st.session_state[f"{prefix}rules"]):
        with st.container():
            st.markdown(f"**规则 {idx + 1}** (优先级: {rule.priority})")
            _render_rule_summary(rule)
            with st_horizontal():
                if st.button("删除", key=f"{prefix}del_rule_{idx}"):
                    rules_to_remove.append(idx)

    # Remove rules marked for deletion
    if rules_to_remove:
        for idx in sorted(rules_to_remove, reverse=True):
            st.session_state[f"{prefix}rules"].pop(idx)
        # Clear the adding_rule checkbox to reset the editor state
        if f"{prefix}adding_rule" in st.session_state:
            st.session_state[f"{prefix}adding_rule"] = False
        st.rerun()

    # Add rule toggle
    if st.checkbox("添加新规则", key=f"{prefix}adding_rule"):
        _render_creation_rule_editor(prefix)

    # === Default Action Section ===
    st.markdown("**默认行为**")
    st.caption("当没有规则匹配时执行此行为")

    default_action_type = st.selectbox(
        "默认动作类型",
        ["continue", "stop"],
        format_func=lambda x: {"stop": "停止抽卡", "continue": "继续抽卡"}[x],
        key=f"{prefix}default_action_type",
    )

    new_default_action = None
    if default_action_type == "stop":
        stop_pay_override = st.selectbox(
            "氪金覆盖",
            [None, True, False],
            format_func=lambda x: {
                None: "使用策略默认",
                True: "强制氪金",
                False: "强制不氪金",
            }[x],
            key=f"{prefix}stop_pay_override",
            help="覆盖策略的默认氪金设置",
        )
        new_default_action = StopAction(pay_override=stop_pay_override)
    else:
        col4, col5 = st.columns(2)
        with col4:
            new_stop_on_main = st.checkbox(
                "抽到UP后停止",
                value=True,
                key=f"{prefix}stop_on_main",
            )
        with col5:
            new_stop_on_highest_rarity = st.checkbox(
                "抽到最高星级后停止",
                value=False,
                key=f"{prefix}stop_on_highest_rarity",
            )

        col6, col7 = st.columns(2)
        with col6:
            new_min_draws = st.number_input(
                "至少抽到",
                min_value=0,
                value=0,
                step=1,
                key=f"{prefix}min_draws",
                help="抽到此数量前不会停止",
            )
        with col7:
            new_max_draws = st.number_input(
                "最多抽到",
                min_value=0,
                value=0,
                step=1,
                key=f"{prefix}max_draws",
                help="抽到此数量后停止(0=不限制)",
            )

        col8, col9 = st.columns(2)
        with col8:
            new_target_potential = st.number_input(
                "目标潜能",
                min_value=0,
                max_value=6,
                value=0,
                step=1,
                key=f"{prefix}target_potential",
                help="0表示不限制",
            )
        with col9:
            new_target_pity = st.number_input(
                "目标保底计数",
                min_value=0,
                value=0,
                step=1,
                key=f"{prefix}target_pity",
                help="抽到此保底计数后停止(0=不限制)",
            )

        col10, col11 = st.columns(2)
        with col10:
            new_target_definitive = st.number_input(
                "目标大保底计数",
                min_value=0,
                value=0,
                step=1,
                key=f"{prefix}target_definitive",
                help="抽到此大保底计数后停止(0=不限制)",
            )
        with col11:
            continue_pay_override = st.selectbox(
                "氪金覆盖",
                [None, True, False],
                format_func=lambda x: {
                    None: "使用策略默认",
                    True: "强制氪金",
                    False: "强制不氪金",
                }[x],
                key=f"{prefix}continue_pay_override",
                help="覆盖策略的默认氪金设置",
            )

        new_default_action = ContinueAction(
            min_draws_per_banner=new_min_draws,
            max_draws_per_banner=new_max_draws if new_max_draws > 0 else None,
            stop_on_main=new_stop_on_main,
            stop_on_highest_rarity=new_stop_on_highest_rarity,
            target_potential=new_target_potential if new_target_potential > 0 else None,
            target_pity=new_target_pity if new_target_pity > 0 else None,
            target_definitive_draw=(
                new_target_definitive if new_target_definitive > 0 else None
            ),
            pay_override=continue_pay_override,
        )

    # Check for duplicate name
    existing_names = [s.name for s in st.session_state.strategies]
    name_exists = new_strategy_name in existing_names

    if name_exists:
        st.error(f"策略名称「{new_strategy_name}」已存在，请使用其他名称")

    # Build preview strategy for description
    preview_strategy = DrawStrategy(
        name=new_strategy_name,
        behavior=DrawBehavior(
            always_single_draw=new_always_single,
            single_draw_after=new_single_after,
            pay=new_pay,
        ),
        rules=list(st.session_state.get(f"{prefix}rules", [])),
        default_action=new_default_action if new_default_action else ContinueAction(),
    )

    # Preview and Create buttons
    with st_horizontal():
        with st.popover("预览策略"):
            strategy_registry = {s.name: s for s in st.session_state.strategies}
            description = preview_strategy.get_description(strategy_registry)
            st.code(description, language=None)

        if st.button(
            "创建策略",
            key=f"{prefix}create_btn",
            disabled=name_exists,
        ):
            if new_strategy_name and not name_exists and new_default_action is not None:
                st.session_state.strategies.append(preview_strategy)
                new_idx = len(st.session_state.strategies) - 1
                st.session_state.current_strategy_idx = new_idx
                # Clear the rules list for next creation
                st.session_state[f"{prefix}rules"] = []
                update_url()
                st.rerun()


def _render_creation_rule_editor(prefix: str):
    """Render the rule editor for strategy creation dialog."""
    st.markdown("##### 新规则配置")

    rule_prefix = f"{prefix}rule_"

    # Priority
    priority = st.number_input(
        "优先级",
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        key=f"{rule_prefix}priority",
        help="数值越大优先级越高",
    )

    # === Conditions ===
    st.markdown("**条件 (全部满足时触发)**")

    # Draw count condition (draws used on current banner, excluding special draws)
    use_draw_count = st.checkbox(
        "已用抽数条件",
        key=f"{rule_prefix}use_draw_count",
        help="当前池已消耗的抽数(不含特殊抽)",
    )
    draw_count_min = None
    draw_count_max = None
    if use_draw_count:
        col1, col2 = st.columns(2)
        with col1:
            draw_count_min = st.number_input(
                "最小已用抽数",
                min_value=0,
                value=0,
                step=1,
                key=f"{rule_prefix}draw_count_min",
            )
            if draw_count_min == 0:
                draw_count_min = None
        with col2:
            draw_count_max = st.number_input(
                "最大已用抽数",
                min_value=0,
                value=0,
                step=1,
                key=f"{rule_prefix}draw_count_max",
                help="0表示不限制",
            )
            if draw_count_max == 0:
                draw_count_max = None

    # Got main condition
    use_got_main = st.checkbox("UP获取条件", key=f"{rule_prefix}use_got_main")
    got_main_value = True
    if use_got_main:
        got_main_value = st.radio(
            "UP状态",
            [True, False],
            format_func=lambda x: "已获得UP" if x else "未获得UP",
            key=f"{rule_prefix}got_main_value",
            horizontal=True,
        )

    # Got highest rarity condition
    use_got_hr = st.checkbox("最高星级获取条件", key=f"{rule_prefix}use_got_hr")
    got_hr_value = True
    if use_got_hr:
        got_hr_value = st.radio(
            "最高星级状态",
            [True, False],
            format_func=lambda x: "已获得最高星级" if x else "未获得最高星级",
            key=f"{rule_prefix}got_hr_value",
            horizontal=True,
        )

    # Got highest rarity but not main condition
    use_got_hr_not_main = st.checkbox(
        "歪最高星级条件", key=f"{rule_prefix}use_got_hr_not_main"
    )
    got_hr_not_main_value = True
    if use_got_hr_not_main:
        got_hr_not_main_value = st.radio(
            "歪最高星级状态",
            [True, False],
            format_func=lambda x: "歪了(出最高星级但非UP)" if x else "未歪最高星级",
            key=f"{rule_prefix}got_hr_not_main_value",
            horizontal=True,
        )

    # Resource threshold condition (available draws that carry over)
    use_resource = st.checkbox(
        "可用抽数条件",
        key=f"{rule_prefix}use_resource",
        help="可跨池继承的剩余抽数",
    )
    resource_min = None
    resource_max = None
    resource_check_once = False
    if use_resource:
        col1, col2 = st.columns(2)
        with col1:
            resource_min = st.number_input(
                "最小可用抽数",
                min_value=0,
                value=0,
                step=1,
                key=f"{rule_prefix}resource_min",
            )
            if resource_min == 0:
                resource_min = None
        with col2:
            resource_max = st.number_input(
                "最大可用抽数",
                min_value=0,
                value=0,
                step=1,
                key=f"{rule_prefix}resource_max",
                help="0表示不限制",
            )
            if resource_max == 0:
                resource_max = None
        resource_check_once = st.checkbox(
            "仅入池时检查",
            key=f"{rule_prefix}resource_check_once",
            help="勾选后仅在进入池子时检查一次，之后不再检查",
        )

    # Pity without main condition
    use_pity = st.checkbox("歪了条件", key=f"{rule_prefix}use_pity")
    pity_value = True
    if use_pity:
        pity_value = st.radio(
            "歪了状态",
            [True, False],
            format_func=lambda x: "歪了(保底未出UP)" if x else "未歪",
            key=f"{rule_prefix}pity_value",
            horizontal=True,
        )

    # Banner index condition
    use_banner_index = st.checkbox(
        "池子序号条件",
        key=f"{rule_prefix}use_banner_index",
        help="仅在特定序号的池子生效(从第1个池子开始计数)",
    )
    banner_every_n = 0
    banner_start_at = 1
    if use_banner_index:
        col1, col2 = st.columns(2)
        with col1:
            banner_every_n = st.number_input(
                "每N个池子",
                min_value=0,
                value=2,
                step=1,
                key=f"{rule_prefix}banner_every_n",
                help="0=仅指定池子, 1=每个池子, 2=每2个池子中选1个",
            )
        with col2:
            # When every_n=0, allow any start_at (specific banner)
            # When every_n=1, start_at doesn't matter (every banner)
            # When every_n>=2, limit start_at to every_n
            max_start = (
                999
                if banner_every_n == 0
                else (banner_every_n if banner_every_n > 1 else 1)
            )
            banner_start_at = st.number_input(
                "第几个池子" if banner_every_n == 0 else "从第几个池子开始",
                min_value=1,
                max_value=max(1, max_start),
                value=1,
                step=1,
                key=f"{rule_prefix}banner_start_at",
                help=(
                    "指定池子序号"
                    if banner_every_n == 0
                    else "1=第1,3,5...个, 2=第2,4,6...个(每2池时)"
                ),
            )

    # Pity counter condition
    use_pity_counter = st.checkbox(
        "保底计数条件",
        key=f"{rule_prefix}use_pity_counter",
        help="距离上次出最高星级的抽数",
    )
    pity_counter_min = None
    pity_counter_max = None
    if use_pity_counter:
        col1, col2 = st.columns(2)
        with col1:
            pity_counter_min = st.number_input(
                "最小保底计数",
                min_value=0,
                value=0,
                step=1,
                key=f"{rule_prefix}pity_counter_min",
            )
            if pity_counter_min == 0:
                pity_counter_min = None
        with col2:
            pity_counter_max = st.number_input(
                "最大保底计数",
                min_value=0,
                value=0,
                step=1,
                key=f"{rule_prefix}pity_counter_max",
                help="0表示不限制",
            )
            if pity_counter_max == 0:
                pity_counter_max = None

    # Definitive draw counter condition
    use_definitive_counter = st.checkbox(
        "大保底计数条件",
        key=f"{rule_prefix}use_definitive_counter",
        help="距离大保底(必得UP)的抽数",
    )
    definitive_counter_min = None
    definitive_counter_max = None
    if use_definitive_counter:
        col1, col2 = st.columns(2)
        with col1:
            definitive_counter_min = st.number_input(
                "最小大保底计数",
                min_value=0,
                value=0,
                step=1,
                key=f"{rule_prefix}definitive_counter_min",
            )
            if definitive_counter_min == 0:
                definitive_counter_min = None
        with col2:
            definitive_counter_max = st.number_input(
                "最大大保底计数",
                min_value=0,
                value=0,
                step=1,
                key=f"{rule_prefix}definitive_counter_max",
                help="0表示不限制",
            )
            if definitive_counter_max == 0:
                definitive_counter_max = None

    # === Action ===
    st.markdown("**动作**")
    action_type = st.selectbox(
        "动作类型",
        ["stop", "continue", "delegate"],
        format_func=lambda x: {
            "stop": "停止抽卡",
            "continue": "继续抽卡",
            "delegate": "执行其他策略",
        }[x],
        key=f"{rule_prefix}action_type",
    )

    action = None
    if action_type == "stop":
        stop_pay_override = st.selectbox(
            "氪金覆盖",
            [None, True, False],
            format_func=lambda x: {
                None: "使用策略默认",
                True: "强制氪金",
                False: "强制不氪金",
            }[x],
            key=f"{rule_prefix}stop_pay_override",
            help="覆盖策略的默认氪金设置",
        )
        action = StopAction(pay_override=stop_pay_override)
    elif action_type == "continue":
        col1, col2 = st.columns(2)
        with col1:
            min_draws = st.number_input(
                "至少抽到",
                min_value=0,
                value=0,
                step=1,
                key=f"{rule_prefix}action_min_draws",
                help="抽到此数量前不会停止",
            )
        with col2:
            max_draws = st.number_input(
                "最多抽到",
                min_value=0,
                value=0,
                step=1,
                key=f"{rule_prefix}action_max_draws",
                help="抽到此数量后停止(0=不限制)",
            )

        col3, col4 = st.columns(2)
        with col3:
            stop_on_main = st.checkbox(
                "抽到UP后停止",
                key=f"{rule_prefix}action_stop_on_main",
            )
        with col4:
            stop_on_hr = st.checkbox(
                "抽到最高星级后停止",
                key=f"{rule_prefix}action_stop_on_hr",
            )

        col5, col6 = st.columns(2)
        with col5:
            target_pot = st.number_input(
                "目标潜能",
                min_value=0,
                max_value=6,
                value=0,
                step=1,
                key=f"{rule_prefix}action_target_potential",
                help="0表示不限制潜能",
            )
        with col6:
            target_pity = st.number_input(
                "目标保底计数",
                min_value=0,
                value=0,
                step=1,
                key=f"{rule_prefix}action_target_pity",
                help="抽到此保底计数后停止(0=不限制)",
            )

        col7, col8 = st.columns(2)
        with col7:
            target_definitive = st.number_input(
                "目标大保底计数",
                min_value=0,
                value=0,
                step=1,
                key=f"{rule_prefix}action_target_definitive",
                help="抽到此大保底计数后停止(0=不限制)",
            )
        with col8:
            continue_pay_override = st.selectbox(
                "氪金覆盖",
                [None, True, False],
                format_func=lambda x: {
                    None: "使用策略默认",
                    True: "强制氪金",
                    False: "强制不氪金",
                }[x],
                key=f"{rule_prefix}continue_pay_override",
                help="覆盖策略的默认氪金设置",
            )

        action = ContinueAction(
            min_draws_per_banner=min_draws,
            max_draws_per_banner=max_draws if max_draws > 0 else None,
            stop_on_main=stop_on_main,
            stop_on_highest_rarity=stop_on_hr,
            target_potential=target_pot if target_pot > 0 else None,
            target_pity=target_pity if target_pity > 0 else None,
            target_definitive_draw=target_definitive if target_definitive > 0 else None,
            pay_override=continue_pay_override,
        )
    elif action_type == "delegate":
        # Get all strategy names except the one being created
        other_strategies = [s.name for s in st.session_state.strategies]
        if other_strategies:
            delegate_to = st.selectbox(
                "执行策略",
                other_strategies,
                key=f"{rule_prefix}action_delegate_to",
            )
            action = DelegateAction(strategy_name=delegate_to)  # type:ignore
        else:
            st.warning("没有其他可用策略")
            action = None

    # Add rule button
    if st.button("添加此规则", key=f"{rule_prefix}add_btn"):
        if action is not None:
            # Build conditions
            conditions: list[StrategyCondition] = []
            if use_draw_count and (
                draw_count_min is not None or draw_count_max is not None
            ):
                conditions.append(
                    DrawCountCondition(
                        min_draws=draw_count_min,
                        max_draws=draw_count_max,
                    )
                )
            if use_got_main:
                conditions.append(GotMainCondition(value=got_main_value))
            if use_got_hr:
                conditions.append(GotHighestRarityCondition(value=got_hr_value))
            if use_got_hr_not_main:
                conditions.append(
                    GotHighestRarityButNotMainCondition(value=got_hr_not_main_value)
                )
            if use_resource and (resource_min is not None or resource_max is not None):
                conditions.append(
                    ResourceThresholdCondition(
                        min_normal_draws=resource_min,
                        max_normal_draws=resource_max,
                        check_once=resource_check_once,
                    )
                )
            if use_pity:
                conditions.append(GotPityWithoutMainCondition(value=pity_value))
            if use_banner_index:
                conditions.append(
                    BannerIndexCondition(
                        every_n=banner_every_n, start_at=banner_start_at
                    )
                )
            if use_pity_counter and (
                pity_counter_min is not None or pity_counter_max is not None
            ):
                conditions.append(
                    PityCounterCondition(
                        min_pity=pity_counter_min,
                        max_pity=pity_counter_max,
                    )
                )
            if use_definitive_counter and (
                definitive_counter_min is not None or definitive_counter_max is not None
            ):
                conditions.append(
                    DefinitiveDrawCounterCondition(
                        min_definitive=definitive_counter_min,
                        max_definitive=definitive_counter_max,
                    )
                )

            # Create rule
            new_rule = StrategyRule(
                conditions=conditions,
                action=action,
                priority=priority,
            )
            st.session_state[f"{prefix}rules"].append(new_rule)
            # Sort by priority
            st.session_state[f"{prefix}rules"].sort(key=lambda r: -r.priority)
            st.rerun()
