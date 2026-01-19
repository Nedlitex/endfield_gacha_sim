"""Strategy selector and editor component."""

import streamlit as st

from gacha import DrawStrategy
from ui.state import update_url


def _get_current_strategy() -> DrawStrategy:
    """Get the currently selected strategy."""
    return st.session_state.strategies[st.session_state.current_strategy_idx]


def _on_strategy_change():
    """Callback when strategy settings change."""
    strategy = _get_current_strategy()
    strategy_key_prefix = f"strategy_{st.session_state.current_strategy_idx}_"
    strategy.always_single_draw = st.session_state[
        f"{strategy_key_prefix}always_single_draw"
    ]
    strategy.single_draw_after = st.session_state[
        f"{strategy_key_prefix}single_draw_after"
    ]
    strategy.skip_banner_threshold = st.session_state[
        f"{strategy_key_prefix}skip_banner_threshold"
    ]
    strategy.min_draws_per_banner = st.session_state[
        f"{strategy_key_prefix}min_draws_per_banner"
    ]
    strategy.max_draws_per_banner = st.session_state[
        f"{strategy_key_prefix}max_draws_per_banner"
    ]
    strategy.stop_on_main = st.session_state[f"{strategy_key_prefix}stop_on_main"]
    strategy.pay = st.session_state[f"{strategy_key_prefix}pay"]
    update_url()


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
    with st.expander(f"策略配置: {current_strategy.name}", expanded=True):
        if is_default_strategy:
            # Show read-only view for default strategy
            st.info(
                "💡 这是默认策略，不可编辑。请在下方输入新策略名称并点击「创建策略」来创建自定义策略。"
            )
            st.markdown("**策略说明:** 氪金抽到UP（抽数不足时额外获得抽数以满足规则）")
        else:
            _render_strategy_editor(current_strategy)

    # New strategy creation (after the expander)
    _render_strategy_creation()


def _render_strategy_editor(current_strategy: DrawStrategy):
    """Render the strategy editor for custom strategies."""
    # Use strategy index in keys to avoid cross-strategy contamination
    strategy_key_prefix = f"strategy_{st.session_state.current_strategy_idx}_"

    # Number inputs row 1
    col3, col4 = st.columns(2)
    with col3:
        st.number_input(
            "每池最少抽数",
            min_value=0,
            value=current_strategy.min_draws_per_banner,
            step=1,
            key=f"{strategy_key_prefix}min_draws_per_banner",
            on_change=_on_strategy_change,
        )
    with col4:
        st.number_input(
            "每池最多抽数",
            min_value=0,
            value=current_strategy.max_draws_per_banner,
            step=1,
            key=f"{strategy_key_prefix}max_draws_per_banner",
            on_change=_on_strategy_change,
            help="每个卡池最多抽取的次数(0表示无限制)",
        )

    # Number inputs row 2
    col5, col6 = st.columns(2)
    with col5:
        st.number_input(
            "跳池阈值",
            min_value=0,
            value=current_strategy.skip_banner_threshold,
            step=1,
            key=f"{strategy_key_prefix}skip_banner_threshold",
            on_change=_on_strategy_change,
            help="剩余抽数低于此值时跳过当前卡池",
        )
    with col6:
        st.number_input(
            "累计抽数后单抽",
            min_value=0,
            value=current_strategy.single_draw_after,
            step=1,
            key=f"{strategy_key_prefix}single_draw_after",
            on_change=_on_strategy_change,
            help="累计抽数达到此值后开始单抽(特殊10连除外)",
        )

    # Checkboxes row
    col7, col8, col9 = st.columns(3)
    with col7:
        st.checkbox(
            "抽到UP后停止",
            value=current_strategy.stop_on_main,
            key=f"{strategy_key_prefix}stop_on_main",
            on_change=_on_strategy_change,
            help="获得UP干员后立即停止抽取当前卡池",
        )
    with col8:
        st.checkbox(
            "始终单抽",
            value=current_strategy.always_single_draw,
            key=f"{strategy_key_prefix}always_single_draw",
            on_change=_on_strategy_change,
            help="始终单抽(特殊10连除外)",
        )
    with col9:
        st.checkbox(
            "氪金",
            value=current_strategy.pay,
            key=f"{strategy_key_prefix}pay",
            on_change=_on_strategy_change,
            help="抽数不足时额外获得抽数以满足规则",
        )

    # min_draws_after_main rules
    _render_after_main_rules(current_strategy)

    # min_draws_after_pity rules
    _render_after_pity_rules(current_strategy)

    # Generate strategy summary button
    _render_strategy_summary(current_strategy)


def _render_after_main_rules(current_strategy: DrawStrategy):
    """Render the after-main rules section."""
    st.subheader("获得UP后规则")
    st.caption("当前抽数 >= 阈值时，获得UP后继续抽至目标抽数")

    # Display existing rules
    for idx, (threshold, target) in enumerate(current_strategy.min_draws_after_main):
        col1, col2 = st.columns([5, 1])
        with col1:
            st.text(f"获得UP后若当前抽数>={threshold}则继续抽至{target}")
        with col2:
            if st.button("删除", key=f"delete_rule_{idx}"):
                current_strategy.min_draws_after_main.pop(idx)
                update_url()
                st.rerun()

    # Add new rule
    col1, col2 = st.columns(2)
    with col1:
        new_threshold = st.number_input(
            "阈值", min_value=0, value=0, step=1, key="new_rule_threshold"
        )
    with col2:
        new_target = st.number_input(
            "目标", min_value=0, value=0, step=1, key="new_rule_target"
        )
    if st.button("添加规则"):
        if new_threshold > 0 and new_target > 0:
            current_strategy.min_draws_after_main.append((new_threshold, new_target))
            update_url()
            st.rerun()


def _render_after_pity_rules(current_strategy: DrawStrategy):
    """Render the after-pity rules section."""
    st.subheader("小保底歪了后规则")
    st.caption("当前抽数 >= 阈值时，歪了(触发小保底但未获得UP)后继续抽至目标抽数")

    # Display existing rules
    for idx, (threshold, target) in enumerate(current_strategy.min_draws_after_pity):
        col1, col2 = st.columns([5, 1])
        with col1:
            st.text(f"歪了后若当前抽数>={threshold}则继续抽至{target}")
        with col2:
            if st.button("删除", key=f"delete_pity_rule_{idx}"):
                current_strategy.min_draws_after_pity.pop(idx)
                update_url()
                st.rerun()

    # Add new rule
    col1, col2 = st.columns(2)
    with col1:
        new_pity_threshold = st.number_input(
            "阈值", min_value=0, value=0, step=1, key="new_pity_rule_threshold"
        )
    with col2:
        new_pity_target = st.number_input(
            "目标", min_value=0, value=0, step=1, key="new_pity_rule_target"
        )
    if st.button(label="添加规则", key="pity"):
        if new_pity_threshold > 0 and new_pity_target > 0:
            current_strategy.min_draws_after_pity.append(
                (new_pity_threshold, new_pity_target)
            )
            update_url()
            st.rerun()


def _render_strategy_summary(current_strategy: DrawStrategy):
    """Render the strategy summary generator."""
    if st.button("生成策略说明"):
        paragraphs = []
        paragraphs.append(f"【{current_strategy.name}】")
        config = st.session_state.config
        resource_desc = f"玩家初始拥有{config.initial_draws}抽"
        if config.draws_gain_per_banner > 0:
            resource_desc += f"，每期卡池额外获得{config.draws_gain_per_banner}抽"
        if config.draws_gain_this_banner > 0:
            resource_desc += (
                f"，每期卡池额外获得{config.draws_gain_this_banner}限定抽(仅限当期使用)"
            )
        resource_desc += "。"
        paragraphs.append(resource_desc)

        if current_strategy.min_draws_per_banner > 0:
            paragraphs.append(
                f"每个卡池至少抽{current_strategy.min_draws_per_banner}抽。"
            )

        if current_strategy.max_draws_per_banner > 0:
            paragraphs.append(
                f"每个卡池最多抽{current_strategy.max_draws_per_banner}抽。"
            )

        if current_strategy.stop_on_main:
            paragraphs.append("获得UP干员后立即停止抽取当前卡池。")

        if current_strategy.skip_banner_threshold > 0:
            paragraphs.append(
                f"当剩余抽数低于{current_strategy.skip_banner_threshold}时，"
                "跳过当前卡池不再抽取。"
            )

        if current_strategy.always_single_draw:
            paragraphs.append("抽卡时始终单抽，特殊10连除外。")
        elif current_strategy.single_draw_after > 0:
            paragraphs.append(
                f"当累计抽数达到{current_strategy.single_draw_after}后，"
                "改为单抽以节省资源，特殊10连除外。"
            )

        if current_strategy.min_draws_after_main:
            rules_desc = []
            for threshold, target in current_strategy.min_draws_after_main:
                rules_desc.append(f"若当前抽数>={threshold}则继续抽至{target}抽")
            paragraphs.append(f"获得UP干员后，{'；'.join(rules_desc)}。")

        if current_strategy.min_draws_after_pity:
            rules_desc = []
            for threshold, target in current_strategy.min_draws_after_pity:
                rules_desc.append(f"若当前抽数>={threshold}则继续抽至{target}抽")
            paragraphs.append(
                f"歪了(触发小保底但未获得UP)后，{'；'.join(rules_desc)}。"
            )

        if current_strategy.pay:
            paragraphs.append(":red[**抽数不足时氪金补充抽数以满足规则。**]")

        st.info("\n\n".join(paragraphs))


def _render_strategy_creation():
    """Render the strategy creation and deletion section."""
    col_create, col_delete = st.columns(2)
    with col_create:
        with st.popover("创建新策略", use_container_width=True):
            st.subheader("创建抽卡策略")

            new_strategy_name = st.text_input(
                "策略名称",
                value="自定义策略",
                key="new_strategy_name",
            )

            st.markdown("**抽数限制**")
            col1, col2 = st.columns(2)
            with col1:
                new_min_draws = st.number_input(
                    "每池最少抽数",
                    min_value=0,
                    value=0,
                    step=1,
                    key="new_strategy_min_draws",
                )
            with col2:
                new_max_draws = st.number_input(
                    "每池最多抽数",
                    min_value=0,
                    value=0,
                    step=1,
                    key="new_strategy_max_draws",
                    help="0表示无限制",
                )

            st.markdown("**抽卡行为**")
            col3, col4 = st.columns(2)
            with col3:
                new_skip_threshold = st.number_input(
                    "跳池阈值",
                    min_value=0,
                    value=0,
                    step=1,
                    key="new_strategy_skip_threshold",
                    help="剩余抽数低于此值时跳过当前卡池",
                )
            with col4:
                new_single_after = st.number_input(
                    "累计抽数后单抽",
                    min_value=0,
                    value=0,
                    step=1,
                    key="new_strategy_single_after",
                    help="累计抽数达到此值后开始单抽",
                )

            col5, col6, col7 = st.columns(3)
            with col5:
                new_stop_on_main = st.checkbox(
                    "抽到UP后停止",
                    value=True,
                    key="new_strategy_stop_on_main",
                )
            with col6:
                new_always_single = st.checkbox(
                    "始终单抽",
                    value=False,
                    key="new_strategy_always_single",
                )
            with col7:
                new_pay = st.checkbox(
                    "氪金",
                    value=False,
                    key="new_strategy_pay",
                    help="抽数不足时额外获得抽数",
                )

            if st.button("创建策略", key="create_strategy_btn"):
                if new_strategy_name:
                    new_strategy = DrawStrategy(
                        name=new_strategy_name,
                        min_draws_per_banner=new_min_draws,
                        max_draws_per_banner=new_max_draws,
                        skip_banner_threshold=new_skip_threshold,
                        single_draw_after=new_single_after,
                        stop_on_main=new_stop_on_main,
                        always_single_draw=new_always_single,
                        pay=new_pay,
                    )
                    st.session_state.strategies.append(new_strategy)
                    new_idx = len(st.session_state.strategies) - 1
                    st.session_state.current_strategy_idx = new_idx
                    update_url()
                    st.rerun()

    with col_delete:
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
