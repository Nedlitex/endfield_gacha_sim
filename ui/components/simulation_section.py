"""Simulation control and results display component."""

import asyncio

import pandas as pd
import streamlit as st

from banner import Banner
from gacha import Config, Run
from strategy import ContinueAction, DrawBehavior, DrawStrategy
from ui.components.strategy_section import render_strategy_section
from ui.constants import RARITY_COLORS
from ui.defaults import create_default_operators
from ui.state import update_url


def _on_run_banner_change(banner_name: str):
    """Callback when banner enabled checkbox changes."""
    st.session_state.run_banner_enabled[banner_name] = st.session_state[
        f"run_banner_{banner_name}"
    ]
    update_url()


def _on_run_strategy_change(banner_name: str):
    """Callback when banner strategy selection changes."""
    st.session_state.run_banner_strategies[banner_name] = st.session_state[
        f"run_config_{banner_name}"
    ]
    update_url()


def _on_auto_banner_config_change():
    """Callback when auto banner configuration changes."""
    update_url()


def _on_num_experiments_change():
    """Callback when number of experiments changes."""
    update_url()


def _on_config_change():
    """Callback when config values change."""
    st.session_state.config.initial_draws = st.session_state.config_initial_draws
    st.session_state.config.draws_gain_per_banner = (
        st.session_state.config_draws_per_banner
    )
    st.session_state.config.draws_gain_this_banner = (
        st.session_state.config_draws_this_banner
    )
    update_url()


def _on_quick_sim_config_change():
    """Callback when quick simulation config changes."""
    update_url()


def _render_quick_simulation_section():
    """Render the quick simulation section (collapsed by default)."""
    with st.expander("快速模拟", expanded=False):
        st.caption("无初始抽数，氪金抽到UP为止，计算每个卡池的平均氪金抽数")

        if not st.session_state.banners:
            st.info("请先创建卡池")
            return

        # Initialize quick sim banner list with all banners by default
        if "quick_sim_banner_list" not in st.session_state:
            st.session_state.quick_sim_banner_list = [
                b.name for b in st.session_state.banners
            ]

        # Banner selection - add banners to list
        st.markdown("**添加卡池**")
        col1, col2 = st.columns([3, 1])
        with col1:
            banner_names = [b.name for b in st.session_state.banners]
            selected_banner_idx = st.selectbox(
                "选择卡池",
                range(len(banner_names)),
                format_func=lambda x: banner_names[x],
                key="quick_sim_banner_select",
                label_visibility="collapsed",
            )
        with col2:
            if st.button("添加", key="quick_sim_add_banner"):
                banner_name = banner_names[selected_banner_idx]
                if banner_name not in st.session_state.quick_sim_banner_list:
                    st.session_state.quick_sim_banner_list.append(banner_name)
                    st.rerun()

        # Show selected banners
        if st.session_state.quick_sim_banner_list:
            st.markdown("**已选卡池序列:**")
            display_text = " → ".join(st.session_state.quick_sim_banner_list)
            st.markdown(
                f"<span style='color: #ff4b4b; font-size: 1.2em; font-weight: bold;'>{display_text}</span>",
                unsafe_allow_html=True,
            )

            if st.button("清空卡池列表", key="quick_sim_clear_banners"):
                st.session_state.quick_sim_banner_list = []
                st.session_state.quick_sim_auto_banner_count = 0
                st.rerun()
        else:
            st.caption("尚未添加卡池")

        # Auto banner configuration
        st.markdown("**自动添加卡池**")
        st.caption("在已选卡池之后，使用模板自动生成更多卡池继续模拟（UP干员随机生成）")
        col1, col2 = st.columns([1, 2])
        with col1:
            auto_count = st.number_input(
                "自动添加数量",
                min_value=0,
                max_value=100,
                value=st.session_state.get("quick_sim_auto_banner_count", 0),
                step=1,
                key="quick_sim_auto_banner_count",
                on_change=_on_quick_sim_config_change,
                help="在已选卡池后自动生成的卡池数量",
            )
        with col2:
            template_names = [t.name for t in st.session_state.banner_templates]
            current_template_idx = st.session_state.get(
                "quick_sim_auto_template_idx", 0
            )
            if current_template_idx >= len(template_names):
                current_template_idx = 0
            st.selectbox(
                "卡池模板",
                range(len(template_names)),
                index=current_template_idx,
                format_func=lambda x: template_names[x],
                key="quick_sim_auto_template_idx",
                on_change=_on_quick_sim_config_change,
                help="自动生成卡池使用的模板",
            )

        st.markdown("---")

        # Configuration row
        num_experiments = st.number_input(
            "模拟次数",
            min_value=1,
            max_value=100000,
            value=st.session_state.get("quick_sim_experiments", 1000),
            step=100,
            key="quick_sim_experiments",
            on_change=_on_quick_sim_config_change,
        )

        # Options row - only single draw option, always pay mode
        single_draw = st.checkbox(
            "单抽模式",
            value=st.session_state.get("quick_sim_single_draw", False),
            key="quick_sim_single_draw",
            on_change=_on_quick_sim_config_change,
            help="勾选后每次抽1次，否则每次抽10次",
        )

        # Run button
        has_banners = st.session_state.quick_sim_banner_list or auto_count > 0
        if has_banners:
            if st.button("运行快速模拟", type="primary", key="quick_sim_run"):
                auto_template_idx = st.session_state.get(
                    "quick_sim_auto_template_idx", 0
                )
                auto_template = None
                if auto_count > 0 and auto_template_idx < len(
                    st.session_state.banner_templates
                ):
                    auto_template = st.session_state.banner_templates[auto_template_idx]
                _execute_quick_simulation(
                    st.session_state.quick_sim_banner_list,
                    num_experiments,
                    single_draw,
                    auto_banner_count=auto_count,
                    auto_banner_template=auto_template,
                )
        else:
            st.button(
                "运行快速模拟", type="primary", key="quick_sim_run", disabled=True
            )

        # Display quick simulation results
        if st.session_state.quick_sim_results:
            _render_quick_simulation_results()


def _execute_quick_simulation(
    banner_name_list: list[str],
    num_experiments: int,
    single_draw: bool,
    auto_banner_count: int = 0,
    auto_banner_template=None,
):
    """Execute quick simulation with multiple banners."""
    # Create a simple "always continue until main" strategy with pay mode
    quick_sim_strategy = DrawStrategy(
        name="快速模拟策略",
        behavior=DrawBehavior(
            always_single_draw=single_draw,
            single_draw_after=0,
            pay=True,  # Always pay mode for quick simulation
        ),
        rules=[],  # No special rules, use default action
        default_action=ContinueAction(
            stop_on_main=True
        ),  # Stop after getting main operator
    )

    # Build banner list from names
    banner_name_to_banner = {b.name: b for b in st.session_state.banners}
    banners_for_run = []
    banner_strategies = {}
    main_operators = []

    for idx, name in enumerate(banner_name_list):
        if name in banner_name_to_banner:
            banner_copy = banner_name_to_banner[name].model_copy(deep=True)
            # Make unique name for each banner
            unique_name = f"{name}_{idx}"
            banner_copy.name = unique_name
            banners_for_run.append(banner_copy)
            banner_strategies[unique_name] = quick_sim_strategy
            if (
                banner_copy.main_operator
                and banner_copy.main_operator.name not in main_operators
            ):
                main_operators.append(banner_copy.main_operator.name)

    if not banners_for_run and auto_banner_count == 0:
        st.error("没有有效的卡池")
        return

    # Quick sim uses default config - no initial draws, pay for everything
    config = Config()

    # Prepare auto banner configuration
    auto_banner_template_copy = None
    auto_banner_default_operators = []
    if auto_banner_count > 0 and auto_banner_template:
        auto_banner_template_copy = auto_banner_template.model_copy(deep=True)
        auto_banner_default_operators = create_default_operators()

    run = Run(
        config=config,
        banners=banners_for_run,
        banner_strategies=banner_strategies,
        repeat=num_experiments,
        auto_banner_template=auto_banner_template_copy,
        auto_banner_strategy=quick_sim_strategy if auto_banner_count > 0 else None,
        auto_banner_count=auto_banner_count,
        auto_banner_default_operators=auto_banner_default_operators,
    )

    # Run with progress
    progress_bar = st.progress(0, text=f"正在运行快速模拟... 0/{num_experiments}")

    def update_progress(current: int, total: int):
        progress = current / total if total > 0 else 0
        progress_bar.progress(progress, text=f"正在运行快速模拟... {current}/{total}")

    async def run_async():
        return await run.run_simulation_async(
            yield_every=max(1, num_experiments // 100),
            progress_callback=update_progress,
        )

    result_player = asyncio.run(run_async())
    progress_bar.empty()

    # Store results
    total_banners = len(banners_for_run) + auto_banner_count
    st.session_state.quick_sim_results = {
        "player": result_player,
        "paid_draws": run.paid_draws,
        "total_draws": run.total_draws,
        "num_experiments": num_experiments,
        "num_banners": total_banners,
        "banner_names": banner_name_list,
        "auto_banner_count": auto_banner_count,
        "main_operators": main_operators,
    }
    update_url()
    st.rerun()


def _render_quick_simulation_results():
    """Render quick simulation results."""
    results = st.session_state.quick_sim_results

    # Build banner display string
    banner_names = results["banner_names"]
    auto_count = results.get("auto_banner_count", 0)
    parts = []
    if banner_names:
        parts.append(" → ".join(banner_names))
    if auto_count > 0:
        parts.append(f"自动池×{auto_count}")
    banner_display = " + ".join(parts) if parts else "无卡池"

    st.markdown("---")
    _render_simulation_results_shared(
        results=results,
        banner_display=banner_display,
        is_quick_sim=True,
        clear_button_key="clear_quick_sim",
        clear_state_key="quick_sim_results",
    )


def _render_simulation_results_shared(
    results: dict,
    banner_display: str,
    is_quick_sim: bool,
    clear_button_key: str,
    clear_state_key: str,
):
    """Shared function to render simulation results for both quick and advanced sim.

    Args:
        results: Dict containing player, paid_draws, total_draws, num_experiments, etc.
        banner_display: String describing the banners that were simulated
        is_quick_sim: Whether this is quick simulation (affects labels)
        clear_button_key: Unique key for the clear button
        clear_state_key: Session state key to clear when button is clicked
    """
    player = results["player"]
    num_exp = results["num_experiments"]

    # Get banner count
    if is_quick_sim:
        num_banners = results.get("num_banners", 1)
    else:
        num_banners = results.get("total_banner_count", len(results.get("banners", [])))

    # Calculate averages
    avg_total_per_run = results["total_draws"] / num_exp if num_exp > 0 else 0
    avg_total_per_banner = avg_total_per_run / num_banners if num_banners > 0 else 0
    avg_paid_per_run = results["paid_draws"] / num_exp if num_exp > 0 else 0
    avg_paid_per_banner = avg_paid_per_run / num_banners if num_banners > 0 else 0

    # Calculate average unique UPs obtained per run
    main_ops = results.get("main_operators", [])
    total_up_acquisitions = 0
    for main_name in main_ops:
        if 6 in player.operators and main_name in player.operators[6]:
            op = player.operators[6][main_name]
            total_up_acquisitions += op.first_draw_count
    avg_unique_ups = total_up_acquisitions / num_exp if num_exp > 0 else 0

    # Calculate main operator first draw expectations
    main_op_stats = []
    for main_name in main_ops:
        if 6 in player.operators and main_name in player.operators[6]:
            op = player.operators[6][main_name]
            expected_first = (
                (op.first_draw_total / op.first_draw_count)
                if op.first_draw_count > 0
                else 0
            )
            acquisition_rate = (
                (op.first_draw_count / num_exp * 100) if num_exp > 0 else 0
            )
            main_op_stats.append((main_name, expected_first, acquisition_rate))

    # Display summary stats prominently
    st.markdown(f"**模拟次数:** {num_exp} | **参与卡池:** {banner_display}")

    # Different metric display based on simulation type
    if is_quick_sim:
        # Quick sim: all draws are paid, so just show paid draws
        col1, col2 = st.columns(2)
        with col1:
            st.metric("每池平均氪金抽数", f"{avg_paid_per_banner:.1f}")
        with col2:
            if len(main_ops) > 0:
                st.metric("平均获得UP数", f"{avg_unique_ups:.2f} / {len(main_ops)}")
    else:
        # Advanced sim: show both total and paid draws
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "每池平均总抽数",
                f"{avg_total_per_banner:.1f}",
                help="包含初始抽数和每期获得抽数",
            )
        with col2:
            st.metric(
                "每池平均氪金抽数",
                f"{avg_paid_per_banner:.1f}",
                help="额外氪金购买的抽数",
            )
        with col3:
            if len(main_ops) > 0:
                st.metric("平均获得UP数", f"{avg_unique_ups:.2f} / {len(main_ops)}")

    # Display main operator stats prominently
    if main_op_stats:
        st.markdown("#### 🎯 UP干员首抽期望(氪金抽)")
        st.info(
            "此数值为**获取到UP干员时**的平均氪金抽数，请结合下方获取率一起参考。"
            "若获取率较低，说明大部分模拟中未抽到该干员。"
        )
        cols = st.columns(len(main_op_stats))
        for i, (name, expected, acq_rate) in enumerate(main_op_stats):
            with cols[i]:
                if acq_rate >= 99.5:
                    # High acquisition rate, just show expected draws
                    st.metric(name, f"{expected:.1f} 抽")
                elif acq_rate > 0:
                    # Show expected draws with acquisition rate
                    st.metric(
                        name,
                        f"{expected:.1f} 抽",
                        delta=f"获取率 {acq_rate:.1f}%",
                        delta_color="off",
                    )
                else:
                    # Never obtained
                    st.metric(name, "未获取")

    # Display operators by rarity
    _render_operator_tables(player, num_exp)

    if st.button("清除结果", key=clear_button_key):
        st.session_state[clear_state_key] = None
        st.rerun()


def _render_resource_config():
    """Render the resource configuration section."""
    st.subheader("资源配置")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input(
                "初始抽数",
                min_value=0,
                value=st.session_state.config.initial_draws,
                step=1,
                key="config_initial_draws",
                on_change=_on_config_change,
            )
        with col2:
            st.number_input(
                "每期卡池获得抽数",
                min_value=0,
                value=st.session_state.config.draws_gain_per_banner,
                step=1,
                key="config_draws_per_banner",
                on_change=_on_config_change,
                help="每期卡池获得的抽数，可以结转到下一期",
            )
        with col3:
            st.number_input(
                "每期限定抽数",
                min_value=0,
                value=st.session_state.config.draws_gain_this_banner,
                step=1,
                key="config_draws_this_banner",
                on_change=_on_config_change,
                help="每期卡池获得的限定抽数，仅限当期使用，不结转",
            )


def render_simulation_section():
    """Render the simulation controls and results."""
    st.header("运行模拟")

    # Initialize run state
    if "run_banner_enabled" not in st.session_state:
        st.session_state.run_banner_enabled = {}
    if "run_banner_strategies" not in st.session_state:
        st.session_state.run_banner_strategies = {}
    if "run_results" not in st.session_state:
        st.session_state.run_results = None
    if "quick_sim_results" not in st.session_state:
        st.session_state.quick_sim_results = None

    # Quick simulation section (collapsed by default)
    _render_quick_simulation_section()

    # Separator and Advanced simulation section
    st.divider()
    st.subheader("高级模拟")

    # Strategy section (moved here from app.py)
    render_strategy_section()

    # Resource configuration (for advanced simulation only)
    _render_resource_config()

    # Number of experiments
    num_experiments = st.number_input(
        "模拟次数",
        min_value=1,
        max_value=100000,
        value=st.session_state.get("num_experiments", 1000),
        step=100,
        key="num_experiments",
        on_change=_on_num_experiments_change,
        help="运行模拟的次数，次数越多结果越准确",
    )

    # Banner selection and config
    _render_banner_selection()

    # Auto banner configuration
    auto_config = _render_auto_banner_config()

    # Run button
    if st.session_state.banners:
        _render_run_button(num_experiments, auto_config)

    # Display results
    if st.session_state.run_results:
        _render_results()


def _render_banner_selection():
    """Render the banner selection section for simulation."""
    st.subheader("选择参与模拟的卡池")

    if st.session_state.banners:
        for banner in st.session_state.banners:
            banner_key = f"run_banner_{banner.name}"
            config_key = f"run_config_{banner.name}"

            # Initialize banner enabled state
            if banner.name not in st.session_state.run_banner_enabled:
                st.session_state.run_banner_enabled[banner.name] = True
            # Initialize banner strategy (use first strategy as default)
            default_strategy_name = st.session_state.strategies[0].name
            if banner.name not in st.session_state.run_banner_strategies:
                st.session_state.run_banner_strategies[banner.name] = (
                    default_strategy_name
                )

            with st.container(border=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.checkbox(
                        f"**{banner.name}**",
                        value=st.session_state.run_banner_enabled[banner.name],
                        key=banner_key,
                        on_change=_on_run_banner_change,
                        args=(banner.name,),
                    )
                    if banner.main_operator:
                        st.caption(f"UP: {banner.main_operator.name}")
                with col2:
                    if st.session_state.run_banner_enabled.get(banner.name, True):
                        # Strategy options from user strategies
                        strategy_options = [s.name for s in st.session_state.strategies]
                        current_selection = st.session_state.run_banner_strategies.get(
                            banner.name, default_strategy_name
                        )
                        if current_selection not in strategy_options:
                            current_selection = default_strategy_name
                        st.selectbox(
                            "策略",
                            strategy_options,
                            index=strategy_options.index(current_selection),
                            key=config_key,
                            label_visibility="collapsed",
                            on_change=_on_run_strategy_change,
                            args=(banner.name,),
                        )
    else:
        st.info("暂无卡池，请先创建卡池。")


def _render_auto_banner_config() -> dict:
    """Render the auto banner configuration section.

    Returns:
        Dictionary with auto banner settings:
        - count: Number of banners to auto-generate
        - template_idx: Index of the template to use
        - strategy_idx: Index of the strategy to use
    """
    st.subheader("自动添加卡池")
    st.caption("在已有卡池之后自动生成更多卡池继续模拟")

    with st.container(border=True):
        # Use value parameter to ensure we read from session state
        current_auto_count = st.session_state.get("auto_banner_count", 0)
        auto_count = st.number_input(
            "自动添加卡池数量",
            min_value=0,
            max_value=1000,
            value=current_auto_count,
            step=1,
            key="auto_banner_count",
            help="0表示不自动添加卡池",
            on_change=_on_auto_banner_config_change,
        )

        # Always show template and strategy selectors
        col1, col2 = st.columns(2)
        with col1:
            # Template selection
            template_names = [t.name for t in st.session_state.banner_templates]
            # Clamp index to valid range
            current_template_idx = st.session_state.get("auto_banner_template_idx", 0)
            if current_template_idx >= len(template_names):
                current_template_idx = 0
                st.session_state.auto_banner_template_idx = 0
            auto_template_idx = st.selectbox(
                "卡池模板",
                range(len(template_names)),
                index=current_template_idx,
                format_func=lambda x: template_names[x],
                key="auto_banner_template_idx",
                help="自动生成卡池使用的模板",
                on_change=_on_auto_banner_config_change,
            )
        with col2:
            # Strategy selection
            strategy_names = [s.name for s in st.session_state.strategies]
            # Clamp index to valid range
            current_strategy_idx = st.session_state.get("auto_banner_strategy_idx", 0)
            if current_strategy_idx >= len(strategy_names):
                current_strategy_idx = 0
                st.session_state.auto_banner_strategy_idx = 0
            auto_strategy_idx = st.selectbox(
                "抽卡策略",
                range(len(strategy_names)),
                index=current_strategy_idx,
                format_func=lambda x: strategy_names[x],
                key="auto_banner_strategy_idx",
                help="自动生成卡池使用的策略",
                on_change=_on_auto_banner_config_change,
            )

        return {
            "count": auto_count,
            "template_idx": auto_template_idx,
            "strategy_idx": auto_strategy_idx,
        }


def _render_run_button(num_experiments: int, auto_config: dict):
    """Render the run simulation button and execute simulation."""
    # Initialize confirmation state
    if "show_run_confirmation" not in st.session_state:
        st.session_state.show_run_confirmation = False
    if "run_simulation_confirmed" not in st.session_state:
        st.session_state.run_simulation_confirmed = False

    # Calculate total banner count for validation
    enabled_banner_count = sum(
        1
        for banner in st.session_state.banners
        if st.session_state.run_banner_enabled.get(banner.name, False)
    )
    auto_banner_count = auto_config.get("count", 0)
    total_banner_count = enabled_banner_count + auto_banner_count
    total_simulations = num_experiments * total_banner_count

    # Check if simulation was confirmed and should run
    if st.session_state.run_simulation_confirmed:
        st.session_state.run_simulation_confirmed = False
        _execute_simulation(num_experiments, auto_config)
        return

    # Validate total simulation count
    if total_simulations > 100000:
        st.error(
            f"模拟总数超过限制：{num_experiments} 次 × {total_banner_count} 卡池 = {total_simulations} > 100000。"
            f"请减少模拟次数或卡池数量。"
        )
        st.button("运行模拟", type="primary", disabled=True)
        return

    if st.button("运行模拟", type="primary"):
        # Show confirmation dialog instead of running immediately
        st.session_state.show_run_confirmation = True
        st.rerun()

    # Render confirmation dialog if active
    if st.session_state.show_run_confirmation:
        _render_run_confirmation_dialog(num_experiments, auto_config)


def _render_run_confirmation_dialog(num_experiments: int, auto_config: dict):
    """Render the confirmation dialog before running simulation."""

    @st.dialog("确认运行模拟", width="large")
    def confirmation_dialog():
        st.markdown("### 模拟配置概览")

        # Number of experiments
        st.markdown(f"**模拟次数:** {num_experiments}")

        # Resource configuration
        st.markdown("---")
        st.markdown("### 资源配置")
        config = st.session_state.config
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("初始抽数", config.initial_draws)
        with col2:
            st.metric("每期获得抽数", config.draws_gain_per_banner)
        with col3:
            st.metric("每期限定抽数", config.draws_gain_this_banner)

        # Build list of enabled banners and their strategies
        enabled_banners = []
        default_strategy = st.session_state.strategies[0]
        strategy_registry = {s.name: s for s in st.session_state.strategies}

        st.markdown("---")
        st.markdown("### 卡池配置")

        for banner in st.session_state.banners:
            if st.session_state.run_banner_enabled.get(banner.name, False):
                enabled_banners.append(banner)
                strategy_name = st.session_state.run_banner_strategies.get(
                    banner.name, default_strategy.name
                )
                strategy = strategy_registry.get(strategy_name, default_strategy)

                with st.expander(
                    f"**{banner.name}** → 策略: {strategy_name}",
                    expanded=False,
                ):
                    if banner.main_operator:
                        st.caption(f"UP干员: {banner.main_operator.name}")
                    # Show strategy description
                    strategy_desc = strategy.get_description(strategy_registry)
                    st.text(strategy_desc)

        if not enabled_banners:
            st.warning("⚠️ 没有选择任何卡池参与模拟")

        # Auto banner config
        auto_count = auto_config.get("count", 0)
        if auto_count > 0:
            st.markdown("---")
            st.markdown("### 自动添加卡池")
            st.markdown(f"**数量:** {auto_count}")

            template_idx = auto_config.get("template_idx", 0)
            strategy_idx = auto_config.get("strategy_idx", 0)

            if template_idx < len(st.session_state.banner_templates):
                template = st.session_state.banner_templates[template_idx]
                st.markdown(f"**模板:** {template.name}")

            if strategy_idx < len(st.session_state.strategies):
                auto_strategy = st.session_state.strategies[strategy_idx]
                with st.expander(f"**策略:** {auto_strategy.name}", expanded=False):
                    strategy_desc = auto_strategy.get_description(strategy_registry)
                    st.text(strategy_desc)

        st.markdown("---")

        # Confirmation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("取消", use_container_width=True):
                st.session_state.show_run_confirmation = False
                st.rerun()
        with col2:
            if st.button("确认运行", type="primary", use_container_width=True):
                st.session_state.show_run_confirmation = False
                st.session_state.run_simulation_confirmed = True
                st.rerun()

    confirmation_dialog()


def _execute_simulation(num_experiments: int, auto_config: dict):
    """Execute the simulation after confirmation."""
    # Build list of enabled banners
    enabled_banners = []
    banner_strategies = {}
    default_strategy = st.session_state.strategies[0]
    for banner in st.session_state.banners:
        if st.session_state.run_banner_enabled.get(banner.name, False):
            # Deep copy banner for simulation
            banner_copy = banner.model_copy(deep=True)
            enabled_banners.append(banner_copy)
            # Get strategy for this banner
            strategy_name = st.session_state.run_banner_strategies.get(
                banner.name, default_strategy.name
            )
            # Find strategy by name
            found_strategy = None
            for s in st.session_state.strategies:
                if s.name == strategy_name:
                    found_strategy = s
                    break
            if found_strategy:
                banner_strategies[banner.name] = DrawStrategy(
                    **found_strategy.model_dump()
                )
            else:
                banner_strategies[banner.name] = DrawStrategy(
                    **default_strategy.model_dump()
                )

    if enabled_banners:
        # Convert banners to dicts and back to ensure clean Pydantic validation
        banners_for_run = [Banner(**b.model_dump()) for b in enabled_banners]

        # Build auto banner configuration
        auto_banner_template = None
        auto_banner_strategy = None
        auto_banner_count = auto_config.get("count", 0)
        auto_banner_default_operators = []

        if auto_banner_count > 0:
            # Get template for auto banners
            template_idx = auto_config.get("template_idx", 0)
            if template_idx < len(st.session_state.banner_templates):
                auto_banner_template = st.session_state.banner_templates[
                    template_idx
                ].model_copy(deep=True)

            # Get strategy for auto banners
            strategy_idx = auto_config.get("strategy_idx", 0)
            if strategy_idx < len(st.session_state.strategies):
                auto_banner_strategy = DrawStrategy(
                    **st.session_state.strategies[strategy_idx].model_dump()
                )

            # Get default operators for auto banners
            auto_banner_default_operators = create_default_operators()

        run = Run(
            config=Config(**st.session_state.config.model_dump()),
            banner_strategies=banner_strategies,
            banners=banners_for_run,
            repeat=num_experiments,
            auto_banner_template=auto_banner_template,
            auto_banner_strategy=auto_banner_strategy,
            auto_banner_count=auto_banner_count,
            auto_banner_default_operators=auto_banner_default_operators,
        )

        # Calculate total banners for progress display
        total_banners = len(enabled_banners) + auto_banner_count

        # Run async simulation with progress bar
        progress_bar = st.progress(0, text=f"正在运行模拟... 0/{num_experiments}")

        def update_progress(current: int, total: int):
            progress = current / total if total > 0 else 0
            progress_bar.progress(progress, text=f"正在运行模拟... {current}/{total}")

        async def run_async():
            return await run.run_simulation_async(
                yield_every=max(1, num_experiments // 100),
                progress_callback=update_progress,
            )

        result_player = asyncio.run(run_async())
        progress_bar.empty()

        # Build banner names list including auto banners
        banner_names = [b.name for b in enabled_banners]
        if auto_banner_count > 0:
            banner_names.append(f"自动池×{auto_banner_count}")

        # Build main operators list (auto banners have dummy operators)
        main_operators = [
            b.main_operator.name for b in enabled_banners if b.main_operator
        ]

        st.session_state.run_results = {
            "player": result_player,
            "paid_draws": run.paid_draws,
            "total_draws": run.total_draws,
            "num_experiments": num_experiments,
            "banners": banner_names,
            "main_operators": main_operators,
            "total_banner_count": total_banners,
        }
        update_url()
        st.rerun()
    else:
        st.warning("请至少选择一个卡池参与模拟。")


def _render_results():
    """Render the simulation results."""
    st.subheader("模拟结果")
    results = st.session_state.run_results

    # Build banner display string
    banner_display = ", ".join(results["banners"])

    _render_simulation_results_shared(
        results=results,
        banner_display=banner_display,
        is_quick_sim=False,
        clear_button_key="clear_run_results",
        clear_state_key="run_results",
    )


def _render_operator_tables(player, num_exp: int):
    """Render operator statistics tables by rarity."""
    for rarity in [6, 5, 4]:
        if rarity in player.operators and player.operators[rarity]:
            color = RARITY_COLORS[rarity]
            st.markdown(
                f"<span style='color:{color}'><b>{rarity}星干员</b></span>",
                unsafe_allow_html=True,
            )

            # Create table data - sort by average potential (descending)
            table_data = []
            operators_with_buckets = []
            sorted_ops = sorted(
                player.operators[rarity].items(),
                key=lambda x: -x[1].potential,
            )
            for name, op in sorted_ops:
                avg_potential = (op.potential / num_exp) if num_exp > 0 else 0
                # Expected draws for first copy = total first draws / count of first draws
                expected_first_draw = (
                    (op.first_draw_total / op.first_draw_count)
                    if op.first_draw_count > 0
                    else 0
                )
                # Acquisition rate: percentage of experiments where operator was obtained
                acquisition_rate = (
                    (op.first_draw_count / num_exp * 100) if num_exp > 0 else 0
                )
                # Special draw ratio: how many of the total gains came from special draws
                special_ratio = (
                    (op.drawn_by_special_count / op.potential * 100)
                    if op.potential > 0
                    else 0
                )

                table_data.append(
                    {
                        "干员": name,
                        "平均数量": f"{avg_potential:.2f}",
                        "获取率": f"{acquisition_rate:.1f}%",
                        "首抽期望(非赠送抽)": f"{expected_first_draw:.1f}",
                        "特殊抽占比": f"{special_ratio:.1f}%",
                    }
                )
                # Store bucket data for histogram
                if op.draw_buckets and op.first_draw_count > 0:
                    operators_with_buckets.append((name, op))

            st.dataframe(table_data, width="stretch", hide_index=True)

            # Display histograms for operators with bucket data
            if operators_with_buckets:
                _render_histograms(operators_with_buckets)


def _render_histograms(operators_with_buckets: list):
    """Render histograms for first draw distribution."""
    st.markdown("**首抽分布(氪金)**")
    # Find the max bucket across all operators for consistent x-axis
    # Exclude -1 (special draw bucket) from min/max calculation
    all_buckets = set()
    has_special_bucket = False
    for _, op in operators_with_buckets:
        for bucket in op.draw_buckets.keys():
            if bucket == -1:
                has_special_bucket = True
            else:
                all_buckets.add(bucket)
    min_bucket = min(all_buckets) if all_buckets else 0
    max_bucket = max(all_buckets) if all_buckets else 120

    # Display histograms in columns (3 per row)
    cols_per_row = 3
    for i in range(0, len(operators_with_buckets), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(operators_with_buckets):
                name, op = operators_with_buckets[i + j]
                with col:
                    # Build histogram data with consistent buckets
                    bucket_labels = []
                    bucket_values = []
                    # Add normal buckets first (numeric order)
                    for bucket in range(min_bucket, max_bucket + 10, 10):
                        bucket_labels.append(str(bucket))
                        count = op.draw_buckets.get(bucket, 0)
                        pct = (
                            (count / op.first_draw_count * 100)
                            if op.first_draw_count > 0
                            else 0
                        )
                        bucket_values.append(pct)
                    # Add special bucket at the end if any operator has it
                    if has_special_bucket:
                        bucket_labels.append("sp")
                        count = op.draw_buckets.get(-1, 0)
                        pct = (
                            (count / op.first_draw_count * 100)
                            if op.first_draw_count > 0
                            else 0
                        )
                        bucket_values.append(pct)
                    # Create chart data with categorical ordering to prevent sorting
                    chart_df = pd.DataFrame(
                        {
                            "抽数": pd.Categorical(
                                bucket_labels, categories=bucket_labels, ordered=True
                            ),
                            "概率%": bucket_values,
                        }
                    )
                    st.caption(name)
                    st.bar_chart(
                        chart_df,
                        x="抽数",
                        y="概率%",
                        height=150,
                    )
