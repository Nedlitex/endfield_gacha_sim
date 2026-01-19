"""Simulation control and results display component."""

import asyncio

import pandas as pd
import streamlit as st

from banner import Banner
from gacha import Config, Run
from strategy import DrawStrategy
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

    # Number of experiments
    num_experiments = st.number_input(
        "模拟次数",
        min_value=1,
        max_value=100000,
        value=1000,
        step=100,
        key="num_experiments",
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
        # Initialize session state defaults if not present
        if "auto_banner_count" not in st.session_state:
            st.session_state.auto_banner_count = 0
        if "auto_banner_template_idx" not in st.session_state:
            st.session_state.auto_banner_template_idx = 0
        if "auto_banner_strategy_idx" not in st.session_state:
            st.session_state.auto_banner_strategy_idx = 0

        auto_count = st.number_input(
            "自动添加卡池数量",
            min_value=0,
            max_value=1000,
            step=1,
            key="auto_banner_count",
            help="0表示不自动添加卡池",
            on_change=_on_auto_banner_config_change,
        )

        if auto_count > 0:
            col1, col2 = st.columns(2)
            with col1:
                # Template selection
                template_names = [t.name for t in st.session_state.banner_templates]
                # Clamp index to valid range
                if st.session_state.auto_banner_template_idx >= len(template_names):
                    st.session_state.auto_banner_template_idx = 0
                auto_template_idx = st.selectbox(
                    "卡池模板",
                    range(len(template_names)),
                    format_func=lambda x: template_names[x],
                    key="auto_banner_template_idx",
                    help="自动生成卡池使用的模板",
                    on_change=_on_auto_banner_config_change,
                )
            with col2:
                # Strategy selection
                strategy_names = [s.name for s in st.session_state.strategies]
                # Clamp index to valid range
                if st.session_state.auto_banner_strategy_idx >= len(strategy_names):
                    st.session_state.auto_banner_strategy_idx = 0
                auto_strategy_idx = st.selectbox(
                    "抽卡策略",
                    range(len(strategy_names)),
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

    return {"count": 0, "template_idx": 0, "strategy_idx": 0}


def _render_run_button(num_experiments: int, auto_config: dict):
    """Render the run simulation button and execute simulation."""
    # Initialize confirmation state
    if "show_run_confirmation" not in st.session_state:
        st.session_state.show_run_confirmation = False
    if "run_simulation_confirmed" not in st.session_state:
        st.session_state.run_simulation_confirmed = False

    # Check if simulation was confirmed and should run
    if st.session_state.run_simulation_confirmed:
        st.session_state.run_simulation_confirmed = False
        _execute_simulation(num_experiments, auto_config)
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
    player = results["player"]
    num_exp = results["num_experiments"]

    # Use total_banner_count if available (includes auto banners), otherwise len(banners)
    num_banners = results.get("total_banner_count", len(results["banners"]))
    avg_total_per_run = results["total_draws"] / num_exp if num_exp > 0 else 0
    avg_total_per_banner = avg_total_per_run / num_banners if num_banners > 0 else 0
    avg_paid_per_run = results["paid_draws"] / num_exp if num_exp > 0 else 0
    avg_paid_per_banner = avg_paid_per_run / num_banners if num_banners > 0 else 0

    # Calculate main operator first draw expectations
    main_ops = results.get("main_operators", [])
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
    st.markdown(
        f"**模拟次数:** {num_exp} | **参与卡池:** {', '.join(results['banners'])}"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("每池平均总抽数(含赠送抽)", f"{avg_total_per_banner:.1f}")
    with col2:
        st.metric("每池平均氪金抽数", f"{avg_paid_per_banner:.1f}")

    # Display main operator stats prominently
    if main_op_stats:
        st.markdown("#### 🎯 主要干员首抽期望(非赠送抽使用)")
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

    if st.button("清除结果"):
        st.session_state.run_results = None
        st.rerun()


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
