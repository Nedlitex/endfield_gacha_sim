import asyncio
import base64
import json
import zlib

import pandas as pd
import streamlit as st

from gacha import Banner, Config, DrawStrategy, Operator, Player, Run

st.set_page_config(page_title="终末地抽卡策略模拟器", layout="wide")


def create_default_operators() -> list[Operator]:
    """创建默认干员列表"""
    operators = [
        Operator(name="别礼", rarity=6),  # type: ignore
        Operator(name="黎风", rarity=6),  # type: ignore
        Operator(name="骏卫", rarity=6),  # type: ignore
        Operator(name="埃尔黛拉", rarity=6),  # type: ignore
        Operator(name="余烬", rarity=6),  # type: ignore
        Operator(name="佩丽卡", rarity=5),  # type: ignore
        Operator(name="狼卫", rarity=5),  # type: ignore
        Operator(name="赛希", rarity=5),  # type: ignore
        Operator(name="艾维文娜", rarity=5),  # type: ignore
        Operator(name="陈千语", rarity=5),  # type: ignore
        Operator(name="大潘", rarity=5),  # type: ignore
        Operator(name="阿列什", rarity=5),  # type: ignore
        Operator(name="弧光", rarity=5),  # type: ignore
        Operator(name="昼雪", rarity=5),  # type: ignore
        Operator(name="秋栗", rarity=4),  # type: ignore
        Operator(name="安塔尔", rarity=4),  # type: ignore
        Operator(name="卡契尔", rarity=4),  # type: ignore
        Operator(name="埃特拉", rarity=4),  # type: ignore
        Operator(name="萤石", rarity=4),  # type: ignore
    ]
    return operators


def create_default_banners() -> list[Banner]:
    """创建默认卡池列表"""
    # Get default operators grouped by rarity
    default_ops = create_default_operators()
    base_operators: dict[int, list[Operator]] = {}
    for op in default_ops:
        if op.rarity not in base_operators:
            base_operators[op.rarity] = []
        base_operators[op.rarity].append(op)

    # Banner-specific 6-star operators
    Laevatain = Operator(name="莱万汀", rarity=6)  # type: ignore
    Gilberta = Operator(name="洁尔佩塔", rarity=6)  # type: ignore
    Yvonne = Operator(name="伊冯", rarity=6)  # type: ignore

    banners = []

    # Banner 1: 熔火灼痕 - 莱万汀 main
    banner1_ops = {
        6: [Laevatain, Gilberta, Yvonne] + base_operators.get(6, []),
        5: base_operators.get(5, []).copy(),
        4: base_operators.get(4, []).copy(),
    }
    banners.append(
        Banner(name="熔火灼痕", operators=banner1_ops, main_operator=Laevatain)  # type: ignore
    )

    # Banner 2: 轻飘飘的信使 - 洁尔佩塔 main
    banner2_ops = {
        6: [Laevatain, Gilberta, Yvonne] + base_operators.get(6, []),
        5: base_operators.get(5, []).copy(),
        4: base_operators.get(4, []).copy(),
    }
    banners.append(
        Banner(name="轻飘飘的信使", operators=banner2_ops, main_operator=Gilberta)  # type: ignore
    )

    # Banner 3: 热烈色彩 - 伊冯 main
    banner3_ops = {
        6: [Laevatain, Gilberta, Yvonne] + base_operators.get(6, []),
        5: base_operators.get(5, []).copy(),
        4: base_operators.get(4, []).copy(),
    }
    banners.append(
        Banner(name="热烈色彩", operators=banner3_ops, main_operator=Yvonne)  # type: ignore
    )

    return banners


def create_default_strategy() -> DrawStrategy:
    """创建默认策略 - 氪金抽到UP"""
    return DrawStrategy(name="默认策略(氪金抽到UP)", pay=True)


def serialize_state() -> str:
    """序列化当前状态为压缩的base64字符串"""
    # Serialize run results if present
    run_results_data = None
    if st.session_state.get("run_results"):
        results = st.session_state.run_results
        run_results_data = {
            "player": results["player"].model_dump(),
            "paid_draws": results["paid_draws"],
            "total_draws": results["total_draws"],
            "num_experiments": results["num_experiments"],
            "banners": results["banners"],
            "main_operators": results.get("main_operators", []),
        }

    state = {
        "operators": [op.model_dump() for op in st.session_state.operators],
        "banners": [banner.model_dump() for banner in st.session_state.banners],
        "box": st.session_state.box.model_dump(),
        "config": st.session_state.config.model_dump(),
        "strategies": [s.model_dump() for s in st.session_state.strategies],
        "current_strategy_idx": st.session_state.current_strategy_idx,
        "run_banner_enabled": st.session_state.get("run_banner_enabled", {}),
        "run_banner_strategies": st.session_state.get("run_banner_strategies", {}),
        "run_results": run_results_data,
    }
    json_str = json.dumps(state, ensure_ascii=False)
    compressed = zlib.compress(json_str.encode(), level=9)
    return base64.urlsafe_b64encode(compressed).decode()


def deserialize_state(encoded: str) -> dict:
    """从压缩的base64字符串反序列化状态"""
    compressed = base64.urlsafe_b64decode(encoded.encode())
    try:
        # Try decompressing (new format)
        json_str = zlib.decompress(compressed).decode()
    except zlib.error:
        # Fall back to old uncompressed format
        json_str = compressed.decode()
    return json.loads(json_str)


def update_url():
    """更新URL参数"""
    encoded = serialize_state()
    st.query_params["state"] = encoded


# Initialize session state from URL or defaults
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    params = st.query_params
    if "state" in params:
        try:
            state = deserialize_state(params["state"])
            st.session_state.operators = [
                Operator(**op) for op in state.get("operators", [])
            ]
            st.session_state.banners = [
                Banner(**banner) for banner in state.get("banners", [])
            ]
            st.session_state.box = Player(**state.get("box", {}))
            # Load config
            if "config" in state:
                st.session_state.config = Config(**state.get("config", {}))
            else:
                st.session_state.config = Config()
            # Load strategies
            if "strategies" in state:
                st.session_state.strategies = [
                    DrawStrategy(**s) for s in state.get("strategies", [])
                ]
                st.session_state.current_strategy_idx = state.get(
                    "current_strategy_idx", 0
                )
            else:
                st.session_state.strategies = [create_default_strategy()]
                st.session_state.current_strategy_idx = 0
            # Load run state
            st.session_state.run_banner_enabled = state.get("run_banner_enabled", {})
            st.session_state.run_banner_strategies = state.get(
                "run_banner_strategies", {}
            )
            # Load run results
            if "run_results" in state and state["run_results"]:
                run_data = state["run_results"]
                st.session_state.run_results = {
                    "player": Player(**run_data["player"]),
                    "paid_draws": run_data["paid_draws"],
                    "total_draws": run_data["total_draws"],
                    "num_experiments": run_data["num_experiments"],
                    "banners": run_data["banners"],
                    "main_operators": run_data.get("main_operators", []),
                }
            else:
                st.session_state.run_results = None
        except Exception:
            st.session_state.operators = create_default_operators()
            st.session_state.banners = create_default_banners()
            st.session_state.box = Player()
            st.session_state.config = Config()
            st.session_state.strategies = [create_default_strategy()]
            st.session_state.current_strategy_idx = 0
            st.session_state.run_banner_enabled = {}
            st.session_state.run_banner_strategies = {}
    else:
        st.session_state.operators = create_default_operators()
        st.session_state.banners = create_default_banners()
        st.session_state.box = Player()
        st.session_state.config = Config()
        st.session_state.strategies = [create_default_strategy()]
        st.session_state.current_strategy_idx = 0
        st.session_state.run_banner_enabled = {}
        st.session_state.run_banner_strategies = {}

st.title("终末地抽卡策略模拟器")

# Share, load, and reset buttons (packed into 1/5 width column)
col_buttons, _ = st.columns([1, 4])
with col_buttons:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.popover("分享"):
            serialized = serialize_state()
            st.code(serialized, language=None)
            st.caption("请复制上方的配置字符串进行分享")
    with c2:
        with st.popover("导入"):
            load_input = st.text_area("粘贴配置字符串", height=100)
            if st.button("加载"):
                if load_input.strip():
                    try:
                        st.session_state.clear()
                        st.query_params["state"] = load_input.strip()
                        st.rerun()
                    except Exception as e:
                        st.error(f"加载失败: {e}")
                else:
                    st.warning("请先粘贴配置字符串")
    with c3:
        with st.popover("重置"):
            st.warning("确定要重置所有数据吗？此操作不可撤销。")
            if st.button("确认重置", type="primary"):
                st.session_state.clear()
                st.query_params.clear()
                st.markdown(
                    '<meta http-equiv="refresh" content="0; url=/">',
                    unsafe_allow_html=True,
                )

# Sidebar for creating banners, operators, and adding operators to banners
with st.sidebar:
    st.header("创建干员")
    operator_name = st.text_input("干员名称")
    operator_rarity = st.selectbox("稀有度", [6, 5, 4])
    if st.button("创建干员"):
        if operator_name:
            new_operator = Operator(name=operator_name, rarity=operator_rarity)  # type: ignore
            st.session_state.operators.append(new_operator)
            update_url()
            st.rerun()

    st.divider()

    st.header("创建卡池")
    banner_name = st.text_input(
        "卡池名称", value=f"卡池_{len(st.session_state.banners) + 1}"
    )
    if st.button("创建卡池"):
        # Add all default operators to the new banner
        default_ops = create_default_operators()
        banner_operators: dict[int, list[Operator]] = {}
        for op in default_ops:
            if op.rarity not in banner_operators:
                banner_operators[op.rarity] = []
            banner_operators[op.rarity].append(op)
        new_banner = Banner(name=banner_name, operators=banner_operators)  # type: ignore
        st.session_state.banners.append(new_banner)
        update_url()
        st.rerun()

    if st.button("创建下一期卡池"):
        # Create a dummy banner with:
        # - A new dummy main operator
        # - Main operators from previous two banners
        # - All default operators
        banners = st.session_state.banners
        banner_idx = len(banners) + 1
        dummy_main = Operator(name=f"新干员_{banner_idx}", rarity=6)  # type: ignore

        # Get default operators grouped by rarity
        default_ops = create_default_operators()
        banner_operators: dict[int, list[Operator]] = {}
        for op in default_ops:
            if op.rarity not in banner_operators:
                banner_operators[op.rarity] = []
            banner_operators[op.rarity].append(op)

        # Add dummy main operator
        if 6 not in banner_operators:
            banner_operators[6] = []
        banner_operators[6].insert(0, dummy_main)

        # Add main operators from previous two banners
        prev_mains = []
        for banner in banners[-2:]:
            if banner.main_operator:
                prev_mains.append(banner.main_operator)
        for main_op in prev_mains:
            # Avoid duplicates
            existing_names = [
                op.name for op in banner_operators.get(main_op.rarity, [])
            ]
            if main_op.name not in existing_names:
                banner_operators[main_op.rarity].insert(0, main_op)

        new_banner = Banner(
            name=f"卡池_{banner_idx}",
            operators=banner_operators,
            main_operator=dummy_main,
        )  # type: ignore
        st.session_state.banners.append(new_banner)
        update_url()
        st.rerun()

    st.divider()

    st.header("添加干员到卡池")
    if st.session_state.banners:
        banner_names = [p.name for p in st.session_state.banners]
        selected_banner_name_add = st.selectbox(
            "选择卡池", banner_names, key="add_banner"
        )
        # Collect all unique operators from session operators and all banners
        all_operators: dict[str, Operator] = {}
        for op in st.session_state.operators:
            all_operators[op.name] = op
        for banner in st.session_state.banners:
            for rarity in banner.operators:
                for op in banner.operators[rarity]:
                    if op.name not in all_operators:
                        all_operators[op.name] = op
        all_operators_list = sorted(
            all_operators.values(), key=lambda x: (-x.rarity, x.name)
        )

        if all_operators_list:
            operator_names = [f"{op.name} ({op.rarity}星)" for op in all_operators_list]
            selected_operator_idx = st.selectbox(
                "选择干员",
                range(len(operator_names)),
                format_func=lambda x: operator_names[x],
            )
            # Only show "set as main" checkbox for 6-star operators
            selected_operator_preview = all_operators_list[selected_operator_idx]
            set_as_main = False
            if selected_operator_preview.rarity == 6:
                set_as_main = st.checkbox("设为UP干员", key="add_op_set_main")

            if st.button("添加到卡池"):
                selected_operator = all_operators_list[selected_operator_idx]
                for banner in st.session_state.banners:
                    if banner.name == selected_banner_name_add:
                        if selected_operator.rarity not in banner.operators:
                            banner.operators[selected_operator.rarity] = []
                        # Check if operator already exists in banner
                        existing_idx = None
                        for i, op in enumerate(
                            banner.operators[selected_operator.rarity]
                        ):
                            if op.name == selected_operator.name:
                                existing_idx = i
                                break
                        if existing_idx is not None:
                            # Replace existing operator
                            banner.operators[selected_operator.rarity][
                                existing_idx
                            ] = selected_operator
                        else:
                            # Add new operator
                            banner.operators[selected_operator.rarity].append(
                                selected_operator
                            )
                        # Set as main operator if checkbox was checked
                        if set_as_main and selected_operator.rarity == 6:
                            banner.main_operator = selected_operator
                        update_url()
                        st.rerun()
        else:
            st.info("请先创建干员")
    else:
        st.info("请先创建卡池")

    st.divider()

    st.header("从卡池移除干员")
    if st.session_state.banners:
        banner_names = [p.name for p in st.session_state.banners]
        selected_banner_name_remove = st.selectbox(
            "选择卡池", banner_names, key="remove_banner"
        )
        # Get operators in selected banner
        selected_banner_for_remove = None
        for banner in st.session_state.banners:
            if banner.name == selected_banner_name_remove:
                selected_banner_for_remove = banner
                break
        if selected_banner_for_remove:
            banner_ops = []
            for rarity in [6, 5, 4]:
                if rarity in selected_banner_for_remove.operators:
                    for op in selected_banner_for_remove.operators[rarity]:
                        banner_ops.append((op, rarity))
            if banner_ops:
                op_display = [f"{op.name} ({r}星)" for op, r in banner_ops]
                selected_op_remove_idx = st.selectbox(
                    "选择干员",
                    range(len(op_display)),
                    format_func=lambda x: op_display[x],
                    key="remove_op",
                )
                if st.button("从卡池移除"):
                    op_to_remove, rarity = banner_ops[selected_op_remove_idx]
                    selected_banner_for_remove.operators[rarity].remove(op_to_remove)
                    update_url()
                    st.rerun()
            else:
                st.info("该卡池暂无干员")
    else:
        st.info("请先创建卡池")

    st.divider()

    st.header("删除卡池")
    if st.session_state.banners:
        banner_names = [p.name for p in st.session_state.banners]
        selected_banner_name_delete = st.selectbox(
            "选择卡池", banner_names, key="delete_banner"
        )
        if st.button("删除卡池"):
            st.session_state.banners = [
                p
                for p in st.session_state.banners
                if p.name != selected_banner_name_delete
            ]
            update_url()
            st.rerun()
    else:
        st.info("暂无卡池")

# Rarity colors
RARITY_COLORS = {6: "red", 5: "orange", 4: "purple"}

# Display banners as boxes
st.header("卡池")
col1, col2, col3 = st.columns([1, 1, 10])
with col1:
    if st.button("展开全部"):
        for banner in st.session_state.banners:
            banner.expanded = True
        update_url()
        st.rerun()
with col2:
    if st.button("折叠全部"):
        for banner in st.session_state.banners:
            banner.expanded = False
        update_url()
        st.rerun()

if st.session_state.banners:
    cols = st.columns(min(len(st.session_state.banners), 3))
    for idx, banner in enumerate(st.session_state.banners):
        with cols[idx % 3]:
            with st.expander(banner.name, expanded=banner.expanded):
                for rarity in [6, 5, 4]:
                    if rarity in banner.operators and banner.operators[rarity]:
                        color = RARITY_COLORS[rarity]
                        op_names = []
                        for op in banner.operators[rarity]:
                            if (
                                banner.main_operator
                                and op.name == banner.main_operator.name
                            ):
                                op_names.append(f"**{op.name}(UP)**")
                            else:
                                op_names.append(op.name)
                        names_str = ", ".join(op_names)
                        st.markdown(
                            f"<span style='color:{color}'><b>{rarity}星:</b> {names_str}</span>",
                            unsafe_allow_html=True,
                        )
                if not any(banner.operators.get(r) for r in [4, 5, 6]):
                    st.write("*暂无干员*")
else:
    st.info("暂无卡池，请在侧边栏创建卡池。")


# Config section (initial draws and draws per banner)
def on_config_change():
    st.session_state.config.initial_draws = st.session_state.config_initial_draws
    st.session_state.config.draws_gain_per_banner = (
        st.session_state.config_draws_per_banner
    )
    update_url()


# Strategy section
def get_current_strategy():
    return st.session_state.strategies[st.session_state.current_strategy_idx]


def on_strategy_change():
    strategy = get_current_strategy()
    strategy.always_single_draw = st.session_state.strategy_always_single_draw
    strategy.single_draw_after = st.session_state.strategy_single_draw_after
    strategy.skip_banner_threshold = st.session_state.strategy_skip_banner_threshold
    strategy.min_draws_per_banner = st.session_state.strategy_min_draws_per_banner
    strategy.pay = st.session_state.strategy_pay
    update_url()


st.header("设置")

# Config settings (initial draws and draws per banner)
st.subheader("资源配置")
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "初始抽数",
            min_value=0,
            value=st.session_state.config.initial_draws,
            step=1,
            key="config_initial_draws",
            on_change=on_config_change,
        )
    with col2:
        st.number_input(
            "每期卡池获得抽数",
            min_value=0,
            value=st.session_state.config.draws_gain_per_banner,
            step=1,
            key="config_draws_per_banner",
            on_change=on_config_change,
        )

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
        key="strategy_selector",
    )
    if selected_strategy_idx != st.session_state.current_strategy_idx:
        st.session_state.current_strategy_idx = selected_strategy_idx
        update_url()
        st.rerun()
with col2:
    pass  # Delete button moved to strategy creation section

current_strategy = get_current_strategy()

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
        st.markdown(f"**策略说明:** 氪金抽到UP（抽数不足时额外获得抽数以满足规则）")
    else:
        # Editable view for custom strategies
        col3, col4 = st.columns(2)
        with col3:
            st.number_input(
                "每池最少抽数",
                min_value=0,
                value=current_strategy.min_draws_per_banner,
                step=1,
                key="strategy_min_draws_per_banner",
                on_change=on_strategy_change,
            )
        with col4:
            st.number_input(
                "跳池阈值",
                min_value=0,
                value=current_strategy.skip_banner_threshold,
                step=1,
                key="strategy_skip_banner_threshold",
                on_change=on_strategy_change,
                help="剩余抽数低于此值时跳过当前卡池",
            )

        col5, col6 = st.columns(2)
        with col5:
            st.checkbox(
                "始终单抽(特殊10连除外)",
                value=current_strategy.always_single_draw,
                key="strategy_always_single_draw",
                on_change=on_strategy_change,
            )
        with col6:
            st.number_input(
                "累计抽数后单抽",
                min_value=0,
                value=current_strategy.single_draw_after,
                step=1,
                key="strategy_single_draw_after",
                on_change=on_strategy_change,
                help="累计抽数达到此值后开始单抽(特殊10连除外)",
            )

        st.checkbox(
            "氪金(抽数不足时额外获得抽数以满足规则)",
            value=current_strategy.pay,
            key="strategy_pay",
            on_change=on_strategy_change,
        )

        # min_draws_after_main: list of (threshold, target) tuples
        st.subheader("获得UP后规则")
        st.caption("当前抽数 < 阈值时，获得UP后继续抽至目标抽数")

        # Display existing rules
        for idx, (threshold, target) in enumerate(
            current_strategy.min_draws_after_main
        ):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.text(f"获得UP后若当前抽数小于{threshold}则继续抽至{target}")
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
                current_strategy.min_draws_after_main.append(
                    (new_threshold, new_target)
                )
                update_url()
                st.rerun()

        # min_draws_after_pity: list of (threshold, target) tuples
        st.subheader("小保底歪了后规则")
        st.caption("当前抽数 < 阈值时，歪了(触发小保底但未获得UP)后继续抽至目标抽数")

        # Display existing rules
        for idx, (threshold, target) in enumerate(
            current_strategy.min_draws_after_pity
        ):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.text(f"歪了后若当前抽数小于{threshold}则继续抽至{target}")
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

        # Generate strategy summary button
        if st.button("生成策略说明"):
            paragraphs = []
            paragraphs.append(f"【{current_strategy.name}】")
            paragraphs.append(
                f"玩家初始拥有{st.session_state.config.initial_draws}抽，每期卡池额外获得{st.session_state.config.draws_gain_per_banner}抽。"
            )

            if current_strategy.min_draws_per_banner > 0:
                paragraphs.append(
                    f"每个卡池至少抽{current_strategy.min_draws_per_banner}抽。"
                )

            if current_strategy.skip_banner_threshold > 0:
                paragraphs.append(
                    f"当剩余抽数低于{current_strategy.skip_banner_threshold}时，跳过当前卡池不再抽取。"
                )

            if current_strategy.always_single_draw:
                paragraphs.append("抽卡时始终单抽，特殊10连除外。")
            elif current_strategy.single_draw_after > 0:
                paragraphs.append(
                    f"当累计抽数达到{current_strategy.single_draw_after}后，改为单抽以节省资源，特殊10连除外。"
                )

            if current_strategy.min_draws_after_main:
                rules_desc = []
                for threshold, target in current_strategy.min_draws_after_main:
                    rules_desc.append(f"若当前抽数小于{threshold}则继续抽至{target}抽")
                paragraphs.append(f"获得UP干员后，{'；'.join(rules_desc)}。")

            if current_strategy.min_draws_after_pity:
                rules_desc = []
                for threshold, target in current_strategy.min_draws_after_pity:
                    rules_desc.append(f"若当前抽数小于{threshold}则继续抽至{target}抽")
                paragraphs.append(
                    f"歪了(触发小保底但未获得UP)后，{'；'.join(rules_desc)}。"
                )

            if current_strategy.pay:
                paragraphs.append(":red[**抽数不足时氪金补充抽数以满足规则。**]")

            st.info("\n\n".join(paragraphs))

# New strategy creation (after the expander)
new_strategy_name = st.text_input(
    "新策略名称", value="", key="new_strategy_name", placeholder="输入新策略名称"
)
col_create, col_delete = st.columns(2)
with col_create:
    if st.button("创建策略"):
        if new_strategy_name:
            st.session_state.strategies.append(DrawStrategy(name=new_strategy_name))
            st.session_state.current_strategy_idx = len(st.session_state.strategies) - 1
            update_url()
            st.rerun()
with col_delete:
    # Delete strategy button (only if more than one strategy exists and not the default strategy)
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


# Run section
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
st.subheader("选择参与模拟的卡池")


def on_run_banner_change(banner_name: str):
    """Callback when banner enabled checkbox changes"""
    st.session_state.run_banner_enabled[banner_name] = st.session_state[
        f"run_banner_{banner_name}"
    ]
    update_url()


def on_run_strategy_change(banner_name: str):
    """Callback when banner strategy selection changes"""
    st.session_state.run_banner_strategies[banner_name] = st.session_state[
        f"run_config_{banner_name}"
    ]
    update_url()


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
            st.session_state.run_banner_strategies[banner.name] = default_strategy_name

        with st.container(border=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.checkbox(
                    f"**{banner.name}**",
                    value=st.session_state.run_banner_enabled[banner.name],
                    key=banner_key,
                    on_change=on_run_banner_change,
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
                        on_change=on_run_strategy_change,
                        args=(banner.name,),
                    )
else:
    st.info("暂无卡池，请先创建卡池。")

# Run button
if st.session_state.banners:
    if st.button("运行模拟", type="primary"):
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
            run = Run(
                config=Config(**st.session_state.config.model_dump()),
                banner_strategies=banner_strategies,
                banners=banners_for_run,
                repeat=num_experiments,
            )

            # Run async simulation with progress bar
            progress_bar = st.progress(0, text=f"正在运行模拟... 0/{num_experiments}")

            def update_progress(current: int, total: int):
                progress = current / total if total > 0 else 0
                progress_bar.progress(
                    progress, text=f"正在运行模拟... {current}/{total}"
                )

            async def run_async():
                return await run.run_simulation_async(
                    yield_every=max(1, num_experiments // 100),
                    progress_callback=update_progress,
                )

            result_player = asyncio.run(run_async())
            progress_bar.empty()

            st.session_state.run_results = {
                "player": result_player,
                "paid_draws": run.paid_draws,
                "total_draws": run.total_draws,
                "num_experiments": num_experiments,
                "banners": [b.name for b in enabled_banners],
                "main_operators": [
                    b.main_operator.name for b in enabled_banners if b.main_operator
                ],
            }
            update_url()
            st.rerun()
        else:
            st.warning("请至少选择一个卡池参与模拟。")

# Display results
if st.session_state.run_results:
    st.subheader("模拟结果")
    results = st.session_state.run_results
    player = results["player"]
    num_exp = results["num_experiments"]

    num_banners = len(results["banners"])
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
            main_op_stats.append((main_name, expected_first))

    # Display summary stats prominently
    st.markdown(
        f"**模拟次数:** {num_exp} | **参与卡池:** {', '.join(results['banners'])}"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("每池平均总抽数", f"{avg_total_per_banner:.1f}")
    with col2:
        st.metric("每池平均氪金抽数", f"{avg_paid_per_banner:.1f}")

    # Display main operator stats prominently
    if main_op_stats:
        st.markdown("#### 🎯 主要干员首抽期望(非赠送抽使用)")
        cols = st.columns(len(main_op_stats))
        for i, (name, expected) in enumerate(main_op_stats):
            with cols[i]:
                st.metric(name, f"{expected:.1f} 抽")

    # Display operators by rarity
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
                st.markdown("**首抽分布(氪金)**")
                # Find the max bucket across all operators for consistent x-axis
                all_buckets = set()
                for _, op in operators_with_buckets:
                    all_buckets.update(op.draw_buckets.keys())
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
                                for bucket in range(min_bucket, max_bucket + 10, 10):
                                    bucket_labels.append(
                                        bucket
                                    )  # Keep as int for proper sorting
                                    count = op.draw_buckets.get(bucket, 0)
                                    pct = (
                                        (count / op.first_draw_count * 100)
                                        if op.first_draw_count > 0
                                        else 0
                                    )
                                    bucket_values.append(pct)
                                # Create chart data
                                chart_df = pd.DataFrame(
                                    {
                                        "抽数": bucket_labels,
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

    if st.button("清除结果"):
        st.session_state.run_results = None
        st.rerun()
