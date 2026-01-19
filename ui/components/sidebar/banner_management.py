"""Sidebar component for creating and deleting banners."""

import streamlit as st

from banner import Banner, Operator
from ui.defaults import create_default_operators
from ui.state import update_url


def render_banner_creation():
    """Render the banner creation section in sidebar."""
    st.header("创建卡池")
    banner_name = st.text_input(
        "卡池名称", value=f"卡池_{len(st.session_state.banners) + 1}"
    )
    # Template selection for new banner
    template_names = [t.name for t in st.session_state.banner_templates]
    selected_template_idx = st.selectbox(
        "卡池模板",
        range(len(template_names)),
        format_func=lambda x: template_names[x],
        key="new_banner_template",
    )
    if st.button("创建卡池"):
        # Add all default operators to the new banner
        default_ops = create_default_operators()
        banner_operators: dict[int, list[Operator]] = {}
        for op in default_ops:
            if op.rarity not in banner_operators:
                banner_operators[op.rarity] = []
            banner_operators[op.rarity].append(op)
        selected_template = st.session_state.banner_templates[selected_template_idx]
        new_banner = Banner(
            name=banner_name,
            operators=banner_operators,
            template=selected_template.model_copy(deep=True),
        )  # type: ignore
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

        selected_template = st.session_state.banner_templates[selected_template_idx]
        new_banner = Banner(
            name=f"卡池_{banner_idx}",
            operators=banner_operators,
            main_operator=dummy_main,
            template=selected_template.model_copy(deep=True),
        )  # type: ignore
        st.session_state.banners.append(new_banner)
        update_url()
        st.rerun()


def render_banner_deletion():
    """Render the banner deletion section in sidebar."""
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
