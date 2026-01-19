"""Sidebar component for adding/removing operators from banners."""

import streamlit as st

from banner import Operator
from ui.state import update_url


def render_operator_assignment():
    """Render the operator assignment sections in sidebar."""
    _render_add_operator()
    st.divider()
    _render_remove_operator()


def _render_add_operator():
    """Render the add operator to banner section."""
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


def _render_remove_operator():
    """Render the remove operator from banner section."""
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
