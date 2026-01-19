"""Sidebar component for creating operators."""

import streamlit as st

from banner import Operator
from ui.state import update_url


def render_operator_management():
    """Render the operator creation section in sidebar."""
    st.header("创建干员")
    operator_name = st.text_input("干员名称")
    operator_rarity = st.selectbox("稀有度", [6, 5, 4])
    if st.button("创建干员"):
        if operator_name:
            new_operator = Operator(name=operator_name, rarity=operator_rarity)  # type: ignore
            st.session_state.operators.append(new_operator)
            update_url()
            st.rerun()
