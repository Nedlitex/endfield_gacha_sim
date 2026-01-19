"""Gacha draw strategy simulator for EndField."""

import streamlit as st

st.set_page_config(page_title="终末地抽卡策略模拟器", layout="wide")

from ui.components.banner_display import render_banner_display
from ui.components.header import render_header
from ui.components.sidebar import (
    render_banner_creation,
    render_banner_deletion,
    render_operator_assignment,
    render_operator_management,
    render_template_management,
)
from ui.components.simulation_section import render_simulation_section
from ui.components.strategy_section import render_strategy_section
from ui.state import initialize_session_state

# Initialize state from URL or defaults
initialize_session_state()

# Render header with title and action buttons
render_header()

# Sidebar for creating banners, operators, and adding operators to banners
with st.sidebar:
    render_operator_management()
    st.divider()
    render_banner_creation()
    st.divider()
    render_operator_assignment()
    st.divider()
    render_banner_deletion()
    st.divider()
    render_template_management()

# Main content
render_banner_display()
render_strategy_section()
render_simulation_section()
