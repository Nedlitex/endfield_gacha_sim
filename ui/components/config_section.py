"""Resource configuration section component."""

import streamlit as st

from ui.state import update_url


def _on_config_change():
    """Callback when config values change."""
    st.session_state.config.initial_draws = st.session_state.config_initial_draws
    st.session_state.config.draws_gain_per_banner = (
        st.session_state.config_draws_per_banner
    )
    update_url()


def render_config_section():
    """Render the resource configuration section."""
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
            )
