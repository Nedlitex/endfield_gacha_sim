"""Resource configuration section component."""

import streamlit as st

from ui.state import update_url


def _on_config_change():
    """Callback when config values change."""
    st.session_state.config.initial_draws = st.session_state.config_initial_draws
    st.session_state.config.draws_gain_per_banner = (
        st.session_state.config_draws_per_banner
    )
    st.session_state.config.draws_gain_per_banner_start_at = (
        st.session_state.config_draws_per_banner_start_at
    )
    st.session_state.config.draws_gain_this_banner = (
        st.session_state.config_draws_this_banner
    )
    update_url()


def render_config_section():
    """Render the resource configuration section."""
    st.header("设置")

    # Config settings (initial draws and draws per banner)
    st.subheader("资源配置")
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
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
                "从第N期开始",
                min_value=1,
                value=st.session_state.config.draws_gain_per_banner_start_at,
                step=1,
                key="config_draws_per_banner_start_at",
                on_change=_on_config_change,
                help="每期卡池获得抽数从第几期开始生效（跳过前N-1期）",
            )
        with col4:
            st.number_input(
                "每期限定抽数",
                min_value=0,
                value=st.session_state.config.draws_gain_this_banner,
                step=1,
                key="config_draws_this_banner",
                on_change=_on_config_change,
                help="每期卡池获得的限定抽数，仅限当期使用，不结转",
            )
