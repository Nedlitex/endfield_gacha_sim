"""Header component with title and share/import/reset buttons."""

import streamlit as st

from ui.state import serialize_state


def render_header():
    """Render the header with title and action buttons."""
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
