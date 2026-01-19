"""Banner display grid component."""

import streamlit as st

from ui.constants import RARITY_COLORS
from ui.state import update_url


def render_banner_display():
    """Render the banner display grid with expand/collapse controls."""
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
