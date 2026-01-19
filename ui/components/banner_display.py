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
                    # Display operators by rarity
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

                    st.divider()

                    # Template info with popover for details and change
                    _render_template_section(banner, idx)
    else:
        st.info("暂无卡池，请在侧边栏创建卡池。")


def _render_template_section(banner, banner_idx: int):
    """Render the template section for a banner with details and change option."""
    template = banner.template

    with st.popover(f"📋 {template.name}", use_container_width=True):
        # Show template description as markdown
        st.markdown(template.get_description())

        st.divider()

        # Change template option
        st.markdown("**更换模板**")
        template_names = [t.name for t in st.session_state.banner_templates]
        current_template_idx = 0
        for i, t in enumerate(st.session_state.banner_templates):
            if t.name == template.name:
                current_template_idx = i
                break

        new_template_idx = st.selectbox(
            "选择新模板",
            range(len(template_names)),
            index=current_template_idx,
            format_func=lambda x: template_names[x],
            key=f"banner_{banner_idx}_change_template",
        )

        if st.button("应用模板", key=f"banner_{banner_idx}_apply_template"):
            new_template = st.session_state.banner_templates[new_template_idx]
            banner.template = new_template.model_copy(deep=True)
            update_url()
            st.rerun()
