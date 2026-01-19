"""Sidebar component for managing banner templates."""

import streamlit as st

from banner import (
    BannerTemplate,
    InheritPolicy,
    RarityProbability,
    RepeatPolicy,
    ResetCondition,
)
from ui.state import update_url


def render_template_management():
    """Render the template management section in sidebar."""
    st.header("卡池模板管理")

    # Show existing templates
    if st.session_state.banner_templates:
        for idx, template in enumerate(st.session_state.banner_templates):
            with st.expander(f"📋 {template.name}", expanded=False):
                st.markdown(f"**{template.name}**")
                st.caption(
                    f"稀有度: {', '.join(str(r) + '星' for r in sorted(template.rarities))}"
                )
                # Show key parameters
                if template.has_pity_draw:
                    st.caption(
                        f"小保底: 第{template.pity_draw_start + 1}抽开始提升, 第{template.pity_draw_limit}抽必出"
                    )
                if template.has_definitive_draw:
                    st.caption(f"大保底: 第{template.definitive_draw_count}抽必得UP")
                if template.has_potential_reward:
                    st.caption(f"潜能奖励: 每{template.potential_reward_draw}抽")

                # Delete button (don't allow deleting the last template)
                if len(st.session_state.banner_templates) > 1:
                    if st.button("删除模板", key=f"delete_template_{idx}"):
                        st.session_state.banner_templates.pop(idx)
                        update_url()
                        st.rerun()

    # Create new template popup
    _render_template_creator()


def _render_template_creator():
    """Render the template creation popover."""
    with st.popover("创建新模板", use_container_width=True):
        st.subheader("创建卡池模板")

        new_template_name = st.text_input(
            "模板名称", value="自定义模板", key="new_template_name"
        )

        st.markdown("**稀有度设置**")
        col1, col2, col3 = st.columns(3)
        with col1:
            prob_r4 = st.number_input(
                "4星概率",
                min_value=0.0,
                max_value=1.0,
                value=0.912,
                step=0.01,
                key="prob_r4",
            )
        with col2:
            prob_r5 = st.number_input(
                "5星概率",
                min_value=0.0,
                max_value=1.0,
                value=0.08,
                step=0.01,
                key="prob_r5",
            )
        with col3:
            prob_r6 = st.number_input(
                "6星概率",
                min_value=0.0,
                max_value=1.0,
                value=0.008,
                step=0.001,
                format="%.3f",
                key="prob_r6",
            )

        main_prob = st.number_input(
            "UP干员概率",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="main_prob",
            help="抽到对应稀有度时，获得UP干员的概率",
        )

        st.markdown("**小保底设置**")
        has_pity = st.checkbox("启用小保底", value=True, key="has_pity")
        pity_start = 66
        pity_limit = 80
        pity_boost = 0.05
        pity_inherit_policy = InheritPolicy.NO_INHERIT
        pity_repeat_policy = RepeatPolicy.NO_REPEAT
        pity_reset_condition = ResetCondition.ON_HIGHEST_RARITY
        if has_pity:
            col1, col2 = st.columns(2)
            with col1:
                pity_start = st.number_input(
                    "概率提升起始", min_value=1, value=66, key="pity_start"
                )
            with col2:
                pity_limit = st.number_input(
                    "小保底抽数", min_value=1, value=80, key="pity_limit"
                )
            pity_boost = st.number_input(
                "每抽提升幅度",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                key="pity_boost",
            )
            pity_inherit = st.selectbox(
                "小保底继承",
                [
                    ("不继承", InheritPolicy.NO_INHERIT),
                    ("继承至下一期", InheritPolicy.INHERIT_TO_NEXT),
                    ("永久继承", InheritPolicy.ALWAYS_INHERIT),
                ],
                format_func=lambda x: x[0],
                key="pity_inherit",
            )
            pity_repeat = st.selectbox(
                "小保底重复",
                [
                    ("不可重复", RepeatPolicy.NO_REPEAT),
                    ("可重复", RepeatPolicy.ALWAYS_REPEAT),
                ],
                format_func=lambda x: x[0],
                key="pity_repeat",
            )
            pity_reset = st.selectbox(
                "小保底重置条件",
                [
                    ("抽到最高星时", ResetCondition.ON_HIGHEST_RARITY),
                    ("抽到UP时", ResetCondition.ON_MAIN),
                    ("无", ResetCondition.NONE),
                ],
                format_func=lambda x: x[0],
                key="pity_reset",
            )
            pity_inherit_policy = pity_inherit[1]
            pity_repeat_policy = pity_repeat[1]
            pity_reset_condition = pity_reset[1]

        st.markdown("**大保底设置**")
        has_definitive = st.checkbox("启用大保底", value=True, key="has_definitive")
        definitive_count = 120
        definitive_inherit_policy = InheritPolicy.NO_INHERIT
        definitive_reset_condition = ResetCondition.NONE
        if has_definitive:
            definitive_count = st.number_input(
                "大保底抽数", min_value=1, value=120, key="definitive_count"
            )
            definitive_inherit = st.selectbox(
                "大保底继承",
                [
                    ("不继承", InheritPolicy.NO_INHERIT),
                    ("继承至下一期", InheritPolicy.INHERIT_TO_NEXT),
                    ("永久继承", InheritPolicy.ALWAYS_INHERIT),
                ],
                format_func=lambda x: x[0],
                key="definitive_inherit",
            )
            definitive_reset = st.selectbox(
                "大保底重置条件",
                [
                    ("抽到UP时", ResetCondition.ON_MAIN),
                    ("抽到最高星时", ResetCondition.ON_HIGHEST_RARITY),
                    ("无", ResetCondition.NONE),
                ],
                format_func=lambda x: x[0],
                key="definitive_reset",
            )
            definitive_inherit_policy = definitive_inherit[1]
            definitive_reset_condition = definitive_reset[1]

        st.markdown("**潜能奖励设置**")
        has_potential = st.checkbox("启用潜能奖励", value=True, key="has_potential")
        potential_draw = 240
        if has_potential:
            potential_draw = st.number_input(
                "奖励间隔抽数", min_value=1, value=240, key="potential_draw"
            )

        st.markdown("**特殊抽奖励设置**")
        col1, col2 = st.columns(2)
        with col1:
            special_draw_reward_at = st.number_input(
                "特殊抽奖励抽数",
                min_value=0,
                value=30,
                key="special_draw_reward_at",
                help="累计抽数达到此值时获得特殊抽奖励(0表示禁用)",
            )
        with col2:
            special_draw_reward_count = st.number_input(
                "特殊抽奖励数量", min_value=0, value=10, key="special_draw_reward_count"
            )
        special_draw_repeat = st.checkbox(
            "可重复触发",
            value=False,
            key="special_draw_repeat",
            help="特殊抽奖励是否可以在同一卡池内多次触发",
        )

        st.markdown("**下期卡池抽奖励设置**")
        col1, col2 = st.columns(2)
        with col1:
            next_banner_draw_reward_at = st.number_input(
                "下期抽奖励抽数",
                min_value=0,
                value=60,
                key="next_banner_draw_reward_at",
                help="累计抽数达到此值时获得下期卡池抽奖励(0表示禁用)",
            )
        with col2:
            next_banner_draw_reward_count = st.number_input(
                "下期抽奖励数量",
                min_value=0,
                value=10,
                key="next_banner_draw_reward_count",
            )
        next_banner_draw_repeat = st.checkbox(
            "可重复触发",
            value=False,
            key="next_banner_draw_repeat",
            help="下期卡池抽奖励是否可以在同一卡池内多次触发",
        )

        if st.button("创建模板", key="create_template_btn"):
            # Validate probabilities sum to 1
            total_prob = prob_r4 + prob_r5 + prob_r6
            if abs(total_prob - 1.0) > 0.001:
                st.error(f"概率之和必须为1，当前为{total_prob:.3f}")
            else:
                new_template = BannerTemplate(
                    name=new_template_name,
                    rarities=[4, 5, 6],
                    default_distribution=[
                        RarityProbability(rarity=4, probability=prob_r4),
                        RarityProbability(rarity=5, probability=prob_r5),
                        RarityProbability(rarity=6, probability=prob_r6),
                    ],
                    main_probability=main_prob,
                    has_pity_draw=has_pity,
                    pity_draw_start=pity_start,
                    pity_draw_limit=pity_limit,
                    pity_rarity_boost_per_draw=pity_boost,
                    pity_draw_inherit_policy=pity_inherit_policy,
                    pity_draw_repeat_policy=pity_repeat_policy,
                    pity_reset_condition=pity_reset_condition,
                    has_definitive_draw=has_definitive,
                    definitive_draw_count=definitive_count,
                    definitive_draw_inherit_policy=definitive_inherit_policy,
                    definitive_reset_condition=definitive_reset_condition,
                    has_potential_reward=has_potential,
                    potential_reward_draw=potential_draw,
                    special_draw_reward_at=special_draw_reward_at,
                    special_draw_reward_count=special_draw_reward_count,
                    special_draw_repeat=special_draw_repeat,
                    next_banner_draw_reward_at=next_banner_draw_reward_at,
                    next_banner_draw_reward_count=next_banner_draw_reward_count,
                    next_banner_draw_repeat=next_banner_draw_repeat,
                )
                st.session_state.banner_templates.append(new_template)
                update_url()
                st.rerun()
