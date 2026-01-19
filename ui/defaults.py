"""Default data creation functions."""

from banner import Banner, EndFieldBannerTemplate, Operator
from strategy import (
    BannerIndexCondition,
    ContinueAction,
    DrawBehavior,
    DrawCountCondition,
    DrawStrategy,
    GotMainCondition,
    ResourceThresholdCondition,
    StopAction,
    StrategyRule,
)


def create_default_operators() -> list[Operator]:
    """Create default operator list."""
    operators = [
        Operator(name="别礼", rarity=6),  # type: ignore
        Operator(name="黎风", rarity=6),  # type: ignore
        Operator(name="骏卫", rarity=6),  # type: ignore
        Operator(name="埃尔黛拉", rarity=6),  # type: ignore
        Operator(name="余烬", rarity=6),  # type: ignore
        Operator(name="佩丽卡", rarity=5),  # type: ignore
        Operator(name="狼卫", rarity=5),  # type: ignore
        Operator(name="赛希", rarity=5),  # type: ignore
        Operator(name="艾维文娜", rarity=5),  # type: ignore
        Operator(name="陈千语", rarity=5),  # type: ignore
        Operator(name="大潘", rarity=5),  # type: ignore
        Operator(name="阿列什", rarity=5),  # type: ignore
        Operator(name="弧光", rarity=5),  # type: ignore
        Operator(name="昼雪", rarity=5),  # type: ignore
        Operator(name="秋栗", rarity=4),  # type: ignore
        Operator(name="安塔尔", rarity=4),  # type: ignore
        Operator(name="卡契尔", rarity=4),  # type: ignore
        Operator(name="埃特拉", rarity=4),  # type: ignore
        Operator(name="萤石", rarity=4),  # type: ignore
    ]
    return operators


def create_default_banners() -> list[Banner]:
    """Create default banner list using EndField template."""
    # Get default operators grouped by rarity
    default_ops = create_default_operators()
    base_operators: dict[int, list[Operator]] = {}
    for op in default_ops:
        if op.rarity not in base_operators:
            base_operators[op.rarity] = []
        base_operators[op.rarity].append(op)

    # Banner-specific 6-star operators
    Laevatain = Operator(name="莱万汀", rarity=6)  # type: ignore
    Gilberta = Operator(name="洁尔佩塔", rarity=6)  # type: ignore
    Yvonne = Operator(name="伊冯", rarity=6)  # type: ignore

    banners = []

    # Banner 1: Laevatain main
    banner1_ops = {
        6: [Laevatain, Gilberta, Yvonne] + base_operators.get(6, []),
        5: base_operators.get(5, []).copy(),
        4: base_operators.get(4, []).copy(),
    }
    banners.append(
        Banner(
            name="熔火灼痕",
            operators=banner1_ops,
            main_operator=Laevatain,
            template=EndFieldBannerTemplate.model_copy(deep=True),
        )  # type: ignore
    )

    # Banner 2: Gilberta main
    banner2_ops = {
        6: [Laevatain, Gilberta, Yvonne] + base_operators.get(6, []),
        5: base_operators.get(5, []).copy(),
        4: base_operators.get(4, []).copy(),
    }
    banners.append(
        Banner(
            name="轻飘飘的信使",
            operators=banner2_ops,
            main_operator=Gilberta,
            template=EndFieldBannerTemplate.model_copy(deep=True),
        )  # type: ignore
    )

    # Banner 3: Yvonne main
    banner3_ops = {
        6: [Laevatain, Gilberta, Yvonne] + base_operators.get(6, []),
        5: base_operators.get(5, []).copy(),
        4: base_operators.get(4, []).copy(),
    }
    banners.append(
        Banner(
            name="热烈色彩",
            operators=banner3_ops,
            main_operator=Yvonne,
            template=EndFieldBannerTemplate.model_copy(deep=True),
        )  # type: ignore
    )

    return banners


def create_default_strategy() -> DrawStrategy:
    """Create default strategy - pay to get UP."""
    return DrawStrategy(
        name="默认策略(氪金抽到UP)",
        behavior=DrawBehavior(pay=True),
        rules=[],
        default_action=ContinueAction(stop_on_main=True),
    )


def create_default_strategies() -> list[DrawStrategy]:
    """Create all default strategies."""
    return [
        # Default: pay to get UP
        create_default_strategy(),
        # Pay for exactly 30 draws
        DrawStrategy(
            name="氪金抽30抽",
            behavior=DrawBehavior(pay=True),
            rules=[],
            default_action=ContinueAction(
                min_draws_per_banner=30, max_draws_per_banner=30
            ),
        ),
        # Pay for exactly 60 draws
        DrawStrategy(
            name="氪金抽60抽",
            behavior=DrawBehavior(pay=True),
            rules=[],
            default_action=ContinueAction(
                min_draws_per_banner=60, max_draws_per_banner=60
            ),
        ),
        # Only draw if 120+ draws available, no pay, stop on UP
        DrawStrategy(
            name="没有120抽不抽(不氪金)",
            behavior=DrawBehavior(pay=False),
            rules=[
                # If available draws < 120 at entry, skip this banner
                StrategyRule(
                    conditions=[
                        ResourceThresholdCondition(
                            max_normal_draws=119, check_once=True
                        )
                    ],
                    action=StopAction(),
                    priority=100,
                ),
            ],
            default_action=ContinueAction(stop_on_main=True),
        ),
        # If < 120 draws: draw 30 (stop early if got 6-star); if >= 120: draw until UP
        DrawStrategy(
            name="没有120抽就抽30拿特殊十连(不氪金)",
            behavior=DrawBehavior(pay=False),
            rules=[
                # If available draws < 120 at entry, draw up to 30 and stop on highest rarity
                StrategyRule(
                    conditions=[
                        ResourceThresholdCondition(
                            max_normal_draws=119, check_once=True
                        )
                    ],
                    action=ContinueAction(
                        max_draws_per_banner=30,
                        stop_on_highest_rarity=True,
                    ),
                    priority=100,
                ),
            ],
            # Default: if >= 120 draws, draw until UP
            default_action=ContinueAction(stop_on_main=True),
        ),
        *create_decision_tree_strategies(),
        # 3-banner cycle: draw 60 for first two, draw to UP on third (ensure min 60)
        # Banner 3: draw until UP, if UP before 60 continue to 60
        DrawStrategy(
            name="每3池前2池抽60第3池抽保底(不氪金)",
            behavior=DrawBehavior(pay=False),
            rules=[
                # Banner 1 in every 3: draw exactly 60
                StrategyRule(
                    conditions=[BannerIndexCondition(every_n=3, start_at=1)],
                    action=ContinueAction(
                        min_draws_per_banner=60, max_draws_per_banner=60
                    ),
                    priority=100,
                ),
                # Banner 2 in every 3: draw exactly 60
                StrategyRule(
                    conditions=[BannerIndexCondition(every_n=3, start_at=2)],
                    action=ContinueAction(
                        min_draws_per_banner=60, max_draws_per_banner=60
                    ),
                    priority=100,
                ),
                # Banner 3: if got UP and draws < 60, continue to 60
                StrategyRule(
                    conditions=[
                        BannerIndexCondition(every_n=3, start_at=3),
                        GotMainCondition(value=True),
                        DrawCountCondition(max_draws=59),
                    ],
                    action=ContinueAction(max_draws_per_banner=60),
                    priority=90,
                ),
            ],
            # Banner 3 (default): draw until UP
            default_action=ContinueAction(stop_on_main=True),
        ),
    ]


def create_decision_tree_strategies() -> list[DrawStrategy]:
    """Create simple utility strategies."""
    return [
        DrawStrategy(
            name="抽到保底",
            behavior=DrawBehavior(pay=False),
            rules=[],
            default_action=ContinueAction(stop_on_main=True),
        ),
        DrawStrategy(
            name="氪金抽到保底",
            behavior=DrawBehavior(pay=True),
            rules=[],
            default_action=ContinueAction(stop_on_main=True),
        ),
    ]
