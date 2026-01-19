"""Default data creation functions."""

from banner import Banner, EndFieldBannerTemplate, Operator
from gacha import DrawStrategy


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
    return DrawStrategy(name="默认策略(氪金抽到UP)", pay=True)
