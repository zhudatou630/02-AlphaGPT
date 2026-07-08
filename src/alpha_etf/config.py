"""Phase 1 configuration for the multi-ETF research system."""

from __future__ import annotations

from dataclasses import dataclass


START_DATE = "2010-01-01"

RAW_FEATURES = ("open", "high", "low", "close", "volume", "amount")

PRICE_FEATURES = ("open", "high", "low", "close")

FLOW_FEATURES = ("volume", "amount")


@dataclass(frozen=True)
class ETF:
    symbol: str
    name: str
    category: str
    bucket: str

    @property
    def tdx_symbol(self) -> str:
        if self.symbol.startswith(("5", "6")):
            return f"sh{self.symbol}"
        if self.symbol.startswith(("0", "1", "3")):
            return f"sz{self.symbol}"
        raise ValueError(f"Cannot infer TDX exchange prefix for {self.symbol}")


ETF_UNIVERSE = (
    ETF("510050", "上证50ETF", "broad", "large_value"),
    ETF("510300", "沪深300ETF", "broad", "csi300"),
    ETF("510500", "中证500ETF", "broad", "csi500"),
    ETF("512100", "中证1000ETF", "broad", "csi1000"),
    ETF("159915", "创业板ETF", "broad", "chinext"),
    ETF("588000", "科创50ETF", "broad", "star50"),
    ETF("510880", "红利ETF", "broad", "dividend"),
    ETF("512800", "银行ETF", "industry", "bank"),
    ETF("512880", "证券ETF", "industry", "broker"),
    ETF("512760", "半导体ETF", "industry", "semiconductor"),
    ETF("512660", "军工ETF", "industry", "defense"),
    ETF("512010", "医药ETF", "industry", "pharma"),
    ETF("159928", "消费ETF", "industry", "consumer"),
    ETF("512400", "有色金属ETF", "industry", "nonferrous"),
    ETF("515220", "煤炭ETF", "industry", "coal"),
    ETF("515790", "光伏ETF", "industry", "solar"),
    ETF("515030", "新能源车ETF", "industry", "new_energy_vehicle"),
)


VOCAB_V1 = {
    "fields": ("open", "high", "low", "close", "volume", "amount"),
    "returns": ("RET_1", "RET_3", "RET_5", "RET_10", "RET_20"),
    "trend": ("MA_5", "MA_10", "MA_20", "EMA_10", "EMA_20"),
    "volatility": ("STD_10", "STD_20", "ZSCORE_20", "TS_RANK_20"),
    "volume_price": (
        "VOLUME_ZSCORE_20",
        "AMOUNT_ZSCORE_20",
        "TURNOVER_PROXY_20",
    ),
    "arithmetic": ("ADD", "SUB", "MUL", "DIV", "ABS", "SIGN", "CLIP"),
    "conditional": ("GT", "LT", "GATE"),
    "temporal": ("DELAY_1", "DELAY_5", "DECAY_5", "DECAY_10"),
}
