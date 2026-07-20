from __future__ import annotations

import unittest

import pandas as pd

from scripts.v3a.scan_termination_cashflows import _parse_cash, _parse_date

from alpha_etf.data.expanded_universe_audit import (
    apply_index_metadata_events,
    apply_variety_aliases,
    build_eligible_variety_counts,
    build_identity_intervals,
    build_product_master,
    build_representative_lifecycle,
    classify_product,
    index_change_report,
)


def _fund_basic() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "510001.SH",
                "name": "最早ETF",
                "fund_type": "股票型",
                "invest_type": "被动指数型",
                "list_date": "20100104",
                "delist_date": "20151231",
                "status": "D",
                "benchmark": "测试指数收益率×100%",
            },
            {
                "ts_code": "510002.SH",
                "name": "接续ETF",
                "fund_type": "股票型",
                "invest_type": "被动指数型",
                "list_date": "20140102",
                "delist_date": None,
                "status": "L",
                "benchmark": "测试指数×100%",
            },
            {
                "ts_code": "510003.SH",
                "name": "跨境ETF(QDII)",
                "fund_type": "股票型",
                "invest_type": "被动指数型",
                "list_date": "20120103",
                "delist_date": None,
                "status": "L",
                "benchmark": "纳斯达克100指数×100%",
            },
            {
                "ts_code": "510004.SH",
                "name": "增强ETF",
                "fund_type": "股票型",
                "invest_type": "增强指数型",
                "list_date": "20120103",
                "delist_date": None,
                "status": "L",
                "benchmark": "测试指数×100%",
            },
        ]
    )


def _etf_basic() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "510001.SH",
                "index_code": None,
                "index_name": None,
                "etf_type": "纯境内",
                "list_status": "D",
                "exchange": "SH",
                "extname": "最早ETF",
            },
            {
                "ts_code": "510002.SH",
                "index_code": "000001.SH",
                "index_name": "测试指数",
                "etf_type": "纯境内",
                "list_status": "L",
                "exchange": "SH",
                "extname": "接续ETF",
            },
            {
                "ts_code": "510003.SH",
                "index_code": "NDX.GI",
                "index_name": "纳斯达克100",
                "etf_type": "QDII",
                "list_status": "L",
                "exchange": "SH",
                "extname": "跨境ETF",
            },
            {
                "ts_code": "510004.SH",
                "index_code": "000001.SH",
                "index_name": "测试指数",
                "etf_type": "纯境内",
                "list_status": "L",
                "exchange": "SH",
                "extname": "增强ETF",
            },
        ]
    )


def _empty_overrides() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ts_code",
            "index_code",
            "index_name",
            "effective_from",
            "effective_to",
            "evidence_url",
            "evidence_type",
            "change_type",
            "notes",
        ]
    )


class ExpandedUniverseAuditTests(unittest.TestCase):
    def test_terminal_cashflow_parser_handles_common_whitespace_and_wording(self) -> None:
        text = "清算资金发放 日为 2022 年 12 月 19 日；本次每份 基金份额可获分配清算资金为人民币 2.05404 元。"
        self.assertEqual(_parse_date(text), "2022-12-19")
        self.assertEqual(_parse_cash(text), 2.05404)

    def test_scope_is_mechanical_and_keeps_cross_border_separate(self) -> None:
        domestic = pd.Series(
            {
                "fund_type": "股票型",
                "invest_type": "被动指数型",
                "etf_type": "纯境内",
                "name": "测试ETF",
                "benchmark": "测试指数",
                "list_status": "L",
            }
        )
        cross_border = domestic.copy()
        cross_border["etf_type"] = "QDII"
        enhanced = domestic.copy()
        enhanced["invest_type"] = "增强指数型"
        self.assertEqual(classify_product(domestic)[0], "include_domestic_passive_equity")
        self.assertEqual(classify_product(cross_border)[0], "separate_cross_border")
        self.assertEqual(classify_product(enhanced)[0], "exclude_non_passive")

    def test_feeder_and_lof_are_not_exchange_etfs(self) -> None:
        product = pd.Series(
            {
                "fund_type": "股票型",
                "invest_type": "被动指数型",
                "etf_type": pd.NA,
                "list_status": pd.NA,
                "name": "沪深300ETF联接(LOF)-A",
                "type": "股票型",
                "benchmark": "沪深300指数×95%",
            }
        )
        self.assertEqual(classify_product(product)[0], "exclude_non_etf_product")

    def test_ended_product_inherits_only_a_reviewable_unique_group_code(self) -> None:
        master = build_product_master(_fund_basic(), _etf_basic())
        ended = master.set_index("ts_code").loc["510001.SH"]
        self.assertEqual(ended["resolved_index_code"], "000001.SH")
        self.assertEqual(
            ended["identity_resolution"], "same_benchmark_unique_current_index"
        )
        self.assertEqual(ended["history_identity_status"], "current_identity_requires_history_review")

    def test_ended_product_can_use_unique_index_master_name_as_reviewable_evidence(self) -> None:
        funds = _fund_basic().copy()
        funds.loc[funds["ts_code"] == "510001.SH", "benchmark"] = "历史独有指数×100%"
        index_basic = pd.DataFrame(
            [{"ts_code": "399999.SZ", "name": "历史独有", "market": "SZSE"}]
        )
        master = build_product_master(funds, _etf_basic(), index_basic)
        ended = master.set_index("ts_code").loc["510001.SH"]
        self.assertEqual(ended["resolved_index_code"], "399999.SZ")
        self.assertEqual(ended["identity_resolution"], "index_basic_unique_name")

    def test_representative_changes_only_after_earliest_product_ends(self) -> None:
        master = build_product_master(_fund_basic(), _etf_basic())
        intervals, exceptions = build_identity_intervals(
            master, _empty_overrides(), as_of="20261231"
        )
        lifecycle = build_representative_lifecycle(intervals)
        rows = lifecycle[lifecycle["index_code"] == "000001.SH"]
        self.assertEqual(rows["representative_ts_code"].tolist(), ["510001.SH", "510002.SH"])
        self.assertEqual(rows.iloc[0]["representative_to"], pd.Timestamp("2015-12-31"))
        self.assertEqual(rows.iloc[1]["representative_from"], pd.Timestamp("2016-01-01"))
        self.assertTrue(exceptions["exception_code"].eq("history_identity_unverified").any())

    def test_existing_representative_is_not_replaced_when_older_product_changes_into_index(self) -> None:
        intervals = pd.DataFrame(
            [
                {
                    "schema_version": "test",
                    "ts_code": "B.SH",
                    "symbol": "B",
                    "product_name": "B",
                    "product_list_date": pd.Timestamp("2015-01-01"),
                    "index_code": "X",
                    "variety_id": "X",
                    "index_name": "X",
                    "effective_from": pd.Timestamp("2015-01-01"),
                    "effective_to": pd.Timestamp("2025-12-31"),
                    "warmup_from": pd.Timestamp("2015-01-01"),
                    "identity_source": "test",
                    "identity_confidence": "test",
                    "evidence": "test",
                },
                {
                    "schema_version": "test",
                    "ts_code": "A.SH",
                    "symbol": "A",
                    "product_name": "A",
                    "product_list_date": pd.Timestamp("2010-01-01"),
                    "index_code": "X",
                    "variety_id": "X",
                    "index_name": "X",
                    "effective_from": pd.Timestamp("2020-01-01"),
                    "effective_to": pd.Timestamp("2025-12-31"),
                    "warmup_from": pd.Timestamp("2020-01-01"),
                    "identity_source": "test",
                    "identity_confidence": "test",
                    "evidence": "test",
                },
            ]
        )
        lifecycle = build_representative_lifecycle(intervals)
        self.assertEqual(lifecycle["representative_ts_code"].tolist(), ["B.SH"])

    def test_official_overrides_split_index_identity_and_report_change(self) -> None:
        master = build_product_master(_fund_basic(), _etf_basic())
        overrides = _empty_overrides()
        overrides.loc[0] = [
            "510002.SH",
            "000001.SH",
            "旧指数",
            "20140102",
            "20191231",
            "https://example.test/old",
            "official_fund_contract",
            "initial_identity",
            "",
        ]
        overrides.loc[1] = [
            "510002.SH",
            "000002.SH",
            "新指数",
            "20200101",
            "",
            "https://example.test/change",
            "official_fund_announcement",
            "index_change",
            "",
        ]
        intervals, _ = build_identity_intervals(master, overrides, as_of="20261231")
        product = intervals[intervals["ts_code"] == "510002.SH"]
        self.assertEqual(product["index_code"].tolist(), ["000001.SH", "000002.SH"])
        self.assertTrue(
            product["identity_confidence"].eq("official_historical_interval").all()
        )
        changes = index_change_report(intervals)
        self.assertEqual(changes["ts_code"].unique().tolist(), ["510002.SH"])

    def test_overlapping_official_intervals_fail(self) -> None:
        master = build_product_master(_fund_basic(), _etf_basic())
        overrides = _empty_overrides()
        overrides.loc[0] = [
            "510002.SH",
            "000001.SH",
            "旧指数",
            "20140102",
            "20200131",
            "https://example.test/old",
            "fund_contract",
            "initial_identity",
            "",
        ]
        overrides.loc[1] = [
            "510002.SH",
            "000002.SH",
            "新指数",
            "20200101",
            "",
            "https://example.test/change",
            "fund_announcement",
            "index_change",
            "",
        ]
        with self.assertRaisesRegex(ValueError, "overlapping identity intervals"):
            build_identity_intervals(master, overrides, as_of="20261231")

    def test_eligibility_requires_41_observations_and_counts_varieties_once(self) -> None:
        master = build_product_master(_fund_basic(), _etf_basic())
        intervals, _ = build_identity_intervals(master, _empty_overrides(), as_of="20261231")
        lifecycle = build_representative_lifecycle(intervals)
        dates = pd.bdate_range("2010-01-04", periods=100)
        timeline, eligibility = build_eligible_variety_counts(
            intervals,
            lifecycle,
            {"510001": dates, "510002": dates},
        )
        self.assertEqual(timeline.iloc[0]["date"], dates[40])
        self.assertTrue(timeline["eligible_variety_count"].eq(1).all())
        first = eligibility[eligibility["representative_ts_code"] == "510001.SH"].iloc[0]
        self.assertEqual(first["warmup_date"], dates[40])

    def test_eligibility_timeline_keeps_zero_count_calendar_dates(self) -> None:
        master = build_product_master(_fund_basic(), _etf_basic())
        intervals, _ = build_identity_intervals(master, _empty_overrides(), as_of="20261231")
        lifecycle = build_representative_lifecycle(intervals)
        dates = pd.bdate_range("2010-01-04", periods=50)
        calendar = dates
        valid = dates.delete(45)
        timeline, _ = build_eligible_variety_counts(
            intervals,
            lifecycle,
            {"510001": valid},
            calendar=pd.DatetimeIndex(calendar),
        )
        missing = timeline[timeline["date"].eq(dates[45])]
        self.assertEqual(missing.iloc[0]["eligible_variety_count"], 0)

    def test_code_adjustment_keeps_one_variety_and_does_not_reset_warmup(self) -> None:
        master = build_product_master(_fund_basic(), _etf_basic())
        intervals, _ = build_identity_intervals(master, _empty_overrides(), as_of="20261231")
        events = pd.DataFrame(
            [
                {
                    "ts_code": "510002.SH",
                    "event_date": "20200102",
                    "event_type": "index_code_adjustment",
                    "old_index_code": "OLD001.SH",
                    "new_index_code": "000001.SH",
                    "old_index_name": "测试旧名",
                    "new_index_name": "测试指数",
                    "canonical_variety_id": "000001.SH",
                    "evidence_url": "https://example.test/adjustment",
                    "requires_roc_reset": "false",
                    "notes": "same index",
                }
            ]
        )
        adjusted, _ = apply_index_metadata_events(intervals, events)
        product = adjusted[adjusted["ts_code"] == "510002.SH"]
        self.assertEqual(product["index_code"].tolist(), ["OLD001.SH", "000001.SH"])
        self.assertEqual(product["variety_id"].nunique(), 1)
        self.assertTrue(
            product["warmup_from"].eq(pd.Timestamp("2014-01-02")).all()
        )
        lifecycle = build_representative_lifecycle(adjusted)
        self.assertEqual(lifecycle["variety_id"].nunique(), 1)
        self.assertEqual(index_change_report(adjusted).shape[0], 0)

    def test_index_aliases_collapse_to_one_variety(self) -> None:
        master = build_product_master(_fund_basic(), _etf_basic())
        intervals, _ = build_identity_intervals(master, _empty_overrides(), as_of="20261231")
        second = intervals[intervals["ts_code"].eq("510002.SH")].copy()
        second["ts_code"] = "510099.SH"
        second["symbol"] = "510099"
        second["index_code"] = "ALIAS.SH"
        second["variety_id"] = "ALIAS.SH"
        combined = pd.concat([intervals, second], ignore_index=True)
        aliases = pd.DataFrame(
            [
                {
                    "alias_index_code": "ALIAS.SH",
                    "canonical_variety_id": "000001.SH",
                    "evidence_url": "https://example.test/alias",
                }
            ]
        )
        normalized = apply_variety_aliases(combined, aliases)
        rows = normalized[normalized["index_code"].isin(["000001.SH", "ALIAS.SH"])]
        self.assertEqual(rows["variety_id"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()