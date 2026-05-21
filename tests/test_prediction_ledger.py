"""Tests for prediction candidate ledger and calibration analytics."""

import json

import pytest

from prediction_ledger import CandidateRecord, PredictionLedger


def test_record_candidate_persists_to_json(tmp_path):
    path = tmp_path / "candidate_ledger.json"
    ledger = PredictionLedger(path=path)

    ledger.record_candidate(
        CandidateRecord(
            candidate_id="scan-1:1.23:101",
            market_id="1.23",
            selection_id=101,
            runner_name="Fast One",
            venue="Bendigo",
            lay_price=1.65,
            best_back=1.62,
            score=82,
            verdict="STRONG_LAY",
            decision="SELECTED",
            estimated_win_prob=0.61,
            estimated_win_prob_source="sp",
            wom_signal="LAY_HEAVY",
            model_signal="LAY",
            model_edge_pct=8.0,
            sp_far=1.5,
            edge_net=0.041,
        )
    )

    raw = json.loads(path.read_text())
    assert len(raw) == 1
    assert raw[0]["candidate_id"] == "scan-1:1.23:101"
    assert raw[0]["runner_name"] == "Fast One"
    assert raw[0]["decision"] == "SELECTED"


def test_calibration_by_score_decile_reports_accuracy_and_roi(tmp_path):
    ledger = PredictionLedger(path=tmp_path / "candidate_ledger.json")
    ledger.record_candidates(
        [
            CandidateRecord(
                candidate_id="a",
                market_id="1.1",
                selection_id=1,
                runner_name="A",
                score=86,
                verdict="STRONG_LAY",
                decision="SELECTED",
                estimated_win_prob=0.20,
                lay_price=2.0,
                stake=10.0,
                liability=10.0,
                won=False,
                profit=10.0,
            ),
            CandidateRecord(
                candidate_id="b",
                market_id="1.1",
                selection_id=2,
                runner_name="B",
                score=82,
                verdict="STRONG_LAY",
                decision="SELECTED",
                estimated_win_prob=0.30,
                lay_price=2.0,
                stake=10.0,
                liability=10.0,
                won=True,
                profit=-10.0,
            ),
        ]
    )

    rows = ledger.calibration_by_score_decile()
    high = rows["80-89"]

    assert high["candidates"] == 2
    assert high["settled"] == 2
    assert high["lay_win_rate"] == pytest.approx(50.0, abs=0.01)
    assert high["avg_predicted_win_prob"] == pytest.approx(0.25, abs=0.001)
    assert high["roi"] == pytest.approx(0.0, abs=0.01)
    assert high["brier_score"] == pytest.approx(((0.20 - 0) ** 2 + (0.30 - 1) ** 2) / 2, abs=0.001)


def test_rejected_winner_analysis_surfaces_skipped_winners(tmp_path):
    ledger = PredictionLedger(path=tmp_path / "candidate_ledger.json")
    ledger.record_candidates(
        [
            CandidateRecord(
                candidate_id="skipped-winner",
                market_id="1.1",
                selection_id=1,
                runner_name="Skipped Winner",
                score=28,
                verdict="SKIP",
                decision="SKIPPED",
                estimated_win_prob=0.40,
                won=True,
                profit=0.0,
            ),
            CandidateRecord(
                candidate_id="selected-loser",
                market_id="1.1",
                selection_id=2,
                runner_name="Selected Loser",
                score=77,
                verdict="LAY",
                decision="SELECTED",
                estimated_win_prob=0.20,
                won=False,
                profit=8.0,
            ),
        ]
    )

    report = ledger.rejected_winner_analysis()

    assert report["skipped_winners"] == 1
    assert report["examples"][0]["candidate_id"] == "skipped-winner"


def test_attach_bet_and_update_outcome_by_bet_id(tmp_path):
    ledger = PredictionLedger(path=tmp_path / "candidate_ledger.json")
    ledger.record_candidate(
        CandidateRecord(
            candidate_id="candidate-1",
            market_id="1.1",
            selection_id=1,
            runner_name="Linked Runner",
            score=81,
            verdict="STRONG_LAY",
            decision="SELECTED",
            estimated_win_prob=0.22,
            lay_price=2.0,
        )
    )

    assert ledger.attach_bet("candidate-1", bet_id="paper-123", stake=10.0, liability=10.0) is True
    assert ledger.update_outcome_by_bet_id(
        "paper-123",
        won=False,
        profit=10.0,
        bsp_actual=1.8,
        result_source="test",
    ) is True

    record = ledger.all_records()[0]
    assert record.bet_id == "paper-123"
    assert record.stake == 10.0
    assert record.liability == 10.0
    assert record.won is False
    assert record.profit == 10.0
    assert record.bsp_actual == 1.8
