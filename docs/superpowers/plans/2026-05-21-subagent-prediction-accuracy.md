# Subagent Prediction Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve prediction accuracy by making every subagent use the same calibrated opportunity score, recording candidate-level feedback, and feeding BSP, movement, and historical calibration into ranking and staking.

**Architecture:** Add two focused modules: `opportunity_scoring.py` owns signal normalization, score composition, and calibrated win probability; `prediction_ledger.py` owns candidate-level evidence and calibration reports. `sub_agents.py`, `paper_autobet.py`, and `flumine_engine.py` consume the shared scorer instead of duplicating core gates.

**Tech Stack:** Python 3.10+, JSON persistence, existing Betfair client/cache/rating APIs, pytest.

---

### Task 1: Shared Opportunity Scorer

**Files:**
- Create: `opportunity_scoring.py`
- Test: `tests/test_opportunity_scoring.py`

- [ ] **Step 1: Write failing tests**
  - Verify `score_lay_opportunity()` rewards BSP value, model lay edge, lay-heavy WOM, tight spreads, positive context, and market drift.
  - Verify configured `min_profit_ratio` and `min_edge_pct` change verdicts.
  - Verify `estimated_win_prob` prefers SP-derived probability over model price.

- [ ] **Step 2: Implement scorer**
  - Define `ScoreConfig`, `score_lay_opportunity()`, and helper functions for safe float parsing and score bounds.
  - Return a dict with `score`, `verdict`, `estimated_win_prob`, `estimated_win_prob_source`, `components`, `issues`, and normalized signal fields.

### Task 2: Prediction Ledger And Calibration

**Files:**
- Create: `prediction_ledger.py`
- Test: `tests/test_prediction_ledger.py`

- [ ] **Step 1: Write failing tests**
  - Verify candidate records persist to JSON.
  - Verify calibration by score decile reports candidate count, settled count, lay win rate, average predicted win probability, ROI, and Brier score.
  - Verify rejected-winner analysis identifies skipped candidates that later won.

- [ ] **Step 2: Implement ledger**
  - Define `CandidateRecord` and `PredictionLedger`.
  - Add `record_candidate()`, `record_candidates()`, `update_outcome()`, `calibration_by_score_decile()`, `closing_line_value_stats()`, and `rejected_winner_analysis()`.
  - Add `get_prediction_ledger()` singleton helper.

### Task 3: Analyst Integration

**Files:**
- Modify: `sub_agents.py`
- Test: `tests/test_sub_agents_prediction.py`

- [ ] **Step 1: Write failing tests**
  - Use fake client/cache/rating objects to verify `VenueAnalystAgent` includes BSP edge in top opportunities.
  - Verify candidates are written to `PredictionLedger`.
  - Verify `min_profit_ratio` and `min_edge_pct` from `OrchestratorAgent.run_venue_session()` reach the analyst.

- [ ] **Step 2: Implement integration**
  - Pass `ScoreConfig` into `VenueAnalystAgent`.
  - Fetch BSP predictions per market through `SharedCache.bsp_predictions()`.
  - Score every active runner through `score_lay_opportunity()`.
  - Record each scanned runner to `PredictionLedger`.

### Task 4: Risk And Execution Alignment

**Files:**
- Modify: `sub_agents.py`
- Modify: `flumine_engine.py`
- Test: `tests/test_sub_agents_prediction.py`

- [ ] **Step 1: Write failing tests**
  - Verify `RiskManagerAgent` allocates by `(market_id, selection_id)` so multiple candidates in one market cannot overwrite one another.
  - Verify Kelly sizing uses `estimated_win_prob` from the scorer.
  - Verify polling execution can place the selected runner from an allocation key.

- [ ] **Step 2: Implement integration**
  - Store allocations by a stable opportunity key.
  - Carry `estimated_win_prob`, signal context, and score into allocation records.
  - Pass analytic context to `place_lay_bet()` in polling mode for memory/ledger continuity.
  - Make Flumine’s venue strategy use `score_lay_opportunity()` for consistent gates.

### Task 5: Paper Automation And Reporting

**Files:**
- Modify: `paper_autobet.py`
- Modify: `betting_memory.py`
- Modify: `mcp_server.py`
- Test: `tests/test_paper_autobet.py`
- Test: `tests/test_mcp_tools.py`

- [ ] **Step 1: Write failing tests**
  - Verify daily paper candidates use the shared scorer.
  - Verify performance summary includes prediction ledger calibration.
  - Verify `get_strategy_insights` exposes prediction calibration when records exist.

- [ ] **Step 2: Implement integration**
  - Replace paper-autobet private scoring with `score_lay_opportunity()`.
  - Add prediction calibration summaries to MCP performance and strategy insight tools.

### Task 6: Verification

**Files:**
- Existing tests and new tests.

- [ ] **Step 1: Run focused tests**
  - `pytest tests/test_opportunity_scoring.py tests/test_prediction_ledger.py tests/test_sub_agents_prediction.py -v`

- [ ] **Step 2: Run full suite**
  - `pytest tests/ -v`

- [ ] **Step 3: Review diff**
  - `git diff --stat`
  - `git diff --check`
