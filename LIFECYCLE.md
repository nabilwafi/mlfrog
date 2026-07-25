# mlfrog — Platform Lifecycle Design Document

| Field | Value |
|---|---|
| Document type | Software Design Document (Lifecycle) |
| Classification | Internal — Quant Platform |
| Status | Design only — **no implementation** |
| Prerequisites | [ARCHITECTURE.md](ARCHITECTURE.md), [DOMAIN_MODEL.md](DOMAIN_MODEL.md) |
| Modes in scope | Offline Training · Offline Backtesting · Paper Trading · Live Trading |

---

## 0. Executive summary

The platform is one domain spine with four **runtime modes**. Modes differ by **adapters** (data source, clock, execution venue) and by **which lifecycles are active**. They do not fork business logic.

| Mode | Active lifecycles | Inactive / offline-only |
|---|---|---|
| Offline Training | Data, Feature, Label→Dataset, Model, Analytics | Decision→Execution hot path |
| Offline Backtesting | Data→…→Portfolio + Analytics (sim execution) | Live broker; streaming |
| Paper Trading | Full hot path; paper execution | Real `order_send`; Label/Train |
| Live Trading | Full hot path; venue execution | Label/Train on critical path |

**Invariant:** `Signal → Decision → TradePlan → Order → Position → Portfolio` is identical in backtest, paper, and live.

---

## 1. Cross-cutting platform contracts

### 1.1 Event bus (logical)

All lifecycles emit **domain events**. Monitoring and Analytics subscribe. Hot-path engines must not block on Analytics.

| Event family | Examples |
|---|---|
| `data.*` | `BarsLoaded`, `BarClosed`, `DataQualityFailed` |
| `feature.*` | `FeatureSetBuilt`, `FeatureValidationFailed` |
| `model.*` | `ModelFitted`, `ModelRegistered`, `InferenceCompleted`, `CalibrationApplied` |
| `signal.*` | `SignalEmitted`, `SignalRejected` |
| `decision.*` | `DecisionMade`, `DecisionSkipped` |
| `risk.*` | `TradePlanCreated`, `RiskRejected` |
| `execution.*` | `OrderSubmitted`, `OrderFilled`, `OrderRejected`, `ReconcileDrift` |
| `position.*` | `PositionOpened`, `PositionUpdated`, `PositionClosed` |
| `portfolio.*` | `PortfolioMarked`, `KillSwitchArmed`, `DrawdownBreach` |
| `analytics.*` | `ReportGenerated`, `WalkForwardCompleted` |

### 1.2 Correlation & identity

Every hot-path chain carries:

- `run_id` — session / backtest / paper / live process
- `bar_as_of` — closed-bar timestamp driving the tick of the loop
- `trace_id` — per bar (or per decision) correlation id
- `strategy_id`, `instrument`, `model_key` as applicable

### 1.3 Logging standard (all lifecycles)

| Level | Use |
|---|---|
| DEBUG | Intermediate matrices sizes, filter scores (sampled) |
| INFO | Lifecycle transitions, acceptances |
| WARN | Soft rejects, retries, data gaps |
| ERROR | Hard failures, invariant breaks |
| CRITICAL | Kill-switch, accounting break, venue disconnect mid-order |

Structured fields (mandatory on hot path): `run_id`, `trace_id`, `mode`, `as_of`, `stage`, `outcome`.

### 1.4 Monitoring standard

| Class | Examples | Sink |
|---|---|---|
| Health | heartbeat, last_bar_age, venue_connected | Monitoring |
| Throughput | bars/sec, inferences/sec, decisions/sec | Monitoring |
| Quality | NaN rate, reject rates by reason | Monitoring |
| Risk | equity, DD, daily PnL, exposure | Monitoring |
| Execution | submit latency, fill rate, slippage | Monitoring |
| Model | p_raw drift, calibration ECE (batch) | Analytics (+ alert hooks) |

### 1.5 Retry taxonomy

| Class | Policy |
|---|---|
| **Idempotent read** (history fetch) | Exponential backoff, capped; fail closed to `DataQualityFailed` |
| **Inference / pure policy** | **No retry** — deterministic; fix input or skip bar |
| **Venue submit** | Bounded retry with idempotency key = `plan_id`; never double-submit without reconcile |
| **Reconcile** | Periodic + on reconnect; authoritative venue state wins after conflict policy |
| **Analytics jobs** | Retryable offline; never gate live trading |

---

## 2. Lifecycle specifications

### 2.1 Data lifecycle

| Aspect | Specification |
|---|---|
| **Purpose** | Produce valid `MarketData` / `Candle` snapshots for all modes |
| **Owner** | `market_data` via `IMarketDataProvider` |
| **Input** | Instrument, timeframe(s), range or lookback; live: venue stream |
| **Output** | Immutable `MarketData` snapshot; events `BarsLoaded` / `BarClosed` |
| **Events** | `BarsLoaded`, `BarClosed`, `GapDetected`, `DataQualityFailed`, `FeedDisconnected`, `FeedReconnected` |
| **Validation** | OHLC invariants; monotonic timestamps; closed-bar gate for signal path; max gap policy; symbol resolution |
| **Failure handling** | Corrupt chunk → reject snapshot, do not partial-feed Features; live disconnect → pause Decision loop, arm monitoring alert |
| **Retry policy** | History: backoff retry (3–5). Live stream: reconnect with resume-from-last-close. No silent gap fill without `gaps` metadata |
| **Logging** | INFO load ranges; WARN gaps; ERROR schema/OHLC failures |
| **Monitoring** | `last_bar_age`, gap count, reconnect count, bars lag vs exchange |

**Mode notes**

- Train/Backtest: bulk historical load, deterministic clock.
- Paper/Live: closed-bar only for Feature→Signal; ticks may update quotes for spread filters only.

---

### 2.2 Feature lifecycle

| Aspect | Specification |
|---|---|
| **Purpose** | Causal `FeatureSet` for a `recipe_id` |
| **Owner** | `features` / `IFeatureEngine` |
| **Input** | `Mapping[Timeframe, MarketData]`, `recipe_id`, `as_of` |
| **Output** | Immutable `FeatureSet` (`causal=True`) |
| **Events** | `FeatureSetBuilt`, `FeatureWarmupInsufficient`, `FeatureValidationFailed` |
| **Validation** | Schema match to recipe; no NaN on inference row (default deny); `as_of` = last closed bar; leakage checks offline in Analytics |
| **Failure handling** | Warmup short → skip bar (`FeatureWarmupInsufficient`), no Signal. Schema drift → hard fail train; skip+alert live |
| **Retry policy** | None (pure). Recompute only if new `MarketData` arrives |
| **Logging** | INFO recipe_id + n_rows; WARN warmup; ERROR NaN/schema |
| **Monitoring** | Build latency, NaN rate, skip rate |

---

### 2.3 Model lifecycle

| Aspect | Specification |
|---|---|
| **Purpose** | Train, register, load, infer, calibrate models identified by `ModelKey` |
| **Owner** | `models` (`IModelTrainer`, `IModelPredictor`, `ICalibrationEngine`, Registry) |
| **Input (train)** | `Dataset`, `ModelKey`, train config |
| **Input (infer)** | `FeatureSet`, `ModelKey` |
| **Output (train)** | `ModelArtifact` (+ optional `CalibratorArtifact`) registered |
| **Output (infer)** | `Signal` (raw then optionally calibrated) |
| **Events** | `ModelFitted`, `ModelRegistered`, `ModelPromoted`, `ModelRetired`, `InferenceCompleted`, `CalibrationApplied`, `ModelLoadFailed` |
| **Validation** | Train: Dataset invariants, sealed/split rules. Infer: FeatureSet schema == artifact schema; model status `promoted` for live |
| **Failure handling** | Train fail → no register. Infer schema mismatch → no Signal, alert. Missing promoted model → kill paper/live loop start |
| **Retry policy** | Train: job-level retry. Infer: no retry. Artifact load: one retry then fail closed |
| **Logging** | INFO fit metrics summary; INFO model_key on infer; ERROR load/schema |
| **Monitoring** | Train job status; infer latency; prediction distribution drift (batch) |

**Promotion state machine (model):** `fitted → validated → staged → promoted → retired` (see §5.3).

---

### 2.4 Signal lifecycle

| Aspect | Specification |
|---|---|
| **Purpose** | Carry model belief into Decision without ML internals |
| **Owner** | Produced in `models`; consumed in `decision` |
| **Input** | Calibrated/raw predictor output fields |
| **Output** | Immutable `Signal` |
| **Events** | `SignalEmitted`, `SignalRejected` (failed validation before Decision) |
| **Validation** | `p_raw`/`p_calibrated` ∈ [0,1]; direction allowed by strategy; `as_of` closed; `ModelKey` consistency |
| **Failure handling** | Invalid Signal discarded; Decision not called; WARN + metric |
| **Retry policy** | None |
| **Logging** | INFO (sampled) p and direction; DEBUG full Signal |
| **Monitoring** | Emit rate, reject rate, mean p, confidence tier histogram |

---

### 2.5 Decision lifecycle

| Aspect | Specification |
|---|---|
| **Purpose** | Policy: trade or skip with audit trail |
| **Owner** | `decision` / `IDecisionEngine` |
| **Input** | `Sequence[Signal]`, `DecisionContext` (session, spread, portfolio snapshot, open flags) |
| **Output** | Immutable `Decision` |
| **Events** | `DecisionMade` (action=trade), `DecisionSkipped` (reason code) |
| **Validation** | Context freshness (`snapshot.as_of` within skew budget); filter chain order deterministic |
| **Failure handling** | Context stale → skip (`skip_stale_context`). Engine exception → treat as skip + CRITICAL (fail closed) |
| **Retry policy** | None (pure policy) |
| **Logging** | INFO action + reason + filters; always log skips (reason cardinality bounded) |
| **Monitoring** | Trade vs skip ratio by reason; filter hit rates |

---

### 2.6 Risk lifecycle

| Aspect | Specification |
|---|---|
| **Purpose** | Size and constrain intent → `TradePlan` or `Reject` |
| **Owner** | `risk` / `IRiskEngine` |
| **Input** | `Decision`, `PortfolioSnapshot`, `RiskConfig` |
| **Output** | `TradePlan` \| `Reject` |
| **Events** | `TradePlanCreated`, `RiskRejected`, `RiskLimitBreached` |
| **Validation** | Qty step/min; SL/TP side; risk_amount; daily loss / DD / exposure / kill-switch |
| **Failure handling** | Any limit breach → `Reject` (not partial plan). Snapshot missing → Reject. Arithmetic error → Reject + CRITICAL |
| **Retry policy** | None |
| **Logging** | INFO plan summary; WARN rejects with code |
| **Monitoring** | Reject codes; planned risk$; utilization vs limits |

---

### 2.7 Execution lifecycle

| Aspect | Specification |
|---|---|
| **Purpose** | Turn `TradePlan` into venue reality (sim / paper / live) |
| **Owner** | `execution` / `IExecutionEngine` |
| **Input** | `TradePlan` |
| **Output** | `Order` + `OrderEvent`/`Fill` stream |
| **Events** | `OrderCreated`, `OrderSubmitted`, `OrderPartial`, `OrderFilled`, `OrderCanceled`, `OrderRejected`, `ReconcileDrift` |
| **Validation** | Plan still valid at send (spread cap); idempotency key; paper adapter must not call live `order_send` |
| **Failure handling** | Submit fail → retry class Venue; after budget → Order rejected → no Position. Disconnect → reconcile on reconnect before new submits |
| **Retry policy** | Bounded exponential backoff; **idempotent** by `plan_id`. Never retry fill application |
| **Logging** | INFO state transitions; ERROR rejects; CRITICAL drift |
| **Monitoring** | Submit latency, fill ratio, slippage, reject rate, open order age |

**Adapter matrix**

| Mode | Adapter behavior |
|---|---|
| Backtest | Immediate/synthetic fills from bar path; configurable slippage |
| Paper | Ledger fills; **tripwire** against live send |
| Live | MT5 (or other) with reconcile |

---

### 2.8 Position lifecycle

| Aspect | Specification |
|---|---|
| **Purpose** | Track per-strategy instrument exposure |
| **Owner** | `portfolio` (Position entity) |
| **Input** | Fills / reduce / close events |
| **Output** | Updated `Position`; events |
| **Events** | `PositionOpened`, `PositionUpdated`, `PositionClosed` |
| **Validation** | Qty accounting; side consistency; open⇒qty>0 |
| **Failure handling** | Inconsistent fill → halt apply, CRITICAL, kill-switch candidate |
| **Retry policy** | Apply is local — no retry; repair via reconcile + manual ops runbook |
| **Logging** | INFO open/close; DEBUG updates |
| **Monitoring** | Open positions count; per-symbol exposure |

---

### 2.9 Portfolio lifecycle

| Aspect | Specification |
|---|---|
| **Purpose** | Account SoT: equity, margin, exposure, kill-switch |
| **Owner** | `portfolio` / `IPortfolioEngine` |
| **Input** | Order events, marks-to-market, deposits (ops) |
| **Output** | Mutable Portfolio + immutable `PortfolioSnapshot` |
| **Events** | `PortfolioMarked`, `EquityUpdated`, `DrawdownBreach`, `KillSwitchArmed`, `KillSwitchCleared` |
| **Validation** | Accounting identity within tolerance; snapshot monotonic `as_of` for a run |
| **Failure handling** | Break identity → freeze new TradePlans (kill-switch), alert |
| **Retry policy** | Mark-to-market: retry price fetch; freeze trading if mark fails beyond SLA |
| **Logging** | INFO equity marks (throttled); CRITICAL breaches |
| **Monitoring** | Equity, DD, daily PnL, margin util, kill-switch state |

---

### 2.10 Analytics lifecycle

| Aspect | Specification |
|---|---|
| **Purpose** | Offline evaluation artifacts (`PerformanceReport`, WF, SHAP, calibration) |
| **Owner** | `analytics` / `IAnalyticsEngine` |
| **Input** | Run logs, trade blotter, equity curve, optional model diagnostics |
| **Output** | `PerformanceReport` (+ refs) |
| **Events** | `ReportGenerated`, `WalkForwardCompleted`, `SensitivityCompleted` |
| **Validation** | Period bounds; required metrics by mode; config hash recorded |
| **Failure handling** | Job fail → retry offline; **never** blocks live loop |
| **Retry policy** | Job-level retry with backoff; poison-message quarantine |
| **Logging** | INFO report_id + headline metrics |
| **Monitoring** | Job success rate; staleness of last sealed report |

---

## 3. Mode end-to-end sequences

### 3.1 Offline Training (sequence)

```
App.Train
  → IMarketDataProvider.get_history()           → MarketData
  → IFeatureEngine.transform()                  → FeatureSet
  → ILabelEngine.label()                        → LabelSet
  → assemble Dataset
  → IModelTrainer.fit()                         → ModelArtifact
  → ICalibrationEngine.fit()                    → CalibratorArtifact
  → Registry.register()                         → ModelRegistered
  → IAnalyticsEngine.evaluate / walk_forward    → PerformanceReport
```

### 3.2 Offline Backtesting (sequence)

```
App.Backtest (historical clock)
  loop bar_as_of:
    MarketData snapshot (causal cut)
    → FeatureSet → Signal(s)
    → Decision → TradePlan|Reject
    → Execution(sim) → Fills
    → Portfolio.apply
  → Analytics.evaluate → PerformanceReport
```

### 3.3 Paper Trading (sequence)

```
App.Paper (live clock, closed bars)
  on BarClosed:
    same spine as backtest
    Execution(paper) — no venue order_send
    Monitoring heartbeats + hypothetical PnL
```

### 3.4 Live Trading (sequence)

```
App.Live
  on BarClosed (and health OK):
    same spine
    Execution(live) submit with idempotency
    reconcile loop (parallel)
    kill-switch short-circuits Risk/Execution
```

---

## 4. Activity diagrams (textual)

### 4.1 Hot-path activity (Backtest / Paper / Live)

```
[Bar closed]
    → Validate MarketData
         ⊦ fail → log + skip bar → [Wait next bar]
    → Build FeatureSet
         ⊦ warmup/NaN → skip bar
    → Predict + Calibrate → Signal
         ⊦ invalid → skip bar
    → Decide
         ⊦ skip → emit DecisionSkipped → [Wait]
         ⊦ trade → Risk.plan
              ⊦ Reject → emit RiskRejected → [Wait]
              ⊦ TradePlan → Execution.submit
                   ⊦ reject/fail → OrderRejected → [Wait]
                   ⊦ fill → Position/Portfolio update → [Wait]
```

### 4.2 Train activity

```
[Load history] → [Features] → [Labels] → [Dataset]
    → [Fit model] → [Calibrate] → [Validate / sealed checks]
    → [Promote?] → [Register] → [Analytics reports]
```

---

## 5. State machines

### 5.1 Order state machine

```
created → submitted → partial → filled
                 ↘ canceled
                 ↘ rejected
                 ↘ expired
partial → filled | canceled | rejected
```

Illegal transitions are hard errors (CRITICAL).

### 5.2 Position state machine

```
(none) → open → open (updated) → closed
```

### 5.3 Model promotion state machine

```
fitted → validated → staged → promoted → retired
              ↘ rejected
staged → retired (abort)
```

Only `promoted` models may be used in paper/live inference.

### 5.4 Kill-switch state machine (Portfolio / App supervisor)

```
disarmed → armed → disarmed (manual clear)
```

While `armed`: Risk returns Reject for new entries; Execution cancels open intents per policy; flatten optional by runbook (not automatic unless configured).

### 5.5 Run mode state (App)

```
init → ready → running → pausing → running
                      ↘ stopping → stopped
                      ↘ faulted → stopped
```

---

## 6. Event flow (platform-wide)

```
Data: BarClosed
  → Feature: FeatureSetBuilt | FeatureValidationFailed
  → Model: InferenceCompleted
  → Signal: SignalEmitted | SignalRejected
  → Decision: DecisionMade | DecisionSkipped
  → Risk: TradePlanCreated | RiskRejected
  → Execution: OrderSubmitted → OrderFilled | OrderRejected
  → Position/Portfolio: Position* / PortfolioMarked
  → Monitoring: metrics/alerts (sync, lightweight)
  → Analytics: consumes persisted event log asynchronously
```

**Back-pressure rule:** If Monitoring sink is slow, drop or sample DEBUG; never block Execution. If event log write fails on live, WARN and continue with local buffer; CRITICAL if buffer overflows.

---

## 7. Sequence diagrams (detailed)

### 7.1 Live bar (happy path)

```
Clock/Feed → App.Live: BarClosed(as_of)
App → MarketDataProvider: get_closed_bars
App → FeatureEngine: transform
App → ModelPredictor: predict
App → CalibrationEngine: apply
App → DecisionEngine: decide(signals, ctx)
App → PortfolioEngine: snapshot
App → RiskEngine: plan(decision, snapshot)
App → ExecutionEngine: submit(plan)
Execution → Venue: place (idempotent)
Venue → Execution: fill
Execution → PortfolioEngine: apply(fill)
App → Monitoring: record(trace)
```

### 7.2 Live bar (risk reject)

```
… → DecisionEngine: Decision(trade)
→ RiskEngine: Reject(daily_loss)
→ Monitoring: RiskRejected
→ (no Execution call)
```

### 7.3 Venue disconnect

```
Execution: FeedDisconnected / submit timeout
→ retry budget
→ OrderRejected or hold in submitted + reconcile
→ Monitoring CRITICAL
→ optional KillSwitchArmed if policy says so
→ on reconnect: reconcile() before new submits
```

---

## 8. Validation matrix by mode

| Check | Train | Backtest | Paper | Live |
|---|---|---|---|---|
| Causal features | Required | Required | Required | Required |
| Closed-bar gate | N/A (batch) | Required | Required | Required |
| Promoted model only | N/A | Configurable | Required | Required |
| Paper tripwire (no live send) | N/A | N/A | Required | N/A |
| Reconcile | N/A | N/A | Optional | Required |
| Kill-switch | N/A | Optional sim | Required | Required |
| Sealed analytics | Required | Required | Periodic | Periodic |

---

## 9. Failure & retry summary table

| Stage | Fail-closed default | Retry |
|---|---|---|
| Data | Skip bar / pause loop | Yes (fetch/reconnect) |
| Feature | Skip bar | No |
| Model infer | Skip bar | Load: once |
| Signal | Skip | No |
| Decision | Skip (exception→CRITICAL) | No |
| Risk | Reject | No |
| Execution | Reject order / reconcile | Yes (idempotent submit) |
| Position/Portfolio | Kill-switch on invariant break | No (repair runbook) |
| Analytics | Job fail | Yes (offline) |

---

## 10. Logging & monitoring map

| Lifecycle | Key log events | Key metrics |
|---|---|---|
| Data | load range, gaps | last_bar_age, gaps |
| Feature | recipe, n_rows, skips | build_ms, nan_rate |
| Model | fit summary, model_key | infer_ms, drift |
| Signal | emit/reject | p_mean, reject% |
| Decision | action, reason | skip_by_reason |
| Risk | plan/reject code | reject_by_code, risk$ |
| Execution | state transitions | latency, slippage |
| Position | open/close | exposure |
| Portfolio | marks, DD, kill | equity, DD%, kill |
| Analytics | report_id | job_success |

---

## 11. Extensibility — add features without modifying existing modules

Principle: **Open for extension via new implementations + composition in `apps/` + new filter/policy plugins registered by settings.** Existing modules depend on interfaces and immutable domain objects only.

| Capability | How to add | What you must not do |
|---|---|---|
| **Multi-Timeframe** | New `ModelKey`s per TF; FeatureEngine recipes per TF; Decision **fusion plugin** (`IConflictResolver` / ranker) selected by strategy settings | Concatenate all TF into one model inside Features by default; modify Risk/Execution |
| **Meta Label** | New `label_type` in `ILabelEngine` impl; secondary `ModelKey` (meta model); Decision or primary model gate consumes meta `Signal` | Change `TradePlan` shape; bake meta into FeatureEngine |
| **Dynamic RR** | Risk policy plugin reads Decision hints + ATR from context DTO; sets SL/TP distances on `TradePlan` | Hardcode RR inside Decision or Execution |
| **Kelly** | New `IRiskEngine` / sizing strategy implementation behind same `plan()` contract | Fork Execution or change Order state machine |
| **Multiple Strategies** | New `StrategyId` + settings profile; App runs N pipelines; Portfolio already per-strategy positions; Decision conflict resolver across strategies | Global mutable “current strategy”; shared Position without strategy id |
| **Transformer Models** | New `IModelTrainer`/`IModelPredictor` family plugin; Registry stores artifact; FeatureSet may add sequence accessor **without** changing Decision | Import Transformer code from Decision/Risk; change Signal required fields (only optional `extras`) |

### Extension seams (stable)

1. `IFeatureEngine` recipe registry  
2. `ILabelEngine` label_type registry  
3. Model family registry (`ModelKey.family`)  
4. Decision filter chain (ordered list of `IDecisionFilter`)  
5. Risk policy registry  
6. Execution adapters (`sim` / `paper` / `mt5` / future)  
7. `apps/*` composition only  

**Compatibility rule:** New optional fields on domain objects are additive. Removing/renaming required fields is a versioned breaking change (`schema_version` / `ModelKey.version`).

---

## 12. Non-goals

- No code, no class bodies, no storage DDL in this document.
- No retuning of settled strategy constants (threshold, Z_BLOCK, sizing maps).
- No requirement that Analytics be synchronous with live trading.

---

## 13. Acceptance criteria for a future implementation

1. One shared hot-path for backtest/paper/live differing only by adapters.  
2. Every stage emits documented events with `trace_id`.  
3. Fail-closed on Feature/Model/Risk/Portfolio invariant breaks.  
4. Paper adapter cannot place live orders (automated test).  
5. New model family or risk policy landable with zero edits to Decision/Execution cores — only registry + app wiring.  
6. Lifecycle behavior matches this document’s state machines and retry taxonomy.

---

## Document control

| Version | Date | Notes |
|---|---|---|
| 1.0 | 2026-07-13 | Initial lifecycle SDD |
