# mlfrog — Domain Model Specification

| Field | Value |
|---|---|
| Document type | Domain Model / Ubiquitous Language |
| Audience | Quant eng, ML eng, platform eng |
| Status | Design only — **no implementation** |
| Companion | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Principle | Domains exchange **domain objects**, not raw `DataFrame`s, at module boundaries |

---

## 0. Modeling rules

1. **Bounded context owns its aggregates.** Cross-context communication uses published domain objects (or shared kernel value objects), never another context’s internal tables.
2. **Pandas is an adapter concern.** A `FeatureSet` *may* store matrix data in columnar form internally, but the public type is `FeatureSet`, not `pd.DataFrame`. Callers must not reach into `.df` as the API.
3. **Immutability by default.** Objects that represent facts at a point in time are immutable. Ledgers (`Order`, `Position`, `Portfolio`) are deliberately mutable aggregates with explicit transition methods.
4. **Identity over columns.** Every series-aligned object carries `InstrumentId`, `Timeframe`, and temporal keys — never “row 42 means XAUUSD H1”.
5. **Validation at construction.** Invalid domain objects must not exist. Factories / constructors enforce invariants; engines assume valid inputs.
6. **No hidden globals.** Lifecycle is explicit: create → use → (optionally) persist via ports. No process-wide singleton market cache as domain state.

### Shared kernel (value objects used everywhere)

These are not listed in the user’s twelve entities but are required vocabulary:

| Type | Meaning | Mutability |
|---|---|---|
| `InstrumentId` | Symbol + venue (+ optional broker suffix) | Immutable |
| `Timeframe` | M1…MN discrete bar size | Immutable |
| `StrategyId` | Named strategy namespace / version | Immutable |
| `ModelKey` | `(StrategyId, InstrumentId, Timeframe, Family, Version)` | Immutable |
| `Timestamp` | Instant with explicit timezone (UTC preferred) | Immutable |
| `Money` | Amount + currency | Immutable |
| `Quantity` | Lots / units with step/min metadata ref | Immutable |
| `Price` | Decimal price in instrument quote | Immutable |
| `Reject` | Typed refusal (`code`, `message`, `stage`) | Immutable |

---

## 1. Entity catalog

### 1.1 `MarketData`

| Aspect | Specification |
|---|---|
| **Purpose** | Aggregate root representing a contiguous (or explicitly gapped) series of candles for one instrument and one timeframe. The sole market-history handle passed into Feature / Label engines. |
| **Owner module** | `market_data` |
| **Lifecycle** | Created by `IMarketDataProvider` (load or stream buffer flush). Read-only thereafter for a given snapshot id. New bars produce a **new** `MarketData` snapshot or an append via provider-owned buffer that yields a new snapshot — consumers never mutate candles in place. Discarded when out of lookback window. |
| **Mutability** | **Immutable snapshot** (preferred). Streaming buffers live inside the provider adapter, not on the domain object. |
| **Required fields** | `instrument: InstrumentId`; `timeframe: Timeframe`; `candles: Sequence[Candle]` (time-ordered); `as_of: Timestamp` (snapshot validity); `source: str` (e.g. `mt5`, `csv`, `parquet`) |
| **Optional fields** | `gaps: Sequence[TimeRange]` (known missing intervals); `quality_flags: frozenset[str]`; `schema_version: str`; `session_calendar_id: str` |
| **Validation** | Non-empty unless explicitly allowed empty warm-up; strictly increasing `Candle.close_time`; all candles share same instrument/timeframe; no duplicate timestamps; OHLC consistency delegated to each `Candle`; `as_of >= last.close_time` for closed-bar snapshots |

---

### 1.2 `Candle`

| Aspect | Specification |
|---|---|
| **Purpose** | Single OHLCV bar — atomic market observation. Timestamps are **bar close** time (platform convention). |
| **Owner module** | `market_data` (value object) |
| **Lifecycle** | Born when bar closes (or historical load). Never updated; corrections replace via new `MarketData` snapshot. |
| **Mutability** | **Immutable** |
| **Required fields** | `instrument: InstrumentId`; `timeframe: Timeframe`; `open_time: Timestamp`; `close_time: Timestamp`; `open: Price`; `high: Price`; `low: Price`; `close: Price`; `volume: float` (tick or real per `volume_type`) |
| **Optional fields** | `spread: float` (points or price); `tick_volume: float`; `real_volume: float`; `volume_type: Literal["tick","real"]`; `is_closed: bool` (must be true on signal path) |
| **Validation** | `high >= max(open, close)` and `low <= min(open, close)`; `high >= low`; `close_time > open_time`; duration matches `timeframe` within tolerance; finite non-NaN prices; `is_closed=True` required before FeatureEngine on live path |

---

### 1.3 `FeatureSet`

| Aspect | Specification |
|---|---|
| **Purpose** | Causal feature matrix aligned to observation times, plus schema identity. Replaces passing raw feature DataFrames across module boundaries. |
| **Owner module** | `features` |
| **Lifecycle** | Produced by `IFeatureEngine` from `MarketData` (multi-TF map allowed as input to engine, output is one `FeatureSet` per recipe target TF). Consumed by trainer/predictor. Immutable once built. Versioned by `recipe_id`. |
| **Mutability** | **Immutable** |
| **Required fields** | `instrument: InstrumentId`; `timeframe: Timeframe` (prediction TF); `recipe_id: str`; `feature_names: Sequence[str]`; `rows: Sequence[FeatureRow]` **or** opaque `matrix` accessible only via typed accessors; `as_of: Timestamp`; `causal: bool` (must be true for production) |
| **Optional fields** | `categorical_names: Sequence[str]`; `normalization_id: str`; `selection_id: str`; `warmup_bars_dropped: int`; `source_market_data_ids: Sequence[str]` |
| **Validation** | `len(feature_names) > 0`; no duplicate names; row count ≥ 1 for inference batch or allow empty with explicit flag; all rows same width; timestamps strictly increasing; `causal=True` for live/paper/backtest hot path; NaN policy explicit (`deny` default for inference last row) |
| **Anti-leak** | Must not contain future-dependent columns; `as_of` equals last row timestamp on live single-bar inference |

`FeatureRow` (value): `timestamp`, `values: Mapping[str, float|int|str]` (or parallel arrays).

---

### 1.4 `LabelSet`

| Aspect | Specification |
|---|---|
| **Purpose** | Supervised targets aligned to observation times. Train/validation only — **not** on live hot path. |
| **Owner module** | `labels` |
| **Lifecycle** | Built by `ILabelEngine` from `MarketData` (+ optional barrier config). Joined with `FeatureSet` into `Dataset`. Immutable. |
| **Mutability** | **Immutable** |
| **Required fields** | `instrument: InstrumentId`; `timeframe: Timeframe`; `label_type: str` (e.g. `triple_barrier`, `meta_label`, `future_return`); `label_names: Sequence[str]`; `rows: Sequence[LabelRow]`; `params: Mapping[str, Any]` (barrier multipliers, horizon, …); `as_of_max: Timestamp` |
| **Optional fields** | `sample_weights: Sequence[float]`; `mask_valid: Sequence[bool]`; `aux_columns: Mapping[str, Sequence]` (e.g. touch time, barrier hit side) |
| **Validation** | Known `label_type`; params complete for type (e.g. TB requires SL/TP/horizon); timestamps unique; no label uses information after its `timestamp` (engine responsibility, asserted in analytics) |

---

### 1.5 `Dataset`

| Aspect | Specification |
|---|---|
| **Purpose** | Inner join of `FeatureSet` and `LabelSet` on `(instrument, timeframe, timestamp)` for training/evaluation. Single object handed to `IModelTrainer`. |
| **Owner module** | `models` (assembly) or `shared` training kernel — **owner: `models`** as consumer-facing train input; constructed by app/train orchestration using features+labels outputs. |
| **Lifecycle** | Created at train time; may be split into train/val/test views (immutable subsets). Discarded after fit or persisted as artifact metadata only. |
| **Mutability** | **Immutable** (splits are new instances) |
| **Required fields** | `features: FeatureSet`; `labels: LabelSet`; `alignment_key: Sequence[Timestamp]`; `n_rows: int`; `recipe_id`; `label_type`; `instrument`; `timeframe` |
| **Optional fields** | `split_id: str`; `sample_weights`; `metadata: Mapping` (git hash, data range, sealed flag) |
| **Validation** | Feature and label instruments/timeframes equal; timestamps exact match on alignment; `n_rows == len(alignment_key) > 0`; feature `recipe_id` recorded; no orphan labels |

---

### 1.6 `Signal`

| Aspect | Specification |
|---|---|
| **Purpose** | Model output for one instrument/timeframe/strategy at one as-of time. ML-facing fact; Decision must not need model internals beyond this object. (Renames/generalizes “Prediction” in platform architecture.) |
| **Owner module** | `models` |
| **Lifecycle** | Emitted by `IModelPredictor` (optionally after `ICalibrationEngine`). Consumed by `IDecisionEngine`. Immutable. Multiple signals (multi-TF/multi-model) may be batched as `Sequence[Signal]`. |
| **Mutability** | **Immutable** |
| **Required fields** | `model_key: ModelKey`; `instrument: InstrumentId`; `timeframe: Timeframe`; `as_of: Timestamp`; `direction: Literal["long","short","flat"]`; `p_raw: float` |
| **Optional fields** | `p_calibrated: float`; `confidence: float` or `confidence_tier: str`; `expected_return: float`; `horizon_bars: int`; `feature_as_of: Timestamp`; `extras: Mapping` (model-specific, ignored by Decision unless declared) |
| **Validation** | Probabilities in `[0,1]` when present; `direction` consistent with strategy mode (e.g. long-only strategies reject `short`); `as_of` equals closed bar time; `model_key` components match instrument/timeframe fields |

---

### 1.7 `Decision`

| Aspect | Specification |
|---|---|
| **Purpose** | Policy outcome: whether to proceed toward a trade, and why. Carries filter audit trail. Does **not** carry final lot size. |
| **Owner module** | `decision` |
| **Lifecycle** | Produced by `IDecisionEngine` from `Signal`(s) + context. Consumed by `IRiskEngine`. Immutable. |
| **Mutability** | **Immutable** |
| **Required fields** | `decision_id: UUID`; `strategy_id: StrategyId`; `instrument: InstrumentId`; `as_of: Timestamp`; `action: Literal["trade","skip"]` (extendable: `skip_threshold`, `skip_regime`, … as reason codes); `side: Literal["long","short","none"]`; `reason: str`; `signals_used: Sequence[Signal]` (by value or id) |
| **Optional fields** | `rank_score: float`; `regime_tier: str`; `regime_size_hint: float` (hint only); `confidence_tier: str`; `filters_passed: Sequence[str]`; `filters_failed: Sequence[str]`; `session_id: str`; `spread_at_decision: float` |
| **Validation** | If `action=="trade"` then `side` in `{long,short}` and `filters_failed` empty (or only soft warnings policy); if skip, `side` may be `none`; `signals_used` non-empty for trade; `regime_size_hint` in `(0,1]` when present |

---

### 1.8 `TradePlan`

| Aspect | Specification |
|---|---|
| **Purpose** | Risk-approved intent to enter (or scale) a position: size, stops, risk budget. Broker-agnostic. |
| **Owner module** | `risk` |
| **Lifecycle** | Created by `IRiskEngine` from `Decision` + `Portfolio`. Consumed by `IExecutionEngine`. Immutable. Rejection is `Reject`, not a partial `TradePlan`. |
| **Mutability** | **Immutable** |
| **Required fields** | `plan_id: UUID`; `decision_id: UUID`; `strategy_id: StrategyId`; `instrument: InstrumentId`; `side: Literal["long","short"]`; `quantity: Quantity`; `entry_type: Literal["market","limit"]`; `stop_loss: Price \| Distance`; `take_profit: Price \| Distance`; `risk_amount: Money`; `as_of: Timestamp` |
| **Optional fields** | `limit_price: Price`; `max_slippage: float`; `time_in_force: str`; `reduce_only: bool`; `tag: str`; `constraints_checked: Sequence[str]` (e.g. `daily_loss`, `dd_cap`, `exposure`) |
| **Validation** | `quantity > 0` and respects step/min; SL/TP on correct side of intended entry; `risk_amount > 0`; `decision_id` present; not creatable when portfolio kill-switch active (engine rule) |

---

### 1.9 `Order`

| Aspect | Specification |
|---|---|
| **Purpose** | Execution lifecycle aggregate for a single order attempt derived from a `TradePlan`. |
| **Owner module** | `execution` |
| **Lifecycle** | `created` → `submitted` → (`partial`) → `filled` \| `canceled` \| `rejected` \| `expired`. Transitions only via `IExecutionEngine` / OrderManager. Terminal states immutable thereafter. |
| **Mutability** | **Mutable aggregate** until terminal; then frozen |
| **Required fields** | `order_id: UUID`; `plan_id: UUID`; `instrument: InstrumentId`; `side`; `quantity: Quantity`; `order_type`; `state: OrderState`; `created_at: Timestamp` |
| **Optional fields** | `broker_order_id: str`; `filled_quantity: Quantity`; `avg_fill_price: Price`; `leaves_quantity: Quantity`; `slippage: float`; `spread_at_send: float`; `reject_reason: str`; `updates: Sequence[OrderEvent]` |
| **Validation** | State machine transitions only; `filled_quantity ≤ quantity`; terminal states have no further transitions; paper mode never sets real `broker_order_id` from live venue |

**OrderState:** `created | submitted | partial | filled | canceled | rejected | expired`

---

### 1.10 `Position`

| Aspect | Specification |
|---|---|
| **Purpose** | Open (or flattened) exposure in one instrument for one strategy (or portfolio net — see note). Tracks qty, avg price, unrealized risk anchors. |
| **Owner module** | `portfolio` |
| **Lifecycle** | Opened on fill; updated on partial fills / add / reduce; closed when qty→0 (retained as historical closed position or archived). |
| **Mutability** | **Mutable** while open; closed snapshot can be immutable archive |
| **Required fields** | `position_id: UUID`; `instrument: InstrumentId`; `strategy_id: StrategyId`; `side`; `quantity: Quantity`; `avg_entry: Price`; `opened_at: Timestamp`; `status: Literal["open","closed"]` |
| **Optional fields** | `stop_loss`; `take_profit`; `unrealized_pnl: Money`; `realized_pnl: Money`; `margin_used: Money`; `linked_order_ids: Sequence[UUID]` |
| **Validation** | Open ⇒ `quantity > 0`; closed ⇒ `quantity == 0`; side consistent with qty sign convention (platform: explicit side + abs qty) |

**Note:** Prefer **per-strategy positions** so multi-strategy on same symbol remains accountable; Portfolio aggregates them.

---

### 1.11 `Portfolio`

| Aspect | Specification |
|---|---|
| **Purpose** | Aggregate root for account-level state: positions, cash/equity, margin, exposure. Source of truth for Risk and Monitoring. |
| **Owner module** | `portfolio` |
| **Lifecycle** | Created at session start (paper/live/backtest). Mutated only by applying execution events / marks-to-market. Snapshots (`PortfolioSnapshot`) emitted for Decision/Risk read models. |
| **Mutability** | **Mutable aggregate**; publish **immutable snapshots** for consumers |
| **Required fields** | `portfolio_id: UUID`; `base_currency: str`; `equity: Money`; `cash: Money`; `positions: Mapping[PositionId, Position]`; `as_of: Timestamp` |
| **Optional fields** | `margin_used`; `margin_free`; `peak_equity`; `drawdown_pct`; `exposure_by_instrument`; `daily_realized_pnl`; `kill_switch: bool` |
| **Validation** | Equity accounting identity holds within tolerance after each apply; no duplicate open position keys for same `(strategy, instrument)` unless stacking policy allows; Risk reads snapshot, never mutates Portfolio directly |

---

### 1.12 `PerformanceReport`

| Aspect | Specification |
|---|---|
| **Purpose** | Immutable analytics artifact summarizing a run (backtest, sealed eval, paper window, live period). |
| **Owner module** | `analytics` |
| **Lifecycle** | Produced by `IAnalyticsEngine` from trade/equity event logs + optional model diagnostics. Persisted under artifacts; never feeds live Order path. |
| **Mutability** | **Immutable** |
| **Required fields** | `report_id: UUID`; `run_id: str`; `mode: Literal["backtest","paper","live","sealed"]`; `period_start`; `period_end`; `instrument_scope`; `strategy_scope`; `metrics: PerformanceMetrics` |
| **Optional fields** | `trades: Sequence[TradeSummary]`; `equity_curve_ref: ArtifactRef`; `feature_importance_ref`; `shap_ref`; `calibration_ref`; `notes: str`; `config_hash: str` |
| **Validation** | `period_end >= period_start`; required metrics present for mode (e.g. backtest: CAGR, MaxDD, Sharpe, n_trades, win_rate); refs resolve or are null |

`PerformanceMetrics` (value): `n_trades`, `win_rate`, `cagr`, `sharpe`, `sortino`, `max_drawdown_pct`, `avg_trade`, `profit_factor`, …

---

## 2. Object relationship (conceptual)

```
Candle ──┐
         ├── MarketData ──► IFeatureEngine ──► FeatureSet ──┐
         │                                                 ├── Dataset ──► IModelTrainer
         └── MarketData ──► ILabelEngine ──► LabelSet ─────┘
                                                              │
                                                     IModelPredictor
                                                              │
                                                           Signal
                                                              │
                                                     IDecisionEngine
                                                              │
                                                          Decision
                                                              │
                                    Portfolio ◄── IRiskEngine ──► TradePlan
                                                              │
                                                     IExecutionEngine
                                                              │
                                                    Order ──fills──► Position
                                                              │
                                                           Portfolio
                                                              │
                                                     IAnalyticsEngine
                                                              │
                                                    PerformanceReport
```

Calibration sits on the ML path: `IModelPredictor` may emit `p_raw` only; `ICalibrationEngine` enriches to `p_calibrated` inside or beside Signal creation (still within `models`).

---

## 3. Interface catalog

All interfaces are **ports**. Implementations live in adapters or domain services; `apps/*` wires them.

### 3.1 `IMarketDataProvider`

| | |
|---|---|
| **Owner** | `market_data` |
| **Purpose** | Supply historical and live market observations as domain objects. |
| **Key operations** | `get_history(instrument, timeframe, start, end) -> MarketData`; `get_closed_bars(instrument, timeframe, lookback) -> MarketData`; `subscribe(instrument) -> Iterator[Candle \| Quote]` (live) |
| **Must not** | Compute features, place orders |
| **Returns** | `MarketData` / `Candle` — never bare DataFrame at the port boundary |

### 3.2 `IFeatureEngine`

| | |
|---|---|
| **Owner** | `features` |
| **Purpose** | Transform market snapshots into causal `FeatureSet`. |
| **Key operations** | `transform(market: Mapping[Timeframe, MarketData], recipe_id, as_of) -> FeatureSet` |
| **Must not** | Know models, labels, or broker |
| **Invariant** | Output `causal=True` for production recipes |

### 3.3 `ILabelEngine`

| | |
|---|---|
| **Owner** | `labels` |
| **Purpose** | Generate supervised `LabelSet` for training/evaluation. |
| **Key operations** | `label(market: MarketData, config: LabelConfig) -> LabelSet` |
| **Must not** | Appear on live order path |
| **Invariant** | Labels use only information available after barriers resolve historically (no peeking beyond configured horizon in unintended ways — verified in analytics) |

### 3.4 `IModelTrainer`

| | |
|---|---|
| **Owner** | `models` |
| **Purpose** | Fit a model artifact for a `ModelKey`. |
| **Key operations** | `fit(dataset: Dataset, key: ModelKey, train_config) -> ModelArtifact` |
| **Input** | `Dataset` only (not raw frames) |
| **Output** | Opaque `ModelArtifact` + metadata registered externally |

### 3.5 `IModelPredictor`

| | |
|---|---|
| **Owner** | `models` |
| **Purpose** | Inference: features → `Signal`. |
| **Key operations** | `predict(features: FeatureSet, key: ModelKey) -> Signal` (or `Sequence[Signal]` for batch) |
| **Must not** | Size positions or call execution |

### 3.6 `ICalibrationEngine`

| | |
|---|---|
| **Owner** | `models` (calibration) |
| **Purpose** | Map raw scores/probabilities to calibrated probabilities / confidence. |
| **Key operations** | `fit(signals_or_scores, labels) -> CalibratorArtifact`; `apply(signal: Signal) -> Signal` (returns new Signal with `p_calibrated` / confidence) |
| **Mutability contract** | Returns new `Signal`; does not mutate input |

### 3.7 `IDecisionEngine`

| | |
|---|---|
| **Owner** | `decision` |
| **Purpose** | Convert `Signal`(s) + context into a `Decision`. |
| **Key operations** | `decide(signals: Sequence[Signal], ctx: DecisionContext) -> Decision` |
| **Context includes** | session, spread, portfolio snapshot (read-only), open position flags, clock |
| **Must not** | Import model families; compute final lots |

### 3.8 `IRiskEngine`

| | |
|---|---|
| **Owner** | `risk` |
| **Purpose** | Convert `Decision` + portfolio state into `TradePlan` or `Reject`. |
| **Key operations** | `plan(decision: Decision, portfolio: PortfolioSnapshot, cfg: RiskConfig) -> TradePlan \| Reject` |
| **Must not** | Submit broker orders |

### 3.9 `IPortfolioEngine`

| | |
|---|---|
| **Owner** | `portfolio` |
| **Purpose** | Own portfolio aggregate; apply fills; expose snapshots. |
| **Key operations** | `snapshot() -> PortfolioSnapshot`; `apply(event: OrderEvent \| Fill \| MarkToMarket) -> Portfolio`; `get_position(strategy, instrument) -> Position \| None` |
| **Must not** | Generate signals or features |

### 3.10 `IExecutionEngine`

| | |
|---|---|
| **Owner** | `execution` |
| **Purpose** | Turn `TradePlan` into venue orders; manage lifecycle; emit fills. |
| **Key operations** | `submit(plan: TradePlan) -> Order`; `cancel(order_id) -> Order`; `reconcile() -> Sequence[OrderEvent]` |
| **Adapters** | sim / paper / mt5 — same interface |
| **Must not** | Know feature recipes or model keys (beyond optional tag passthrough) |

### 3.11 `IAnalyticsEngine`

| | |
|---|---|
| **Owner** | `analytics` |
| **Purpose** | Build `PerformanceReport` and research diagnostics offline. |
| **Key operations** | `evaluate(run: RunLog \| TradeLog, config) -> PerformanceReport`; optional `walk_forward`, `sensitivity`, `calibration_report` |
| **Must not** | Be imported by Decision / Risk / Execution hot path |

---

## 4. Dependency graph (domain objects & ports)

```
                    ┌──────────────── apps (composition) ────────────────┐
                    │  constructs adapters; passes domain objects only   │
                    └─────────────────────────┬──────────────────────────┘
                                              │
     IMarketDataProvider                      │
              │                               │
              ▼                               │
         MarketData / Candle                  │
              │                               │
       ┌──────┴──────┐                        │
       ▼             ▼                        │
 IFeatureEngine  ILabelEngine                 │
       │             │                        │
       ▼             ▼                        │
   FeatureSet     LabelSet                    │
       │             │                        │
       └──────┬──────┘                        │
              ▼                               │
           Dataset                            │
              │                               │
       IModelTrainer / IModelPredictor / ICalibrationEngine
              │                               │
              ▼                               │
            Signal                            │
              │                               │
       IDecisionEngine                        │
              │                               │
              ▼                               │
          Decision                            │
              │                               │
       IRiskEngine ◄──── PortfolioSnapshot ── IPortfolioEngine
              │                                      ▲
              ▼                                      │
          TradePlan                                  │
              │                                      │
       IExecutionEngine ── Order / Fill ─────────────┘
              │
              ▼
         (events log)
              │
       IAnalyticsEngine
              │
              ▼
      PerformanceReport
```

### Dependency rules

| Rule | Detail |
|---|---|
| Direction | Depend inward on shared kernel + upstream outputs only |
| Object flow | Downstream may hold references to upstream immutable objects (`Decision.signals_used`) |
| No cycles | `execution` must not import `features` / `models`; `decision` must not import `risk` implementations |
| Snapshot isolation | Risk/Decision receive `PortfolioSnapshot` (immutable), not the live mutable `Portfolio` |
| DataFrame ban at boundaries | Ports accept/return domain types; adapters may use pandas internally |

---

## 5. Lifecycle summary (immutability map)

| Object | Immutable? | Why |
|---|---|---|
| Candle | Yes | Fact |
| MarketData | Yes (snapshot) | Reproducible inputs |
| FeatureSet | Yes | Causal audit |
| LabelSet | Yes | Train reproducibility |
| Dataset | Yes | Train reproducibility |
| Signal | Yes | Model fact |
| Decision | Yes | Policy audit |
| TradePlan | Yes | Risk audit |
| Order | Mutable → freeze | Venue lifecycle |
| Position | Mutable while open | Ledger |
| Portfolio | Mutable + snapshots | Ledger |
| PerformanceReport | Yes | Analytics artifact |

---

## 6. Validation & testing expectations (design)

- Every immutable type: construction tests for invariant violations.
- `Order` / `Portfolio`: explicit state-transition tests.
- Port contract tests: round-trip CSV/MT5 adapters produce valid `MarketData`.
- Golden path test: `Signal → Decision → TradePlan` with fixture objects, zero pandas at the seam.
- Leakage tests belong to Analytics, not to FeatureEngine unit tests alone.

---

## 7. Explicit non-goals (this document)

- No class method signatures in Python.
- No storage schema / ORM mapping.
- No change to settled strategy numeric constants (thr, Z_BLOCK, sizing maps).

---

## 8. Alignment with platform architecture

This domain model is the **ubiquitous language** binding for [ARCHITECTURE.md](ARCHITECTURE.md). Where architecture said `Prediction` / `BarSeries`, this document standardizes production names **`Signal`** and **`MarketData`/`Candle`**. Both refer to the same seams; implementation must pick these names and not fork synonyms across packages.
