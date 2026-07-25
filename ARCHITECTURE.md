# mlfrog — Production Platform Architecture

**Status:** design only (no implementation in this document).  
**Nature:** Machine Learning Trading **Platform** — not a research repo, not notebooks-as-source-of-truth, not a backtesting framework.

Interactive companion: open the canvas beside chat  
(`canvases/mlfrog-platform-architecture.canvas.tsx`).

**Domain model (entities + ports):** [DOMAIN_MODEL.md](DOMAIN_MODEL.md) · canvas `mlfrog-domain-model.canvas.tsx`.  
**Lifecycle SDD (modes + stage machines):** [LIFECYCLE.md](LIFECYCLE.md) · canvas `mlfrog-lifecycle-sdd.canvas.tsx`.

---

## 1. Vision

The ML model is **one pluggable component**. The platform must support, without major refactor:

| Capability | How the architecture enables it |
|---|---|
| Training | `apps/train` wires Market Data → Features → Labels → Models → Analytics |
| Backtesting | Same Decision→Risk→Portfolio path; `ExecutionPort` = simulator |
| Paper trading | Same path; `ExecutionPort` = paper ledger (no live `order_send`) |
| Live trading | Same path; `ExecutionPort` = MT5 (or other) adapter |
| Multiple strategies | First-class `StrategyId` + per-strategy settings / models / policies |
| Multiple symbols | First-class `InstrumentId`; Portfolio owns cross-symbol state |
| Multiple timeframes | **One model per timeframe**; fusion in Decision — not one mega-model |
| Multiple ML models | `ModelTrainer` / `ModelPredictor` plugins + `ModelRegistry` |
| Multiple risk models | `RiskPolicy` plugins emitting the same `TradePlan` |

**Design decision:** Modes differ only by **adapters** (data source, clock, execution). Domain policies are shared. This prevents the classic failure mode where paper/live diverge silently.

---

## 2. Design principles

1. **Clean Architecture** — domain policies never import infrastructure (MT5, filesystem, network).
2. **DDD where it pays** — bounded contexts ≈ domains below; ubiquitous language in contracts (`FeatureSet`, `Decision`, `TradePlan`).
3. **Single responsibility** — one domain, one job, one primary output type.
4. **Interface-driven** — domains depend on Protocols/ports, not concretes.
5. **No circular dependencies** — enforce a DAG; only `apps/` composes the graph.
6. **No hidden global state** — pass `Settings`, `Clock`, `PortfolioState` explicitly.
7. **Testability** — Decision/Risk unit-testable with fixture DTOs; no MT5 required.
8. **Causal by default** — FeatureFactory and live inference use closed-bar as-of semantics.

---

## 3. System domains (bounded contexts)

```
shared (kernel contracts)
    ↑
market_data → features → models → decision → risk → execution
                ↘ labels ↗              ↑
                                   portfolio
monitoring (observes events)
analytics (offline evaluation; never on live hot path)
apps/* (composition roots only)
```

### 3.1 Market Data

**Owns:** load OHLCV, live streaming, historical access, cleaning, resampling.  
**Forbids:** ML, features, decisions, orders.  
**Output:** `BarSeries`, `QuoteTick`.

**Why separate:** Venue quirks (MT5 session gaps, symbol suffixes, missing bars) must not leak into feature or model code. Swap store (CSV → Parquet → DB) without touching Features.

### 3.2 Feature Engineering

**Owns:** indicators, interactions, pipeline, factory, normalization, selection.  
**Forbids:** labels, models, broker.  
**Output:** `FeatureSet` (matrix + schema + `recipe_id` + as-of timestamp).

**Why separate:** Same bars can feed many strategies/models. Features must remain pure functions of market data with an explicit causal contract.

### 3.3 Label Engine

**Owns:** triple barrier, meta-label, future return, dynamic labels.  
**Forbids:** models, risk, execution.  
**Output:** `LabelSet`.

**Why separate:** Label definitions change more often than feature recipes during research; live path does not need labels at all. Train-only domain on the hot path.

### 3.4 Machine Learning

**Owns:** trainer, predictor, calibration, registry, loader, inference.  
**Forbids:** position sizing, broker I/O, portfolio ledger.  
**Output:** `Prediction` (`p_raw`, `p_calibrated`, `direction`, optional `expected_return`, `confidence`).

**Why separate:** Model family (LightGBM → Transformer) must be swappable. Calibration is an ML concern (probability quality), not a trading-policy concern.

**Plugin rule:** New family = implement `ModelTrainer` + `ModelPredictor`; register under `ModelKey`. No other domain changes.

### 3.5 Decision Engine

**Owns:** threshold, regime filter, session filter, spread filter, ranking, conflict resolver, trade limiter.  
**Forbids:** knowledge of how ML works, lot math, MT5.  
**Input:** one or more `Prediction` + `DecisionContext`.  
**Output:** `Decision` (action, reason, applied filters, rank score, sizing *hints* only).

**Why separate:** “Should we trade?” is policy, not probability math. Multi-timeframe / multi-model **fusion lives here** (vote, veto, weighted rank) — not inside a single concatenated feature matrix.

### 3.6 Risk Engine

**Owns:** position size, confidence size, risk-per-trade, daily loss limit, drawdown limit, exposure, correlation; future Kelly / dynamic risk.  
**Forbids:** feature recipes, broker calls.  
**Input:** `Decision` + `PortfolioState`.  
**Output:** `TradePlan` or typed reject.

**Why separate:** Size depends on equity and limits, not on model internals. Risk can reject a valid Decision.

### 3.7 Portfolio Engine

**Owns:** current positions, exposure, margin, open trades, portfolio risk, cross-symbol risk.  
**Forbids:** feature generation, training.  
**Output:** `PortfolioState` snapshots; applies `Fill` / `OrderEvent`.

**Why separate:** Even with only XAUUSD today, live multi-symbol needs a ledger that Execution does not own. Portfolio is the source of truth for “what are we in?”.

### 3.8 Execution Engine

**Owns:** broker adapters (MT5), retry, slippage model hooks, spread at send time, order manager, order lifecycle.  
**Forbids:** feature engineering, labels, ML internals.  
**Input:** `TradePlan` → `OrderIntent`.  
**Output:** `OrderEvent` / `Fill`.

**Why separate:** Retries and venue semantics change per broker; trading policy must not.

### 3.9 Monitoring

**Owns:** trades, PnL, equity, drawdown, latency, spread, execution status, alerts.  
**Forbids:** mutating trading decisions (except feeding a kill-switch *signal* consumed by apps/risk supervisor).  
**Pattern:** subscribe to domain events; never command Features/Models.

### 3.10 Analytics

**Owns:** performance reports, feature importance, SHAP, calibration reports, walk-forward, sensitivity, stress tests.  
**Forbids:** placement on the live order path.  
**Pattern:** offline consumer of artifacts and event logs.

---

## 4. Folder structure (target)

```
mlfrog/
  shared/                 # IDs, DTOs, errors, Clock, EventBus protocols
  market_data/            # rename from data/ — loaders, clean, resample, ports
    adapters/             # MT5 feed, file store (infra)
  features/
    indicators/
    interaction/
    pipeline/
    factory/
  labels/
    triple_barrier/
    meta_label/
  models/
    signal/               # family plugins (lgbm, xgb, ...)
    calibration/
    registry/
  decision/
    threshold/
    regime_guard/
    session/
    spread/
    ranking/
    conflict/
    limiter/
  risk/
    position_sizing/
    confidence_sizing/
    limits/
    exposure/
  portfolio/
  execution/
    paper/
    live_mt5/
    sim/                  # backtest fills
    order_manager/
  monitoring/
  analytics/              # merge reports/ + validation/
    walk_forward/
    sealed/
    sensitivity/
    reports/
  apps/                   # ONLY composition roots
    train/
    backtest/
    paper/
    live/
  configs/                # yaml secrets (gitignored config.yaml)
  settings/               # typed StrategySettings / PlatformSettings
  artifacts/              # gitignored blobs
```

**Design decision — `apps/` only composition:** Domains never “new up” MT5. This is what keeps the dependency rule enforceable and tests fast.

**Design decision — `simulation` is not a domain:** Backtest is an app mode using `execution/sim` + a historical clock. Treating “backtest” as the center of the universe produces a god object.

**Migration from current skeleton:** Keep `decision/`, `risk/`, `models/calibration/`, `execution/ports.py`. Introduce `shared/`, rename `data` → `market_data`, fold `reports`+`validation` → `analytics`, add `monitoring/`.

---

## 5. Dependency diagram

```
                    ┌──────────── apps/* ────────────┐
                    │  wires ports + use-cases       │
                    └───────────────┬────────────────┘
                                    │ depends on
        ┌───────────┬───────────┬───┴───┬───────────┬──────────┐
        ▼           ▼           ▼       ▼           ▼          ▼
  market_data   features     models  decision     risk    execution
        ▲           │           │       │           │          │
        │           ▼           │       │           ▼          │
        │        labels ────────┘       │      portfolio ◄─────┘ (events)
        │                               │
        └────────── shared ◄────────────┴──────────────────────
        
  monitoring ──subscribes──► events (no inbound deps from domains)
  analytics  ──reads──► artifacts (offline; not imported by hot path)
```

### Allowed import matrix (summary)

| From → | shared | market_data | features | labels | models | decision | risk | portfolio | execution |
|--------|--------|-------------|----------|--------|--------|----------|------|-----------|-----------|
| features | Y | Y | — | N | N | N | N | N | N |
| labels | Y | Y | N* | — | N | N | N | N | N |
| models | Y | N | Y | Y | — | N | N | N | N |
| decision | Y | N | N | N | N | — | N | N | N |
| risk | Y | N | N | N | N | Y | — | Y | N |
| portfolio | Y | N | N | N | N | N | N | — | N |
| execution | Y | N | N | N | N | N | DTO only | N | — |
| apps | Y | Y | Y | Y | Y | Y | Y | Y | Y |

\* Labels may align on timestamps; must not import feature *recipes*.  
DTO only = Execution accepts `TradePlan` / `OrderIntent` types from shared (or thin risk API types), not risk policy internals.

**Arrow direction:** “depends on / may import”. No cycles. Dashed/forbidden edges are lintable later (import-linter).

---

## 6. Class / module responsibilities

| Component | Responsibility |
|---|---|
| `BarRepository` / feed adapter | Implement `MarketDataPort` |
| `FeaturePipeline` | Deterministic `bars → FeatureSet` |
| `FeatureFactory` | Named recipes (`v3_long`, …) composing indicators |
| `LabelEngine` | `bars + LabelConfig → LabelSet` |
| `ModelTrainer` | Fit artifact from FeatureSet+LabelSet |
| `ModelPredictor` | FeatureSet → Prediction |
| `Calibrator` | Probability mapping (e.g. Platt) |
| `ModelRegistry` | Persist/load by `ModelKey` |
| `DecisionEngine` | Filter chain + multi-model fusion → Decision |
| `RiskEngine` | Decision + PortfolioState → TradePlan \| Reject |
| `PortfolioService` | Apply fills; expose snapshot |
| `OrderManager` | Order lifecycle state machine |
| `ExecutionPort` adapters | paper / sim / mt5 |
| `MetricsCollector` | Monitoring sinks |
| `ReportGenerators` | Analytics offline jobs |
| `*App` in `apps/` | Parse settings, construct graph, run loop |

---

## 7. Interfaces (ports)

```text
MarketDataPort
  historical_bars(instrument, timeframe, start, end) -> BarSeries
  closed_bars(instrument, timeframe, lookback) -> BarSeries
  stream_ticks(instrument) -> Iterator[QuoteTick]          # live

FeaturePipeline
  transform(bars: Mapping[Timeframe, BarSeries], as_of) -> FeatureSet

LabelEngine
  label(bars, config: LabelConfig) -> LabelSet

ModelTrainer
  fit(features: FeatureSet, labels: LabelSet, key: ModelKey) -> ModelArtifact

ModelPredictor
  predict(features: FeatureSet, key: ModelKey) -> Prediction

Calibrator
  fit(p_raw, y) -> CalibratorArtifact
  transform(p_raw) -> p_calibrated

ModelRegistry
  register(key, artifact, meta)
  get(key) -> ModelArtifact

DecisionPolicy / DecisionEngine
  decide(predictions: Sequence[Prediction], ctx: DecisionContext) -> Decision

RiskPolicy / RiskEngine
  plan(decision: Decision, portfolio: PortfolioState, cfg) -> TradePlan | Reject

PortfolioService
  snapshot() -> PortfolioState
  apply(event: OrderEvent | Fill) -> PortfolioState

ExecutionPort
  submit(intent: OrderIntent) -> OrderAck
  cancel(order_id) -> CancelAck
  reconcile() -> Sequence[OrderEvent]

Clock
  now() -> Timestamp                      # injectable for replay

EventBus (optional)
  publish(event) / subscribe(handler)     # monitoring
```

**Design decision — Decision does not depend on ModelPredictor:** Apps call Predictor, then pass `Prediction` DTOs into Decision. That keeps Decision free of ML imports and makes multi-model fusion trivial.

---

## 8. Object contracts (minimum)

| Object | Meaning |
|---|---|
| `InstrumentId` | symbol + venue (+ contract specs ref) |
| `Timeframe` | M1…MN enum/value object |
| `StrategyId` | stable strategy name/version namespace |
| `ModelKey` | `(StrategyId, InstrumentId, Timeframe, Family, Version)` |
| `BarSeries` | OHLCV; index = **bar close time**; timezone explicit |
| `FeatureSet` | `X`, column schema, `recipe_id`, `as_of`, row instrument/time keys |
| `LabelSet` | `y`, label type, barrier params, alignment keys |
| `Prediction` | `model_key`, `p_raw`, `p_calibrated`, `direction`, optional `E[R]`, `confidence` |
| `DecisionContext` | session, spread, open position flags, portfolio snapshot ref, clock |
| `Decision` | `action`, `reason`, filters fired, `rank_score`, optional size hints |
| `TradePlan` | side, qty, SL/TP, risk currency, constraints checked |
| `OrderIntent` | broker-agnostic request derived from TradePlan |
| `Fill` / `OrderEvent` | lifecycle for Portfolio + Monitoring |
| `PortfolioState` | positions, equity, margin, exposure map |
| `Reject` | typed code + message (threshold, regime, risk limit, …) |

Immutability preferred (`frozen` DTOs). Crossing a domain boundary with a bare `DataFrame` is an anti-pattern unless wrapped as FeatureSet/LabelSet/BarSeries.

---

## 9. Data flow

### Train
`MarketData → Features → Labels → Models(fit+calibrate) → Registry → Analytics(WF/sealed)`

### Backtest / Paper / Live (shared spine)
```
MarketData (hist | live)
  → Features (closed bar)
  → Models.predict → Prediction(+)
  → DecisionEngine → Decision
  → RiskEngine → TradePlan
  → ExecutionPort → OrderEvent/Fill
  → Portfolio.apply
  → Monitoring (side)
```

Analytics consumes logs/artifacts **offline**.

**Invariant:** `Prediction → Decision → TradePlan → OrderIntent` is identical across backtest/paper/live.

**Live constraint:** signal on **closed bars only** unless a strategy explicitly defines tick logic in its own app loop (still through the same contracts).

---

## 10. Multi-timeframe & multi-strategy

### Multi-timeframe
- H1, H4, D1 ⇒ **independent** `ModelKey`s and usually independent Feature recipes.
- Each produces a `Prediction`.
- `DecisionEngine` combines (example policies): H4 veto, weighted rank, agreement gate.
- **Do not** make “concat all TF features into one model” the architectural default — it couples TFs, complicates leakage control, and blocks per-TF replacement.

### Multi-strategy
- Each `StrategyId` owns: feature recipe(s), label config (train), model key(s), decision profile, risk profile.
- Portfolio/Execution already multi-position; ConflictResolver arbitrates same-symbol collisions.

---

## 11. Future extensibility checklist

| Change | Touch |
|---|---|
| New model family | `models/` plugin + registry |
| New timeframe model | new `ModelKey` + Decision fusion rule |
| New strategy | settings + recipes + keys + profiles |
| New broker | `execution` adapter |
| New risk model | `RiskPolicy` impl |
| New data store | `market_data` adapter |
| Kill switch | Monitoring signal → apps/live supervisor or Risk hard limit |

---

## 12. Anti-patterns to avoid

1. **God backtester** — one engine that owns features+ML+risk+broker.  
2. **Features know the model** — column lists living inside LightGBM modules.  
3. **Decision imports LightGBM / sklearn**.  
4. **Risk calls MT5**.  
5. **Merge all timeframes into one model by default**.  
6. **Mutable global config / singletons** for strategy state.  
7. **Circular package imports** between domains.  
8. **Notebooks as source of truth** for production paths.  
9. **Different math for paper vs live** (duplicate engines).  
10. **Hidden look-ahead** in feature factories.  
11. **Analytics on the live hot path**.  
12. **Passing raw DataFrames across domains without schema**.  
13. **Portfolio logic inside Execution**.  
14. **Sizing inside Decision** (hints OK; final qty is Risk).

---

## 13. Relation to settled research constants

`settings/strategy.py` currently holds settled LONG v3 constants (thr 0.51, barriers, Z_BLOCK=4.0, conservative confidence sizing). Architecturally these become **versioned `StrategySettings`** loaded by apps — **values must not be retuned as part of structural work**. Structure changes; economics stay until research says otherwise.

---

## 14. Implementation order (when coding starts)

1. `shared` contracts + Clock  
2. Expand Decision filter chain behind stable `Decision`  
3. Risk `TradePlan` + Portfolio snapshot  
4. Execution ports: sim / paper / mt5  
5. Market data ports + FeatureSet factory rebuild  
6. Labels + ModelRegistry + Predictor  
7. apps: backtest → paper → live  
8. Monitoring + Analytics migration  

No domain should be “finished” before its **output contract** is frozen and tested.
