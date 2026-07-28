"""Sprint XX — Feature Selection & Ablation (research only).

Fixed: dataset, labels, LightGBM params, WF protocol, entry/exit/risk.
Only the feature *subset* changes. No production / FE code edits.

Phases:
  ranking | single | group | forward | backward | compare | all

Example:
  python apps/run_feature_selection_ablation.py --apply-schema
  python apps/run_feature_selection_ablation.py --phase all
  python apps/run_feature_selection_ablation.py --phase single --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from apps.run_rolling_walkforward import (
    TOP_PCT,
    WFWindow,
    apply_trail,
    build_test_entries,
    build_windows,
    train_frozen,
    _slice_year,
    _slice_years,
)
from apps.report_feature_library_research import (
    CATEGORY,
    categorize,
    corr_redundancy,
    frozen_gain_importance,
    mean_abs_shap,
    mi_scores,
    permutation_importance,
)
from production.db.fs_writer import FeatureSelectionWriter
from simulation.wf.sim import _feat_names, _load_side, load_h1, prepare_market, regime_thresholds, run_portfolio

logger = logging.getLogger(__name__)
OUT = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprintXX_feature_selection"
FROZEN = _ROOT / "artifacts" / "models" / "frozen"
STATE_PATH = OUT / "state.json"

# Align with sprint brief; unknown categories still appear via categorize()
GROUPS = (
    "candle",
    "session",
    "momentum",
    "volatility",
    "trend",
    "context",
    "statistical",
    "liquidity",
    "market_structure",
    "mtf",
)


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("config root must be mapping")
    return data


def _save_state(state: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def binary_label(y: pd.Series) -> np.ndarray:
    return (y.astype(float).to_numpy() > 0).astype(int)


def library_features(long_df: pd.DataFrame) -> list[str]:
    """Same intersection rolling WF uses (frozen long ∩ dataset)."""
    return _feat_names(long_df)


def run_wf_subset(
    *,
    feat: list[str],
    long_df: pd.DataFrame,
    short_df: pd.DataFrame,
    h1: pd.DataFrame,
    mkt: dict,
    windows: list[WFWindow],
    vol_lo: float,
    vol_hi: float,
    starting: float = 80.0,
    leverage: float = 500.0,
) -> dict[str, Any]:
    """One full rolling WF with a fixed feature subset. Returns combined + diagnostics."""
    if len(feat) < 3:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "final_equity": starting,
            "roc_auc": float("nan"),
            "avg_pr": float("nan"),
            "skipped": True,
            "reason": "too_few_features",
        }

    panels: list[pd.DataFrame] = []
    roc_scores: list[float] = []
    pr_scores: list[float] = []
    shared = [f for f in feat if f in long_df.columns and f in short_df.columns]
    if len(shared) < 3:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "final_equity": starting,
            "roc_auc": float("nan"),
            "avg_pr": float("nan"),
            "skipped": True,
            "reason": "too_few_shared_features",
        }

    for w in windows:
        boosters: dict[str, lgb.Booster] = {}
        for side, df in (("long", long_df), ("short", short_df)):
            train = _slice_years(df, w.train_start, w.train_end)
            val = _slice_year(df, w.val_year)
            if len(train) < 500 or len(val) < 50:
                continue
            booster, _meta = train_frozen(train, val, shared)
            boosters[side] = booster
            test = _slice_year(df, w.test_year)
            if not test.empty:
                y = binary_label(test["label"])
                p = booster.predict(test[shared], num_iteration=booster.best_iteration)
                if y.min() != y.max():
                    roc_scores.append(float(roc_auc_score(y, p)))
                    pr_scores.append(float(average_precision_score(y, p)))

        if not boosters:
            continue
        entries = build_test_entries(
            long_df=long_df,
            short_df=short_df,
            h1=h1,
            feat=shared,
            window=w,
            boosters=boosters,
            top_pct=TOP_PCT,
            vol_lo=vol_lo,
            vol_hi=vol_hi,
        )
        panel = apply_trail(entries, mkt)
        if not panel.empty:
            panels.append(panel)

    if not panels:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "final_equity": starting,
            "roc_auc": float(np.mean(roc_scores)) if roc_scores else float("nan"),
            "avg_pr": float(np.mean(pr_scores)) if pr_scores else float("nan"),
            "skipped": True,
            "reason": "no_panels",
        }

    all_panel = pd.concat(panels, ignore_index=True).sort_values("timestamp")
    _traded, metrics = run_portfolio(all_panel, starting=starting, ruin_stop=True, leverage=leverage)
    pf = metrics["profit_factor"]
    return {
        "n_trades": int(metrics["n_trades"]),
        "win_rate": float(metrics["win_rate"]),
        "profit_factor": float(pf) if np.isfinite(pf) else 0.0,
        "total_return": float(metrics["total_return"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "final_equity": float(metrics["final_equity"]),
        "roc_auc": float(np.mean(roc_scores)) if roc_scores else float("nan"),
        "avg_pr": float(np.mean(pr_scores)) if pr_scores else float("nan"),
        "skipped": False,
        "n_features": len(shared),
    }


def score_key(m: dict[str, Any]) -> tuple[float, float, float]:
    """Primary sort: PF, return, -drawdown."""
    return (float(m.get("profit_factor") or 0.0), float(m.get("total_return") or 0.0), -float(m.get("max_drawdown") or 1.0))


def better(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True if a improves on b (WF PF first)."""
    return score_key(a) > score_key(b)


def persist_exp(
    db: FeatureSelectionWriter,
    *,
    run_id: str,
    phase: str,
    experiment_id: str,
    features: list[str],
    metrics: dict[str, Any],
    removed_feature: str | None = None,
    removed_group: str | None = None,
) -> None:
    db.upsert_experiment(
        {
            "run_id": run_id,
            "phase": phase,
            "experiment_id": experiment_id,
            "features_json": features,
            "n_features": len(features),
            "removed_feature": removed_feature,
            "removed_group": removed_group,
            "n_trades": metrics.get("n_trades"),
            "win_rate": metrics.get("win_rate"),
            "profit_factor": metrics.get("profit_factor"),
            "total_return": metrics.get("total_return"),
            "max_drawdown": metrics.get("max_drawdown"),
            "final_equity": metrics.get("final_equity"),
            "roc_auc": metrics.get("roc_auc"),
            "avg_pr": metrics.get("avg_pr"),
            "metrics": metrics,
        }
    )


def phase_ranking(long_df: pd.DataFrame, feat: list[str], out_dir: Path) -> pd.DataFrame:
    """Step 1–2: multi-method ranking + redundancy candidates (no retrain)."""
    booster = lgb.Booster(model_file=str(FROZEN / "primary_long.txt"))
    y = binary_label(long_df["label"])
    X = long_df[feat].apply(pd.to_numeric, errors="coerce")
    ok = X.notna().mean(axis=1) > 0.8
    X = X.loc[ok].fillna(X.loc[ok].median(numeric_only=True))
    y_ok = y[ok.to_numpy()]

    mi = mi_scores(X, y_ok).rename(columns={"mutual_info": "mi"})
    shap = mean_abs_shap(booster, X)
    perm = permutation_importance(booster, X, y_ok)
    gain = frozen_gain_importance(booster)
    gain = gain[gain["feature"].isin(feat)]
    red = corr_redundancy(X, thr=0.90)

    rank = (
        pd.DataFrame({"feature": feat})
        .merge(mi, on="feature", how="left")
        .merge(shap, on="feature", how="left")
        .merge(perm[["feature", "perm_auc_drop"]], on="feature", how="left")
        .merge(gain[["feature", "gain", "split"]], on="feature", how="left")
    )
    rank["category"] = rank["feature"].map(categorize)
    for col in ("mi", "mean_abs_shap", "perm_auc_drop", "gain"):
        s = rank[col].fillna(0.0)
        rank[f"{col}_n"] = (s - s.min()) / (s.max() - s.min() + 1e-12)
    rank["importance"] = (
        0.35 * rank["mean_abs_shap_n"]
        + 0.35 * rank["perm_auc_drop_n"]
        + 0.20 * rank["gain_n"]
        + 0.10 * rank["mi_n"]
    )
    rank = rank.sort_values("importance", ascending=False).reset_index(drop=True)
    rank["rank"] = np.arange(1, len(rank) + 1)

    # redundancy: mark weaker twin
    strength = {r.feature: float(r.importance) for r in rank.itertuples()}
    drop_cand = set()
    if not red.empty:
        for _, row in red.iterrows():
            a, b = row["feature_a"], row["feature_b"]
            if a in strength and b in strength:
                drop_cand.add(a if strength[a] <= strength[b] else b)
    # low importance tail
    drop_cand |= set(rank.tail(max(5, len(rank) // 5))["feature"])
    rank["candidate_remove"] = rank["feature"].isin(drop_cand)

    rank.to_csv(out_dir / "feature_ranking.csv", index=False)
    rank.to_csv(out_dir / "feature_importance.csv", index=False)
    red.to_csv(out_dir / "redundancy.csv", index=False)
    return rank


METRIC_COLS = ("n_trades", "win_rate", "profit_factor", "total_return", "max_drawdown", "roc_auc", "avg_pr")


def export_from_db(db: FeatureSelectionWriter, run_id: str, out_dir: Path) -> None:
    """Regenerate all phase CSVs from research.fs_experiments (resume-safe source of truth)."""
    rows = db.fetch_experiments(run_id)
    if not rows:
        print("export: no experiments in DB for", run_id)
        return
    df = pd.DataFrame(rows)
    df["skipped"] = df["metrics"].apply(lambda m: bool((m or {}).get("skipped")))

    def sub(phase: str) -> pd.DataFrame:
        return df[df["phase"] == phase].copy()

    s = sub("single")
    if not s.empty:
        s["category"] = s["removed_feature"].map(categorize)
        s[["removed_feature", "category", *METRIC_COLS]].to_csv(out_dir / "single_feature_ablation.csv", index=False)
    g = sub("group")
    if not g.empty:
        g[["removed_group", "n_features", *METRIC_COLS]].to_csv(out_dir / "group_ablation.csv", index=False)
    f = sub("forward")
    if not f.empty:
        f[["experiment_id", "n_features", *METRIC_COLS]].to_csv(out_dir / "forward_selection.csv", index=False)
    b = sub("backward")
    if not b.empty:
        b[["experiment_id", "removed_feature", "n_features", *METRIC_COLS]].to_csv(out_dir / "backward_elimination.csv", index=False)
    c = sub("compare")
    if not c.empty:
        c[["experiment_id", "n_features", *METRIC_COLS, "final_equity"]].sort_values(
            ["profit_factor", "total_return", "max_drawdown"], ascending=[False, False, True]
        ).to_csv(out_dir / "set_comparison.csv", index=False)

    # global best: any experiment with a real portfolio (>=100 trades), ranked PF/ret/dd
    real = df[(~df["skipped"]) & (df["n_trades"].fillna(0) >= 100)].copy()
    if not real.empty:
        real = real.sort_values(["profit_factor", "total_return", "max_drawdown"], ascending=[False, False, True])
        top = real.iloc[0]
        feats = top["features_json"] or []
        pd.DataFrame(
            {
                "source": [f"{top['phase']}:{top['experiment_id']}"],
                "n_features": [len(feats)],
                "features": ["|".join(feats)],
                **{k: [top[k]] for k in METRIC_COLS},
            }
        ).to_csv(out_dir / "best_feature_set.csv", index=False)
        (out_dir / "best_feature_set.json").write_text(
            json.dumps({"source": f"{top['phase']}:{top['experiment_id']}", "features": list(feats)}, indent=2),
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Feature selection & ablation (WF protocol fixed)")
    p.add_argument("--config", type=Path, default=_ROOT / "configs" / "config.yaml")
    p.add_argument("--phase", default="all", choices=("ranking", "single", "group", "forward", "backward", "compare", "all"))
    p.add_argument("--apply-schema", action="store_true")
    p.add_argument("--keep-history", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip experiments already in state/DB for this run_id")
    p.add_argument("--run-id", default=None)
    p.add_argument("--max-single", type=int, default=0, help="Limit single ablations (0=all); useful for smoke")
    p.add_argument("--features-file", type=Path, default=None, help="JSON list of features to use as the library (loop iterations)")
    p.add_argument("--export-only", action="store_true", help="Only regenerate CSVs from DB for run-id and exit")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(args.config)
    dsn = (cfg.get("paper_trading") or {}).get("postgres_dsn")
    db = FeatureSelectionWriter(str(dsn) if dsn else None)

    if args.apply_schema:
        if not db.enabled:
            raise SystemExit("--apply-schema needs postgres_dsn")
        db.apply_schema((_ROOT / "sql" / "research_schema.sql").read_text(encoding="utf-8"))
        print("schema applied: research.fs_*")
        return 0

    state = _load_state() if args.resume else {}
    run_id = args.run_id or state.get("run_id") or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    )
    done: set[str] = set(state.get("done_experiments") or [])

    if db.enabled and not args.keep_history and not args.resume:
        n = db.wipe_all()
        if n:
            print(f"fs db fresh: wiped {n} prior run(s)")

    print(f"fs_run_id={run_id} phase={args.phase}")
    db.upsert_run({"run_id": run_id, "status": "running", "baseline_pf": None, "baseline_return": None, "baseline_dd": None, "notes": None})

    if args.export_only:
        export_from_db(db, run_id, OUT)
        print(f"exported CSVs from DB for {run_id} -> {OUT}")
        return 0

    print("Loading datasets…")
    long_df = _load_side("long")
    short_df = _load_side("short")
    feat_all = library_features(long_df)
    if args.features_file:
        wanted = json.loads(args.features_file.read_text(encoding="utf-8"))
        feat_all = [f for f in feat_all if f in set(wanted)]
        print(f"features-file: restricted library to {len(feat_all)}")
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    mkt = prepare_market(h1)
    vol_lo, vol_hi = regime_thresholds(long_df)
    windows = build_windows()
    print(f"library={len(feat_all)} windows={len(windows)}")

    def wf(features: list[str]) -> dict[str, Any]:
        return run_wf_subset(
            feat=list(features),
            long_df=long_df,
            short_df=short_df,
            h1=h1,
            mkt=mkt,
            windows=windows,
            vol_lo=vol_lo,
            vol_hi=vol_hi,
        )

    def record(phase: str, eid: str, features: list[str], metrics: dict[str, Any], **kw: Any) -> None:
        persist_exp(db, run_id=run_id, phase=phase, experiment_id=eid, features=features, metrics=metrics, **kw)
        done.add(f"{phase}:{eid}")
        st = {"run_id": run_id, "done_experiments": sorted(done), "phase": phase, "updated_at": datetime.now(timezone.utc).isoformat()}
        _save_state(st)
        print(
            f"  [{phase}/{eid}] n={len(features)} trades={metrics.get('n_trades')} "
            f"pf={metrics.get('profit_factor'):.3f} ret={metrics.get('total_return'):+.1%} "
            f"dd={metrics.get('max_drawdown'):.1%}"
        )

    # ---- STEP 1–2 ranking ----
    if args.phase in ("ranking", "all"):
        print("\n=== STEP 1–2 ranking + redundancy ===")
        rank = phase_ranking(long_df, feat_all, OUT)
        print(f"wrote {OUT / 'feature_ranking.csv'} candidates_remove={int(rank['candidate_remove'].sum())}")
    else:
        rank_path = OUT / "feature_ranking.csv"
        if not rank_path.exists():
            rank = phase_ranking(long_df, feat_all, OUT)
        else:
            rank = pd.read_csv(rank_path)

    ordered = list(rank.sort_values("importance", ascending=False)["feature"])
    # keep only library feats
    ordered = [f for f in ordered if f in feat_all]

    # ---- baseline ----
    if args.resume and "compare:current_library" in done:
        print("\n=== BASELINE (recompute for compare keys) ===")
        baseline = wf(feat_all)
    else:
        print("\n=== BASELINE current library ===")
        baseline = wf(feat_all)
        record("compare", "current_library", feat_all, baseline)
        db.upsert_run(
            {
                "run_id": run_id,
                "status": "running",
                "baseline_pf": baseline["profit_factor"],
                "baseline_return": baseline["total_return"],
                "baseline_dd": baseline["max_drawdown"],
                "notes": "baseline=current WF feature intersection",
            }
        )

    best_overall = {"features": list(feat_all), "metrics": baseline, "source": "current_library"}

    # ---- STEP 3–4 single ablation ----
    if args.phase in ("single", "all"):
        print("\n=== STEP 3–4 single-feature ablation ===")
        rows = []
        # Prefer ablating remove-candidates first, then the rest
        cand = list(rank.loc[rank["candidate_remove"], "feature"])
        rest = [f for f in ordered if f not in cand]
        abl_order = cand + rest
        if args.max_single > 0:
            abl_order = abl_order[: args.max_single]
        candidate_remove_confirmed = []
        for i, f in enumerate(abl_order, 1):
            eid = f"drop_{f}"
            if args.resume and f"single:{eid}" in done:
                print(f"  skip {eid}")
                continue
            subset = [x for x in feat_all if x != f]
            print(f"single {i}/{len(abl_order)} drop={f}")
            m = wf(subset)
            record("single", eid, subset, m, removed_feature=f)
            rows.append({"removed_feature": f, "category": categorize(f), **{k: m[k] for k in ("n_trades", "win_rate", "profit_factor", "total_return", "max_drawdown", "roc_auc", "avg_pr")}})
            if better(m, baseline):
                candidate_remove_confirmed.append(f)
                print(f"    -> IMPROVES vs baseline (candidate_remove)")
                if better(m, best_overall["metrics"]):
                    best_overall = {"features": subset, "metrics": m, "source": f"single_drop_{f}"}
        pd.DataFrame(rows).to_csv(OUT / "single_feature_ablation.csv", index=False)
        pd.DataFrame({"feature": candidate_remove_confirmed}).to_csv(OUT / "candidate_remove.csv", index=False)
        print(f"confirmed removals that improve WF: {candidate_remove_confirmed}")

    # ---- STEP 5 group ablation ----
    if args.phase in ("group", "all"):
        print("\n=== STEP 5 group ablation ===")
        rows = []
        for g in GROUPS:
            eid = f"drop_group_{g}"
            if args.resume and f"group:{eid}" in done:
                continue
            members = [f for f in feat_all if categorize(f) == g]
            if not members:
                continue
            subset = [f for f in feat_all if f not in members]
            print(f"group drop={g} (-{len(members)}) keep={len(subset)}")
            m = wf(subset)
            record("group", eid, subset, m, removed_group=g)
            rows.append({"removed_group": g, "n_removed": len(members), **{k: m[k] for k in ("n_trades", "win_rate", "profit_factor", "total_return", "max_drawdown", "roc_auc", "avg_pr")}})
            if better(m, best_overall["metrics"]):
                best_overall = {"features": subset, "metrics": m, "source": f"group_drop_{g}"}
        pd.DataFrame(rows).to_csv(OUT / "group_ablation.csv", index=False)

    # ---- STEP 6 forward selection (ranked path: top → next until no gain) ----
    if args.phase in ("forward", "all"):
        print("\n=== STEP 6 forward selection (importance order) ===")
        # ponytail: path along ranking, not full greedy O(n²) search — "tambahkan satu-persatu berikutnya"
        # seed with top-3: run_wf_subset needs >=3 features to produce real metrics
        selected: list[str] = list(ordered[:3])
        forward_rows = []
        current_m: dict[str, Any] | None = wf(selected) if selected else None
        if current_m is not None:
            record("forward", f"fwd_seed_n{len(selected)}", selected, current_m)
            print(f"  forward seed n={len(selected)} pf={current_m['profit_factor']:.3f}")
        for step, f in enumerate(ordered[3:], 1):
            trial = selected + [f]
            eid = f"fwd_n{len(trial)}_{f}"
            if args.resume and f"forward:{eid}" in done and current_m is not None:
                selected.append(f)
                continue
            m = wf(trial)
            record("forward", eid, trial, m)
            forward_rows.append(
                {
                    "step": step,
                    "added": f,
                    "accepted": False,
                    "n_features": len(trial),
                    **{k: m[k] for k in ("n_trades", "win_rate", "profit_factor", "total_return", "max_drawdown", "roc_auc", "avg_pr")},
                }
            )
            if current_m is None or better(m, current_m):
                selected.append(f)
                current_m = m
                forward_rows[-1]["accepted"] = True
                print(f"  forward accept +{f} n={len(selected)} pf={current_m['profit_factor']:.3f}")
                if better(current_m, best_overall["metrics"]):
                    best_overall = {"features": list(selected), "metrics": current_m, "source": "forward"}
            else:
                print(f"  forward stop at +{f} (no improvement); optimal n={len(selected)}")
                break
        pd.DataFrame(forward_rows).to_csv(OUT / "forward_selection.csv", index=False)
        (OUT / "forward_best_features.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    # ---- STEP 7 backward elimination (drop lowest-importance first while PF holds) ----
    if args.phase in ("backward", "all"):
        print("\n=== STEP 7 backward elimination (lowest importance first) ===")
        # ponytail: remove by ascending importance; stop when removal hurts — O(n) not O(n²)
        selected = list(feat_all)
        current_m = baseline
        backward_rows = []
        # drop candidates from weakest → strongest
        drop_order = list(reversed(ordered))
        for step, f in enumerate(drop_order, 1):
            if f not in selected or len(selected) <= 5:
                break
            trial = [x for x in selected if x != f]
            eid = f"bwd_n{len(trial)}_drop_{f}"
            if args.resume and f"backward:{eid}" in done:
                continue
            m = wf(trial)
            record("backward", eid, trial, m, removed_feature=f)
            # keep removal if not worse than current (preserve performance)
            not_worse = score_key(m) >= score_key(current_m)
            backward_rows.append(
                {
                    "step": step,
                    "removed": f,
                    "accepted": bool(not_worse),
                    "n_features": len(trial),
                    **{k: m[k] for k in ("n_trades", "win_rate", "profit_factor", "total_return", "max_drawdown", "roc_auc", "avg_pr")},
                }
            )
            if not_worse:
                selected = trial
                current_m = m
                print(f"  backward drop -{f} n={len(selected)} pf={current_m['profit_factor']:.3f}")
                if better(current_m, best_overall["metrics"]):
                    best_overall = {"features": list(selected), "metrics": current_m, "source": "backward"}
            else:
                print(f"  backward keep {f} (removal hurts); stop. n={len(selected)}")
                break
        pd.DataFrame(backward_rows).to_csv(OUT / "backward_elimination.csv", index=False)
        (OUT / "backward_best_features.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    # ---- STEP 8 compare ----
    if args.phase in ("compare", "all"):
        print("\n=== STEP 8 compare ===")
        # Ensure we have forward/backward artifacts
        sets = [("current_library", feat_all)]
        fwd_path = OUT / "forward_best_features.json"
        bwd_path = OUT / "backward_best_features.json"
        if fwd_path.exists():
            sets.append(("forward_selection", json.loads(fwd_path.read_text(encoding="utf-8"))))
        if bwd_path.exists():
            sets.append(("backward_elimination", json.loads(bwd_path.read_text(encoding="utf-8"))))
        # also best_overall source
        if best_overall["source"] not in {s for s, _ in sets}:
            sets.append((best_overall["source"], best_overall["features"]))

        compare_rows = []
        for name, feats in sets:
            if not feats:
                continue
            eid = f"final_{name}"
            m = wf(feats) if name != "current_library" else baseline
            if name != "current_library":
                record("compare", eid, feats, m)
            else:
                # already recorded
                m = baseline
            compare_rows.append(
                {
                    "set_name": name,
                    "n_features": len(feats),
                    "features": "|".join(feats),
                    **{k: m[k] for k in ("n_trades", "win_rate", "profit_factor", "total_return", "max_drawdown", "final_equity", "roc_auc", "avg_pr")},
                }
            )
            if better(m, best_overall["metrics"]):
                best_overall = {"features": list(feats), "metrics": m, "source": name}

        cmp = pd.DataFrame(compare_rows).sort_values(
            ["profit_factor", "total_return", "max_drawdown"],
            ascending=[False, False, True],
        )
        cmp.to_csv(OUT / "set_comparison.csv", index=False)
        best_df = pd.DataFrame(
            {
                "source": [best_overall["source"]],
                "n_features": [len(best_overall["features"])],
                "features": ["|".join(best_overall["features"])],
                **{k: [best_overall["metrics"].get(k)] for k in ("n_trades", "win_rate", "profit_factor", "total_return", "max_drawdown", "roc_auc", "avg_pr")},
            }
        )
        best_df.to_csv(OUT / "best_feature_set.csv", index=False)
        (OUT / "best_feature_set.json").write_text(json.dumps(best_overall, indent=2, default=str), encoding="utf-8")
        print("BEST:", best_overall["source"], "n=", len(best_overall["features"]), "pf=", best_overall["metrics"].get("profit_factor"))

    db.upsert_run(
        {
            "run_id": run_id,
            "status": "completed",
            "baseline_pf": baseline.get("profit_factor"),
            "baseline_return": baseline.get("total_return"),
            "baseline_dd": baseline.get("max_drawdown"),
            "notes": f"best={best_overall.get('source')}",
        }
    )
    export_from_db(db, run_id, OUT)  # DB is source of truth; resume-safe CSVs
    _save_state({"run_id": run_id, "done_experiments": sorted(done), "phase": "completed", "best": best_overall})
    print(f"\nDONE fs_run_id={run_id}")
    print(f"artifacts -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
