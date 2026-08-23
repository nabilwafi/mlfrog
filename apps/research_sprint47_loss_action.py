"""Sprint 47 — Conditional loss ACTION (research only).

Frozen Sprint 46 trigger: first current_R <= -0.50R AND p_lgbm >= 0.80.
Do not retune. Do not change P0 / entry. No P1/P2/P3 / reversal.

  python apps/research_sprint47_loss_action.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

from apps.research_sprint40_iter1_time_aware_exit import STARTING
from apps.research_sprint41_iter4_switch import _port, _year_row
from apps.research_sprint42_attribution import TRUE_OOS, _fmt, _md, _pooled
from apps.research_sprint46_loss_reduction import (
    CHECKS,
    FOLDS,
    MODEL_FEATS,
    P0,
    _build_obs,
    _fit,
    _mc,
    _r_stats,
)
from apps.run_exit_engine_grid import Paths, simulate_combo
from apps.research_sprint39_exit_state import _join_feat7, _load_panel
from settings.strategy import ASSUMED_SLIPPAGE_POINTS, FALLBACK_SPREAD_POINTS, POINT, SL_ATR_MULT
from simulation.wf.sim import COST, load_h1, prepare_market

OUT = _ROOT / "artifacts/pipeline_backtest/loss_reduction/sprint47"
CK, THR = 0.50, 0.80
P0_ACT, P0_DIST = 0.25, 0.08
N_PLACEBO = 1000
MAE_BINS = (
    ("0.50_0.60R", 0.50, 0.60),
    ("0.60_0.75R", 0.60, 0.75),
    ("0.75_1.00R", 0.75, 1.00),
    ("1.00_1.25R", 1.00, 1.25),
    (">=1.25R", 1.25, 10.0),
)
TIME_BINS = (
    ("1_2", 1, 2),
    ("3_4", 3, 4),
    ("5_8", 5, 8),
    ("9_16", 9, 16),
    (">=17", 17, 99),
)


def _log(iter_n, result, gate, surviving, finding, nxt) -> None:
    print(f"ITERATION: {iter_n}")
    print(f"RESULT: {result}")
    print(f"GATE: {gate}")
    print(f"SURVIVING ACTIONS: {surviving}")
    print(f"KEY FINDING: {finding}")
    print(f"NEXT ITERATION: {nxt}")


def _frozen_p_loss(obs: pd.DataFrame) -> np.ndarray:
    """Reproduce Sprint 46 expanding LGBM scores. Frozen hyperparams / folds. No search."""
    early = obs[obs["checkpoint"] <= 0.75]
    X = early[MODEL_FEATS].astype(float).fillna(0.0).to_numpy()
    y = early["large_loss_1"].astype(int).to_numpy()
    years = early["year"].to_numpy()
    eidx = early.index.to_numpy()
    pred = np.full(len(obs), np.nan)
    for te, tr_years, _va in FOLDS:
        tr = np.isin(years, tr_years)
        te_m = years == te
        if tr.sum() < 200 or te_m.sum() < 50 or y[tr].min() == y[tr].max():
            continue
        pred[eidx[te_m]] = _fit("lgbm", X[tr], y[tr], X[te_m])
    m22, tr21 = years == 2022, years == 2021
    if tr21.sum() >= 200 and m22.sum() >= 50 and y[tr21].min() != y[tr21].max():
        pred[eidx[m22]] = _fit("lgbm", X[tr21], y[tr21], X[m22])
    return pred


def _triggers(obs: pd.DataFrame) -> dict[int, dict]:
    sub = obs[(obs["checkpoint"] == CK) & np.isfinite(obs["p_loss"]) & (obs["p_loss"] >= THR)]
    out: dict[int, dict] = {}
    for row in sub.itertuples(index=False):
        tid = int(row.trade_id)
        if tid in out:
            continue
        fill = max(float(row.current_R), -CK)
        out[tid] = {
            "j": int(row.bars_in_trade),
            "fill": fill,
            "current_R": float(row.current_R),
            "mae": float(row.mae_so_far_R),
            "year": int(row.year),
            "p": float(row.p_loss),
        }
    return out


def _net(gross: float, ru: float) -> float:
    return float(gross) - COST / max(ru, 1e-12)


def _a4_path(p: Paths, i: int, j: int, last: int, *, sl0: float = -0.75) -> tuple[float, int]:
    sl = float(sl0)
    extreme = 0.0
    for t in range(1, j + 1):
        f = p.fav[i, t]
        if np.isfinite(f):
            extreme = max(extreme, float(f))
    advj = p.adv[i, j]
    if np.isfinite(advj) and float(advj) >= -sl:
        return sl, j
    for k in range(j + 1, last + 1):
        fav, adv = p.fav[i, k], p.adv[i, k]
        if np.isfinite(fav):
            extreme = max(extreme, float(fav))
        if extreme >= P0_ACT and np.isfinite(p.atr_over_r[i, k]):
            sl = max(sl, extreme - P0_DIST * float(p.atr_over_r[i, k]))
        if np.isfinite(adv) and float(adv) >= -sl:
            return sl, k
    cr = p.close_r[i, last]
    return (float(cr) if np.isfinite(cr) else sl), last


def _apply_actions(p: Paths, sim: dict, trig: dict[int, dict], *, tick_r: np.ndarray | None = None) -> dict[str, dict]:
    r0 = np.array(sim["r_multiple"], dtype=float)
    h0 = np.array(sim["holding_bars"], dtype=np.int64)
    ru = np.maximum(p.r_unit_pct, 1e-12)
    tr = tick_r if tick_r is not None else np.zeros(p.n)
    out = {
        "A0": {"r": r0.copy(), "hold": h0.copy()},
        "A1": {"r": r0.copy(), "hold": h0.copy()},
        "A2": {"r": r0.copy(), "hold": h0.copy()},
        "A3": {"r": r0.copy(), "hold": h0.copy()},
        "A4": {"r": r0.copy(), "hold": h0.copy()},
    }
    for i, info in trig.items():
        j, fill = int(info["j"]), float(info["fill"])
        last = int(h0[i])
        fill_g = fill - float(tr[i])
        g0 = r0[i] + COST / ru[i]
        out["A1"]["r"][i] = _net(fill_g, ru[i])
        out["A1"]["hold"][i] = max(j, 1)
        out["A2"]["r"][i] = _net(0.50 * fill_g + 0.50 * g0, ru[i])
        out["A3"]["r"][i] = _net(0.75 * fill_g + 0.25 * g0, ru[i])
        gross4, h4 = _a4_path(p, i, j, last, sl0=-0.75 - float(tr[i]))
        out["A4"]["r"][i] = _net(gross4, ru[i])
        out["A4"]["hold"][i] = max(h4, 1)
    return out


def _sim_from(sim: dict, pack: dict, p: Paths) -> dict:
    r = pack["r"]
    hold = pack["hold"]
    net = r * p.r_unit_pct
    out = dict(sim)
    out["r_multiple"], out["net_return"], out["holding_bars"] = r, net, hold
    return out


def _action_row(name: str, r: np.ndarray, r0: np.ndarray, trades: pd.DataFrame, trig: dict) -> dict:
    ll = trades["large_loss_1"].to_numpy(dtype=bool)
    rec = trades["p0_R"].to_numpy() > 0
    tid = trades["trade_id"].to_numpy(dtype=int)
    hit = np.array([int(t) in trig for t in tid])
    G = (trades["mae_R"].to_numpy() >= 0.75) & rec
    G1 = (trades["mae_R"].to_numpy() >= 1.0) & rec
    destroyed = G & (r <= 0) & rec
    destroyed1 = G1 & (r <= 0) & rec
    tail_saved = float(((r - r0)[ll & hit]).sum()) if (ll & hit).any() else 0.0
    rec_lost = float(((r0 - r)[rec & hit]).sum()) if (rec & hit).any() else 0.0
    st = _r_stats(r)
    return {
        "action": name,
        **st,
        "p_le_m0_75": float((r <= -0.75).mean()),
        "tail_saved_R": tail_saved,
        "recovery_lost_R": rec_lost,
        "net_benefit_R": tail_saved - rec_lost,
        "n_triggered": int(hit.sum()),
        "recovery_preservation": float(1.0 - destroyed.sum() / G.sum()) if G.sum() else 1.0,
        "recovery_preservation_mae1": float(1.0 - destroyed1.sum() / G1.sum()) if G1.sum() else 1.0,
        "n_rec_mae075": int(G.sum()),
        "n_rec_mae1": int(G1.sum()),
        "n_rec_destroyed": int(destroyed.sum()),
        "n_rec_destroyed_mae1": int(destroyed1.sum()),
        "tail_loss_rate": float((r <= -1.00).mean()),
        "p0_tail_loss_rate": float(ll.mean()),
        "tail_reduced": float((r <= -1.00).mean()) < float(ll.mean()) - 1e-12 or tail_saved > 0,
    }


def _tick_r(p: Paths, n_ticks: float) -> np.ndarray:
    one_r = np.maximum(SL_ATR_MULT * p.atr, 1e-12)
    return np.full(p.n, n_ticks) * POINT / one_r


def _charts(oos: pd.DataFrame, packs: dict, yports: dict, best: str) -> None:
    r0 = oos["p0_R"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ("A0", "A1", "A2", "A3", "A4"):
        if name not in packs:
            continue
        r = oos[f"{name}_R"].to_numpy() if f"{name}_R" in oos.columns else None
        if r is None:
            continue
        ax.hist(r, bins=40, histtype="step", label=name, density=True)
    ax.axvline(-1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("R"); ax.set_ylabel("density"); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sprint47_tail_loss_distribution.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ("A1", "A2", "A3", "A4"):
        if f"{name}_R" not in oos.columns:
            continue
        ax.scatter(float(packs[name]["recovery_lost_R"]), float(packs[name]["tail_saved_R"]), s=60, label=name)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("recovery_lost_R"); ax.set_ylabel("tail_saved_R"); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sprint47_recovery_tradeoff.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for y, book in yports.items():
        eq = np.asarray(book.get("equity", []), dtype=float)
        if eq.size:
            ax.plot(eq, label=str(y), lw=1)
    ax.set_title(f"equity {best} isolated $280/year")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sprint47_action_equity_curves.png", dpi=120)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = _join_feat7(_load_panel())
    mkt = prepare_market(load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet"))
    p = Paths(panel, mkt)
    sim0 = simulate_combo(p, **P0)
    trades, obs = _build_obs(p, mkt, sim0)
    obs = obs.copy()
    obs["p_loss"] = _frozen_p_loss(obs)
    trig_all = _triggers(obs)
    oos_tr = trades[trades["year"].isin(TRUE_OOS)].copy()
    trig = {k: v for k, v in trig_all.items() if v["year"] in TRUE_OOS}
    r0 = oos_tr["p0_R"].to_numpy()
    print(f"oos trades={len(oos_tr)} triggered={len(trig)} ck=-{CK} p>={THR}")

    packs = _apply_actions(p, sim0, trig)
    # map action R onto oos_tr order
    for name, pack in packs.items():
        mp = dict(enumerate(pack["r"]))
        oos_tr[f"{name}_R"] = [mp[int(t)] for t in oos_tr["trade_id"]]

    # ----- Iter 1 -----
    rows1 = []
    for name in ("A0", "A1", "A2", "A3", "A4"):
        rr = oos_tr[f"{name}_R"].to_numpy()
        rows1.append(_action_row(name, rr, r0, oos_tr, trig if name != "A0" else {}))
    df1 = pd.DataFrame(rows1)
    df1.to_csv(OUT / "sprint47_iter1_action_replay.csv", index=False)
    surv = []
    for r in rows1:
        if r["action"] == "A0":
            continue
        if r["tail_reduced"] and r["recovery_preservation"] >= 0.80:
            surv.append(r["action"])
    if not surv:
        _log(1, "action replay", "FAIL", "none", "no action reduces tail and keeps >=80% recoveries", "STOP")
        (OUT / "sprint47_final_verdict.json").write_text(json.dumps({
            "sprint": 47, "verdict": "NO_ACTION_EDGE", "production_changed": False,
        }, indent=2), encoding="utf-8")
        (OUT / "sprint47_report.md").write_text(
            "# Sprint 47\n\n**Verdict: NO_ACTION_EDGE**\n\nNo action passed Iter 1.\n", encoding="utf-8"
        )
        print("FINAL VERDICT: NO_ACTION_EDGE")
        print("PRODUCTION: UNCHANGED")
        print("NEXT RESEARCH STEP: stop loss-action research")
        return 0
    _log(1, "action replay", "PASS_ITER1", ", ".join(surv),
         f"triggered={len(trig)}; tail-reducing actions={surv}", 2)

    # ----- Iter 2 -----
    g1 = oos_tr[(oos_tr["mae_R"] >= 1.0) & (oos_tr["p0_R"] > 0)]
    rows2 = []
    surv2 = []
    for name in surv:
        sub_r = g1[f"{name}_R"].to_numpy()
        p0g = g1["p0_R"].to_numpy()
        lost = float((p0g - sub_r).sum())
        pres = float((sub_r > 0).mean()) if len(sub_r) else 1.0
        rows2.append({
            "action": name, "n": int(len(g1)),
            "recovery_rate": pres,
            "mean_P0_R": float(p0g.mean()) if len(p0g) else 0.0,
            "mean_action_R": float(sub_r.mean()) if len(sub_r) else 0.0,
            "recovery_lost_R": lost,
            "recovery_preserved_pct": pres,
            "large_MAE_recovery_loss": lost,
        })
        if pres >= 0.80:
            surv2.append(name)
    pd.DataFrame(rows2).to_csv(OUT / "sprint47_iter2_recovery_protection.csv", index=False)
    if not surv2:
        _log(2, "recovery protection", "FAIL_RECOVERY", "none",
             "MAE>=1 recovered trades destroyed", "STOP")
        (OUT / "sprint47_final_verdict.json").write_text(json.dumps({
            "sprint": 47, "verdict": "ACTION_RECOVERY_DESTRUCTIVE", "production_changed": False,
            "iter1_surviving": surv,
        }, indent=2), encoding="utf-8")
        (OUT / "sprint47_report.md").write_text(
            "# Sprint 47\n\n**Verdict: ACTION_RECOVERY_DESTRUCTIVE**\n", encoding="utf-8"
        )
        print("FINAL VERDICT: ACTION_RECOVERY_DESTRUCTIVE")
        print("PRODUCTION: UNCHANGED")
        return 0
    _log(2, "recovery protection", "PASS_RECOVERY", ", ".join(surv2),
         f"MAE>=1 recovered n={len(g1)}", 3)

    # ----- Iter 3 -----
    rows3 = []
    shares = {a: [] for a in surv2}
    for name in surv2:
        tot = 0.0
        bucket_net = []
        for lab, lo, hi in MAE_BINS:
            m = (oos_tr["mae_R"] >= lo) & (oos_tr["mae_R"] < hi)
            sub = oos_tr[m]
            if not len(sub):
                bucket_net.append((lab, 0.0, 0))
                continue
            r = sub[f"{name}_R"].to_numpy(); b = sub["p0_R"].to_numpy()
            hit = np.array([int(t) in trig for t in sub["trade_id"]])
            ll = sub["large_loss_1"].to_numpy(dtype=bool)
            rec = b > 0
            ts = float(((r - b)[ll & hit]).sum()) if (ll & hit).any() else 0.0
            rl = float(((b - r)[rec & hit]).sum()) if (rec & hit).any() else 0.0
            net = ts - rl
            tot += net
            bucket_net.append((lab, net, int(len(sub))))
            rows3.append({
                "action": name, "mae_bucket": lab, "n": int(len(sub)),
                "baseline_mean_R": float(b.mean()), "action_mean_R": float(r.mean()),
                "delta_R": float((r - b).mean()), "tail_saved_R": ts, "recovery_lost_R": rl,
                "net_R": net, "recovery_rate": float((r > 0).mean()),
            })
        for lab, net, n in bucket_net:
            sh = net / tot if abs(tot) > 1e-9 else 0.0
            shares[name].append((lab, sh, n, net))
            for row in rows3:
                if row["action"] == name and row["mae_bucket"] == lab:
                    row["share_of_net"] = sh
    pd.DataFrame(rows3).to_csv(OUT / "sprint47_iter3_tail_attribution.csv", index=False)
    tail_dep = False
    cluster = None
    for name, shs in shares.items():
        if not shs:
            continue
        lab, sh, n, net = max(shs, key=lambda x: abs(x[1]))
        if sh > 0.50 and n < 80:
            tail_dep, cluster = True, lab
    _log(3, "tail attribution", "TAIL_DEPENDENT" if tail_dep else "PASS",
         ", ".join(surv2),
         f"cluster={cluster}" if tail_dep else "benefit not a tiny discrete cluster", 4)

    # MAE chart
    fig, ax = plt.subplots(figsize=(8, 4))
    labs = [b[0] for b in MAE_BINS]
    x = np.arange(len(labs))
    w = 0.18
    for i, name in enumerate(surv2):
        d = [next((r["delta_R"] for r in rows3 if r["action"] == name and r["mae_bucket"] == lab), 0) for lab in labs]
        ax.bar(x + i * w, d, w, label=name)
    ax.set_xticks(x + w); ax.set_xticklabels(labs, rotation=20, ha="right")
    ax.axhline(0, color="k", lw=0.5); ax.legend(); ax.set_ylabel("delta R")
    fig.tight_layout()
    fig.savefig(OUT / "sprint47_action_by_mae.png", dpi=120)
    plt.close(fig)

    # ----- Iter 4 -----
    rows4 = []
    late_share = {}
    for name in surv2:
        ts_late = ts_all = 0.0
        for lab, lo, hi in TIME_BINS:
            ids = [tid for tid, inf in trig.items() if lo <= inf["j"] <= hi]
            sub = oos_tr[oos_tr["trade_id"].isin(ids)]
            if not len(sub):
                rows4.append({"action": name, "time_bucket": lab, "trigger_count": 0,
                              "tail_loss_rate": 0, "recovery_rate": 0, "action_delta_R": 0,
                              "recovery_lost_R": 0, "tail_saved_R": 0})
                continue
            r = sub[f"{name}_R"].to_numpy(); b = sub["p0_R"].to_numpy()
            ll = sub["large_loss_1"].to_numpy(dtype=bool)
            rec = b > 0
            ts = float(((r - b)[ll]).sum()) if ll.any() else 0.0
            rl = float(((b - r)[rec]).sum()) if rec.any() else 0.0
            ts_all += ts
            if lo >= 17:
                ts_late += ts
            rows4.append({
                "action": name, "time_bucket": lab, "trigger_count": int(len(sub)),
                "tail_loss_rate": float((r <= -1).mean()),
                "recovery_rate": float((r > 0).mean()),
                "action_delta_R": float((r - b).mean()),
                "recovery_lost_R": rl, "tail_saved_R": ts,
            })
        late_share[name] = ts_late / ts_all if abs(ts_all) > 1e-9 else 1.0
    pd.DataFrame(rows4).to_csv(OUT / "sprint47_iter4_timing_attribution.csv", index=False)
    dead = oos_tr[oos_tr["trade_id"].isin([t for t, inf in trig.items() if inf["mae"] >= 1.0 or inf["current_R"] <= -1.0])]
    timing_fail = all(v > 0.70 for v in late_share.values()) if late_share else True
    rec_ok = all(next(r["recovery_preservation"] for r in rows1 if r["action"] == a) >= 0.80 for a in surv2)
    gate4 = "ACTION_TOO_LATE" if timing_fail or not rec_ok else "PASS_TIMING"
    _log(4, "time-to-action", gate4, ", ".join(surv2),
         f"late>=17 share={ {k: round(v, 2) for k, v in late_share.items()} }", 5)

    fig, ax = plt.subplots(figsize=(8, 4))
    labs = [b[0] for b in TIME_BINS]
    x = np.arange(len(labs)); w = 0.18
    for i, name in enumerate(surv2):
        d = [next((r["action_delta_R"] for r in rows4 if r["action"] == name and r["time_bucket"] == lab), 0) for lab in labs]
        ax.bar(x + i * w, d, w, label=name)
    ax.set_xticks(x + w); ax.set_xticklabels(labs); ax.axhline(0, color="k", lw=0.5); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sprint47_action_by_time.png", dpi=120)
    plt.close(fig)

    # ----- Iter 5 -----
    rows5 = []
    year_net = {a: [] for a in surv2}
    for y in TRUE_OOS:
        ty = oos_tr[oos_tr["year"] == y]
        for name in ["A0", *surv2]:
            r = ty[f"{name}_R"].to_numpy(); b = ty["p0_R"].to_numpy()
            hit = np.array([int(t) in trig for t in ty["trade_id"]])
            ll = ty["large_loss_1"].to_numpy(dtype=bool)
            rec = b > 0
            ts = float(((r - b)[ll & hit]).sum()) if (ll & hit).any() else 0.0
            rl = float(((b - r)[rec & hit]).sum()) if (rec & hit).any() else 0.0
            G = (ty["mae_R"] >= 0.75) & rec
            dest = G & (r <= 0)
            st = _r_stats(r)
            rows5.append({
                "year": y, "action": name, "trades": int(len(ty)),
                "pf": st["pf"], "avg_r": st["mean_R"], "wr": st["wr"],
                "worst_R": st["worst_R"], "tail_loss_rate": float((r <= -1).mean()),
                "tail_saved_R": ts, "recovery_lost_R": rl, "net_benefit_R": ts - rl,
                "recovery_preservation": float(1.0 - dest.sum() / G.sum()) if G.sum() else 1.0,
            })
            if name != "A0":
                year_net[name].append(ts - rl)
    # portfolio DD/return per year
    for y in TRUE_OOS:
        for name in ["A0", *surv2]:
            sim = _sim_from(sim0, packs[name], p)
            port = _year_row(y, _port(p, sim, y), sim)
            for row in rows5:
                if row["year"] == y and row["action"] == name:
                    row["DD"] = port["dd"]; row["Return"] = port["ret"]
                    row["MaxLoss"] = port["max_loss_streak"]; row["PF_port"] = port["pf"]
    pd.DataFrame(rows5).to_csv(OUT / "sprint47_iter5_yearly_oos.csv", index=False)
    surv5 = []
    for name in surv2:
        ydf = [r for r in rows5 if r["action"] == name]
        p0y = {r["year"]: r for r in rows5 if r["action"] == "A0"}
        n_imp = sum(r["tail_loss_rate"] <= p0y[r["year"]]["tail_loss_rate"] + 1e-12 for r in ydf)
        nets = np.array(year_net[name], dtype=float)
        share = float(nets.max() / nets.sum()) if nets.sum() > 1e-9 else 1.0
        pres = float(np.mean([r["recovery_preservation"] for r in ydf]))
        ok = n_imp >= 4 and share <= 0.50 and pres >= 0.80
        if ok:
            surv5.append(name)
        print(f"  {name}: years_tail_ok={n_imp}/5 max_share={share:.2f} pres={pres:.2f} gate={'PASS' if ok else 'FAIL'}")
    gate5 = "PASS" if surv5 else "FAIL"
    if not surv5:
        surv5 = list(surv2)  # keep for later iters as documentation; verdict FRAGILE
    _log(5, "yearly OOS", gate5, ", ".join(surv5),
         "4/5 years + no single-year >50%" if gate5 == "PASS" else "year/concentration fail", 6)

    # pick best surviving by net_benefit among gate5 if PASS else surv2
    cand = surv5 if gate5 == "PASS" else surv2
    best = max(cand, key=lambda a: next(r["net_benefit_R"] for r in rows1 if r["action"] == a))

    # ----- Iter 6 -----
    rows6 = []
    yports = {}
    ports = {}
    for name in ["A0", best]:
        sim = _sim_from(sim0, packs[name], p)
        yrows = []
        for y in TRUE_OOS:
            port = _port(p, sim, y)
            yr = _year_row(y, port, sim)
            yrows.append(yr)
            if name == best:
                yports[y] = port
        po = _pooled(yrows)
        mc = _mc(po["pnl"], name)
        rows6.append({
            "book": "BOOK_A" if name == "A0" else "BOOK_B",
            "action": name,
            "pf": po["pf"], "ret": po["ret"], "dd": po["dd"], "wr": po["wr"],
            "avg_r": po["avg_r"], "max_loss": po["max_loss_streak"],
            **{k: mc[k] for k in ("prob_ruin", "median_dd", "p95_dd", "p99_dd", "worst_dd")},
        })
        ports[name] = po
    pd.DataFrame(rows6).to_csv(OUT / "sprint47_iter6_portfolio_risk.csv", index=False)
    a0, b0 = [r for r in rows6 if r["action"] == "A0"][0], [r for r in rows6 if r["action"] == best][0]
    best_row = next(r for r in rows1 if r["action"] == best)
    gate6 = (
        best_row["tail_reduced"]
        and b0["prob_ruin"] <= a0["prob_ruin"] + 1e-12
        and b0["dd"] <= a0["dd"] + 0.01
        and best_row["net_benefit_R"] > 0
    )
    _log(6, "portfolio+MC", "PASS" if gate6 else "FAIL", best,
         f"PF {a0['pf']:.2f}->{b0['pf']:.2f} DD {a0['dd']:.3f}->{b0['dd']:.3f} ruin {b0['prob_ruin']:.3f}", 7)

    # ----- Iter 7 -----
    stress = [
        ("BASE", 0.0),
        ("+1tick", 1.0),
        ("+2tick", 2.0),
        ("spread_slip", float(FALLBACK_SPREAD_POINTS + ASSUMED_SLIPPAGE_POINTS)),
    ]
    rows7 = []
    for lab, ntk in stress:
        tr = _tick_r(p, ntk)
        pk = _apply_actions(p, sim0, trig, tick_r=tr)
        r = np.array([pk[best]["r"][int(t)] for t in oos_tr["trade_id"]])
        st = _action_row(best, r, r0, oos_tr, trig)
        sim = _sim_from(sim0, pk[best], p)
        yrows = [_year_row(y, _port(p, sim, y), sim) for y in TRUE_OOS]
        po = _pooled(yrows)
        rows7.append({
            "stress": lab, "n_ticks": ntk, "action": best,
            "pf": po["pf"], "dd": po["dd"], "ret": po["ret"],
            "tail_saved_R": st["tail_saved_R"], "recovery_lost_R": st["recovery_lost_R"],
            "net_benefit_R": st["net_benefit_R"],
        })
    pd.DataFrame(rows7).to_csv(OUT / "sprint47_iter7_execution_stress.csv", index=False)
    gate7 = all(r["net_benefit_R"] > 0 for r in rows7)
    _log(7, "execution stress", "PASS" if gate7 else "FAIL", best,
         f"nets={[round(r['net_benefit_R'], 1) for r in rows7]}", 8)

    # ----- Iter 8 -----
    rng = np.random.default_rng(42)
    actual_ts = best_row["tail_saved_R"]
    actual_net = best_row["net_benefit_R"]
    ck_ids_by_y = (
        obs[(obs["checkpoint"] == CK) & obs["year"].isin(TRUE_OOS)]
        .groupby("year")["trade_id"]
        .apply(lambda s: [int(x) for x in s.unique()])
        .to_dict()
    )
    n_by_y = {y: sum(1 for t, inf in trig.items() if inf["year"] == y) for y in TRUE_OOS}
    plc_net, plc_ts = [], []
    fill_of = {int(t): trig[int(t)]["fill"] if int(t) in trig else None for t in oos_tr["trade_id"]}
    # fills for any -0.50 obs
    fill_ck = {}
    for row in obs[obs["checkpoint"] == CK].itertuples(index=False):
        fill_ck.setdefault(int(row.trade_id), max(float(row.current_R), -CK))
    for _ in range(N_PLACEBO):
        fake = {}
        for y, n in n_by_y.items():
            pool = ck_ids_by_y[y]
            if n <= 0 or not pool:
                continue
            pick = rng.choice(pool, size=min(n, len(pool)), replace=False)
            for tid in pick:
                fake[int(tid)] = {"j": 1, "fill": fill_ck.get(int(tid), -CK), "year": y,
                                  "current_R": fill_ck.get(int(tid), -CK), "mae": 0.0, "p": 1.0}
        pk = _apply_actions(p, sim0, fake)
        r = np.array([pk[best]["r"][int(t)] for t in oos_tr["trade_id"]])
        st = _action_row(best, r, r0, oos_tr, fake)
        plc_net.append(st["net_benefit_R"])
        plc_ts.append(st["tail_saved_R"])
    plc_net, plc_ts = np.asarray(plc_net), np.asarray(plc_ts)
    p_net = float((plc_net >= actual_net).mean())
    p_ts = float((plc_ts >= actual_ts).mean())
    pd.DataFrame([{
        "actual_net_benefit_R": actual_net, "placebo_net_mean": float(plc_net.mean()),
        "placebo_net_std": float(plc_net.std()), "empirical_p_net": p_net,
        "actual_tail_saved_R": actual_ts, "placebo_ts_mean": float(plc_ts.mean()),
        "empirical_p_ts": p_ts, "n_rep": N_PLACEBO,
    }]).to_csv(OUT / "sprint47_iter8_placebo.csv", index=False)

    loo = []
    for drop in TRUE_OOS:
        sub = oos_tr[oos_tr["year"] != drop]
        r = sub[f"{best}_R"].to_numpy(); b = sub["p0_R"].to_numpy()
        hit = np.array([int(t) in trig for t in sub["trade_id"]])
        ll = sub["large_loss_1"].to_numpy(dtype=bool); rec = b > 0
        ts = float(((r - b)[ll & hit]).sum()) if (ll & hit).any() else 0.0
        rl = float(((b - r)[rec & hit]).sum()) if (rec & hit).any() else 0.0
        loo.append({"exclude_year": drop, "net_benefit_R": ts - rl, "tail_saved_R": ts, "recovery_lost_R": rl})
    pd.DataFrame(loo).to_csv(OUT / "sprint47_iter8_leave_year.csv", index=False)
    loo_ok = all(x["net_benefit_R"] > 0 for x in loo)

    leave_tail = {"cluster": cluster, "applied": False, "net_benefit_R": actual_net}
    if cluster:
        lo, hi = next((a, b) for lab, a, b in MAE_BINS if lab == cluster)
        sub = oos_tr[~((oos_tr["mae_R"] >= lo) & (oos_tr["mae_R"] < hi))]
        r = sub[f"{best}_R"].to_numpy(); b = sub["p0_R"].to_numpy()
        hit = np.array([int(t) in trig for t in sub["trade_id"]])
        ll = sub["large_loss_1"].to_numpy(dtype=bool); rec = b > 0
        ts = float(((r - b)[ll & hit]).sum()) if (ll & hit).any() else 0.0
        rl = float(((b - r)[rec & hit]).sum()) if (rec & hit).any() else 0.0
        leave_tail = {"cluster": cluster, "applied": True, "net_benefit_R": ts - rl,
                      "tail_saved_R": ts, "recovery_lost_R": rl, "n_left": int(len(sub))}
        pd.DataFrame([leave_tail]).to_csv(OUT / "sprint47_iter8_leave_tail.csv", index=False)
    cluster_ok = (not leave_tail["applied"]) or leave_tail["net_benefit_R"] > 0

    gate8 = p_net < 0.05 and loo_ok and cluster_ok and not tail_dep
    _log(8, "placebo+leave-out", "PASS" if gate8 else "FAIL", best,
         f"p_net={p_net:.3f} loo_ok={loo_ok} cluster_ok={cluster_ok} tail_dep={tail_dep}", "FINAL")

    # charts
    _charts(oos_tr, {r["action"]: r for r in rows1}, yports, best)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(oos_tr["p0_R"], bins=40, alpha=0.5, label="P0", density=True)
    ax.hist(oos_tr[f"{best}_R"], bins=40, alpha=0.5, label=best, density=True)
    ax.axvline(-1.0, color="k", ls="--", lw=0.8)
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "sprint47_tail_loss_distribution.png", dpi=120)
    plt.close(fig)

    # ----- verdict -----
    if gate4 == "ACTION_TOO_LATE":
        verdict = "ACTION_TOO_LATE"
    elif gate5 != "PASS":
        verdict = "ACTION_FRAGILE"
    elif not gate7:
        verdict = "ACTION_FRAGILE"
    elif tail_dep or not cluster_ok:
        verdict = "ACTION_TAIL_DEPENDENT"
    elif not gate8:
        verdict = "ACTION_FRAGILE"
    elif gate6 and gate7 and gate8 and gate5 == "PASS":
        verdict = "ACTION_TAIL_REDUCTION_CONFIRMED"
    else:
        verdict = "ACTION_FRAGILE"

    rec_map = {
        "ACTION_TAIL_REDUCTION_CONFIRMED": "CANDIDATE_FOR_LOSS_ACTION_RESEARCH — still not production",
        "ACTION_TAIL_DEPENDENT": "stop; benefit is a discrete cluster",
        "ACTION_FRAGILE": "stop; fails year / execution / placebo robustness",
        "ACTION_TOO_LATE": "stop; trigger fires too late to matter",
        "ACTION_RECOVERY_DESTRUCTIVE": "stop",
        "NO_ACTION_EDGE": "stop",
    }
    a0r = next(r for r in rows1 if r["action"] == "A0")
    lines = [
        "# Sprint 47 — Conditional Loss Action",
        "",
        "Research only. Frozen Sprint 46 trigger: first current_R ≤ **−0.50R** and p ≥ **0.80**. P0 / entry unchanged.",
        "",
        f"## Best action: **{best}**",
        "",
        f"1. Best action: **{best}**",
        f"2. Tail saved: {best_row['tail_saved_R']:+.2f}R  (LL rate {a0r['tail_loss_rate']:.3f} → {best_row['tail_loss_rate']:.3f})",
        f"3. Recovery lost: {best_row['recovery_lost_R']:.2f}R  preservation {100*best_row['recovery_preservation']:.1f}%",
        f"4. Net R: **{best_row['net_benefit_R']:+.2f}**",
        f"5. DD: {a0['dd']:.3f} → {b0['dd']:.3f}",
        f"6. PF: {a0['pf']:.2f} → {b0['pf']:.2f}",
        f"7. OOS years tail improved: see iter5 (gate {gate5})",
        f"8. Execution stress net all positive: {gate7}",
        f"9. Placebo p(net)={p_net:.4f}  p(tail_saved)={p_ts:.4f}",
        f"10. Leave-one-year-out all positive: {loo_ok}",
        f"11. Broad vs cluster: {'TAIL_DEPENDENT '+str(cluster) if tail_dep else 'not a tiny discrete cluster'}",
        f"12. Failure mode: {verdict if verdict != 'ACTION_TAIL_REDUCTION_CONFIRMED' else 'none — confirmed on frozen trigger'}",
        "13. Production: **UNCHANGED** (not shipped)",
        "",
        "### Iter 1 actions",
        "",
        _md(rows1, ["action", "mean_R", "p5_R", "worst_R", "p_le_m1_00", "tail_saved_R", "recovery_lost_R",
                    "net_benefit_R", "recovery_preservation"],
            {"mean_R": 3, "p5_R": 3, "worst_R": 3, "p_le_m1_00": 3, "tail_saved_R": 2,
             "recovery_lost_R": 2, "net_benefit_R": 2, "recovery_preservation": 3}),
        "",
        "### Yearly OOS (best vs P0 tail-loss rate)",
        "",
        _md([r for r in rows5 if r["action"] in ("A0", best)],
            ["year", "action", "trades", "tail_loss_rate", "net_benefit_R", "recovery_preservation", "PF_port", "DD"],
            {"year": 0, "trades": 0, "tail_loss_rate": 3, "net_benefit_R": 2, "recovery_preservation": 3, "PF_port": 2, "DD": 3}),
        "",
        "### Execution stress",
        "",
        _md(rows7, ["stress", "pf", "dd", "net_benefit_R"], {"pf": 2, "dd": 3, "net_benefit_R": 2}),
        "",
        f"## Verdict",
        "",
        f"**`{verdict}`**",
        "",
        "## Production",
        "",
        "Entry: UNCHANGED",
        "",
        "P0: UNCHANGED",
        "",
        "Loss action: NOT SHIPPED",
        "",
        "## Recommendation",
        "",
        rec_map[verdict] + ".",
        "",
    ]
    (OUT / "sprint47_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "sprint47_final_verdict.json").write_text(json.dumps({
        "sprint": 47, "verdict": verdict, "production_changed": False,
        "best_action": best, "checkpoint": CK, "threshold": THR, "model": "sprint46_lgbm_frozen",
        "net_benefit_R": actual_net, "tail_saved_R": actual_ts,
        "recovery_preservation": best_row["recovery_preservation"],
        "placebo_p_net": p_net, "loo_ok": loo_ok, "tail_dependent": tail_dep,
        "gate1": "PASS_ITER1", "gate2": "PASS_RECOVERY", "gate4": gate4,
        "gate5": gate5, "gate6": bool(gate6), "gate7": bool(gate7), "gate8": bool(gate8),
        "recommendation": rec_map[verdict],
    }, indent=2, default=float), encoding="utf-8")
    print(f"FINAL VERDICT: {verdict}")
    print("PRODUCTION: UNCHANGED")
    print(f"NEXT RESEARCH STEP: {rec_map[verdict]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
