"""Train and evaluate V2 against V1 with leave-events-out cross-validation (FR-25, UC-06).

    python -m flood_risk.train --data data/inventory_synthetic.csv

Writes models/v2_hgb.joblib, models/v2_logistic.joblib, reports/v2_report.md and figures.
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from . import v2
from .features import drainage_vulnerability, engineer
from .schema import HISTORY_FEATURES, LABEL, LEVELS, RAW_FEATURES, RELATIVE_FEATURES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LEADS_MIN = [0, 10, 20, 30]
MAIN_LEAD = 20
LEVEL_PROTOCOL = {
    "recall": "The High threshold is placed for 70% recall on training events, so recall is a design target; "
              "the false-alarm rate at that recall is the real comparison.",
    "budget": "Levels are placed by alert budget on training events (schema.ALERT_SHARE: High = top 2 % of rows).",
    "area": "Levels follow an alert budget inside each mapped area of the test event (schema.ALERT_SHARE: "
            "High = riskiest 2 % of each area), so every model raises the same number of alerts and "
            "recall and false-alarm rate compare directly.",
}
NAMES = {"v1": "V1 multi-criteria", "v1_relative": "Relative susceptibility", "logistic": "V2 logistic",
         "hgb": "V2 gradient boosting", "hgb_no_drainage": "V2 GB without drainage proxies",
         "hgb_terrain": "V2 GB terrain only", "hgb_rain": "V2 GB rain only",
         "hgb_inter": "V2 GB + rain x terrain products", "hybrid": "V2 hybrid (terrain, then rain)",
         "A_terrain": "A terrain", "B_rain": "B rain (observed + forecast)", "C_terrain_rain": "C terrain + rain",
         "D_terrain_rain_impervious": "D terrain + rain + built-up", "D_observed_only": "D without forecast (observed rain only)",
         "E_full": "E full (with drainage proxies)", "E_no_drainage": "E full without drainage proxies"}


def with_forecast_error(df: pd.DataFrame, lead_min, rng) -> pd.DataFrame:
    """Replace rainfall at ETA by what was known lead_min earlier.

    The error grows with lead time and is larger when 15-min values are interpolated
    from hourly forecasts (section 2.4 constraint). The part of the 1 h and 3 h
    accumulations that falls after the forecast time is a forecast too, with the same error.
    """
    lead = np.broadcast_to(np.asarray(lead_min, dtype=float), (len(df),))
    sigma = 0.15 + 0.012 * lead + 0.15 * df["rain_interpolated"].to_numpy()
    error = rng.lognormal(-sigma ** 2 / 2, sigma)
    out = df.copy()
    out["rain_eta_mmh"] = df["rain_eta_mmh"] * error
    for col, window in [("rain_1h_mm", 60), ("rain_3h_mm", 180)]:
        future = np.minimum(lead / window, 1)
        out[col] = df[col] * (1 - future + future * error)
    return out


def metrics(p, levels, y, base_rate: float | None = None) -> dict:
    """Section 8 metrics; with `base_rate` (flood rate of the training rows) also the Brier skill score
    against a constant forecast of that rate: 1 = perfect, 0 = no better than the constant, < 0 = worse."""
    alert = np.isin(levels, ["High", "Very high"])
    tp = int((alert & (y == 1)).sum())
    ref = {}
    if base_rate is not None:
        brier_ref = float(np.mean((base_rate - y) ** 2))
        ref = {"brier_ref": brier_ref, "bss": 1 - brier_score_loss(y, p) / brier_ref}
    return {
        "recall_high": tp / max(int(y.sum()), 1),
        "false_alarm": 1 - tp / max(int(alert.sum()), 1),
        "precision_high": tp / max(int(alert.sum()), 1),
        "accuracy_high": float((alert == (y == 1)).mean()),
        "alerts_per_1000": 1000 * alert.mean(),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "ece": ece(p, y),
        **{f"recall_top{int(100 * share)}": budget_recall(p, y, share) for share in (0.01, 0.02, 0.05)},
        **ref,
    }


def budget_recall(p, y, share) -> float:
    """Share of floods covered when alerting exactly on the `share` riskiest rows (ties broken by order)."""
    top = np.argsort(-np.asarray(p), kind="stable")[: max(int(len(p) * share), 1)]
    return float(y[top].sum() / max(int(y.sum()), 1))


def ece(p, y, bins=10):
    """Expected calibration error on quantile bins (the positives are rare)."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(len(edges) - 1) if (idx == b).any()))


def cross_validate(df: pd.DataFrame, n_folds: int, seed: int, neg_sample: float = 1.0,
                   params: dict | None = None, levels: str = "recall", kinds=None):
    kinds = kinds or v2.KINDS
    rng = np.random.default_rng(seed)
    rows, oof = [], {k: np.zeros(len(df)) for k in kinds}
    folds = GroupKFold(n_splits=n_folds).split(df, df[LABEL], df["event_id"])
    for fold, (tr, te) in enumerate(folds):
        # Training rows see forecasts with a random lead, like the operational inputs.
        train = with_forecast_error(df.iloc[tr], rng.uniform(0, 30, len(tr)), rng)
        test_events = ",".join(sorted(map(str, df.iloc[te]["event_id"].unique())))
        base_rate = float(df.iloc[tr][LABEL].mean())   # reference of the Brier skill score, from training rows only
        for kind in kinds:
            b = v2.fit(kind, train, seed, neg_sample=neg_sample, params=params, levels=levels)
            for lead in LEADS_MIN:
                # Built one at a time to save memory; the seed gives every model the same forecasts.
                test = with_forecast_error(df.iloc[te], lead, np.random.default_rng([seed, fold, lead]))
                p = b.predict_proba(test)
                area = test["aoi"].to_numpy() if "aoi" in test else None
                lv = v2.levels_by_share(p, area) if levels == "area" else v2.to_level(p, b.thresholds)
                for target in ["flooded", "flooded_true"]:
                    if target in test:
                        rows.append({"fold": fold, "events": test_events, "model": kind, "lead": lead, "target": target,
                                     **metrics(p, lv, test[target].to_numpy(), base_rate)})
                # Event level: was the segment flooded at some point of the storm?
                ev = pd.DataFrame({"p": p, "y": test[LABEL].to_numpy(), "seg": test["segment_id"].to_numpy(),
                                   "ev": test["event_id"].to_numpy(),
                                   "area": area if area is not None else 0}).groupby(["seg", "ev"]).max()
                ev_lv = (v2.levels_by_share(ev["p"].to_numpy(), ev["area"].to_numpy()) if levels == "area"
                         else v2.to_level(ev["p"].to_numpy(), b.thresholds))
                rows.append({"fold": fold, "events": test_events, "model": kind, "lead": lead, "target": "event_level",
                             **metrics(ev["p"].to_numpy(), ev_lv, ev["y"].to_numpy(), base_rate)})
                if lead == MAIN_LEAD:
                    oof[kind][te] = p
        print(f"fold {fold + 1}/{n_folds} done ({df.iloc[te]['event_id'].nunique()} test events)")
    return pd.DataFrame(rows), oof


def cross_validate_times(df: pd.DataFrame, kinds, group_col: str = "event_id", seed: int = 0,
                         neg_sample: float = 0.15, params: dict | None = None):
    """Grouped cross-validation on prediction-time rows (flood_risk/real/temporal.py).

    Rain is already what was known at each prediction time, so no forecast error is added here.
    Levels are an alert budget inside each area *at each prediction time*. Returns one row of
    metrics per test group x model x prediction time, and the out-of-fold probabilities.
    """
    rows, oof = [], {k: np.zeros(len(df)) for k in kinds}
    groups = df[group_col].to_numpy()
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=len(set(groups))).split(df, df[LABEL], groups)):
        train, test = df.iloc[tr], df.iloc[te]
        base_rate = float(train[LABEL].mean())
        for kind in kinds:
            b = v2.fit(kind, train, seed, neg_sample=neg_sample, params=params, levels="area")
            p = b.predict_proba(test)
            oof[kind][te] = p
            for off, idx in test.groupby("offset_h").indices.items():
                lv = v2.levels_by_share(p[idx], test["aoi"].to_numpy()[idx])
                rows.append({"fold": fold, "group": str(groups[te][0]), "model": kind, "offset_h": off,
                             **metrics(p[idx], lv, test[LABEL].to_numpy()[idx], base_rate)})
        print(f"fold {fold + 1}: {groups[te][0]} done", flush=True)
    return pd.DataFrame(rows), oof


def oracle_false_alarm(df: pd.DataFrame, segments_csv: str):
    """Synthetic data only: false-alarm rate at 70% recall of the hidden true probability."""
    if "flooded_true" not in df or not Path(segments_csv).exists():
        return None
    from .synth import flood_probability
    latent = pd.read_csv(segments_csv)[["segment_id", "_urban", "_true_inlets", "_maintenance"]]
    full = df.merge(latent, on="segment_id", how="left")
    p = flood_probability(full)
    out = {}
    for target in ["flooded", "flooded_true"]:
        y = full[target].to_numpy()
        alert = p >= v2.threshold_for_recall(p, y, 0.70)
        out[target] = 1 - (alert & (y == 1)).sum() / alert.sum()
    return out


def drainage_effect(df: pd.DataFrame) -> dict:
    """Section 8: do high-vulnerability segments flood at lower rainfall than others?"""
    flooded = df[df[LABEL] == 1]
    dv = drainage_vulnerability(flooded)
    hi, lo = flooded[dv >= dv.quantile(2 / 3)], flooded[dv <= dv.quantile(1 / 3)]
    test = mannwhitneyu(hi["rain_eta_mmh"], lo["rain_eta_mmh"], alternative="less")
    return {"n_high": len(hi), "n_low": len(lo),
            "median_rain_high": hi["rain_eta_mmh"].median(), "median_rain_low": lo["rain_eta_mmh"].median(),
            "p_value": test.pvalue}


def figures(df, oof, res, final, out: Path):
    y = df[LABEL].to_numpy()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for kind in [k for k in ["v1", "v1_relative", "logistic", "hgb"] if k in oof]:
        p = oof[kind]
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 13)))
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        pts = [(p[idx == b].mean(), y[idx == b].mean()) for b in range(len(edges) - 1) if (idx == b).any()]
        ax.plot(*zip(*pts), marker="o", label=NAMES[kind])
    lim = 1.1 * max(max(max(line.get_xdata()), max(line.get_ydata())) for line in ax.lines)
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect calibration")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set(xlabel="predicted probability", ylabel="observed flood frequency",
           title=f"Reliability (held-out events, lead {MAIN_LEAD} min)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "reliability.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    d = res[res.target == "flooded"].groupby(["model", "lead"])[["recall_high", "pr_auc"]].mean()
    for kind in [k for k in NAMES if k in set(res.model)]:
        axes[0].plot(LEADS_MIN, d.loc[kind, "recall_high"], marker="o", label=NAMES[kind])
        axes[1].plot(LEADS_MIN, d.loc[kind, "pr_auc"], marker="o", label=NAMES[kind])
    axes[0].axhline(0.70, color="grey", ls=":", lw=1)
    axes[0].set(xlabel="forecast lead time (min)", ylabel="recall at High+", title="Recall vs lead time")
    axes[1].set(xlabel="forecast lead time (min)", ylabel="PR-AUC", title="Ranking skill vs lead time")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "lead_time.png", dpi=130); plt.close(fig)

    X = engineer(df)[final.columns]
    sample = np.random.default_rng(0).choice(len(X), min(15000, len(X)), replace=False)
    imp = permutation_importance(final.model, X.iloc[sample], y[sample], scoring="average_precision",
                                 n_repeats=5, random_state=0)
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(np.array(final.columns)[order], imp.importances_mean[order], xerr=imp.importances_std[order])
    ax.set(xlabel="drop in PR-AUC when shuffled", title="V2 gradient boosting: feature importance")
    fig.tight_layout(); fig.savefig(out / "importance.png", dpi=130); plt.close(fig)
    return pd.Series(imp.importances_mean, index=final.columns).sort_values(ascending=False)


def fmt_table(res: pd.DataFrame, target: str, lead: int) -> str:
    cols = ["recall_high", "false_alarm", "alerts_per_1000", "roc_auc", "pr_auc", "recall_top2", "brier", "ece"]
    sub = res[(res.target == target) & (res.lead == lead)]
    mean, std = sub.groupby("model")[cols].mean(), sub.groupby("model")[cols].std()
    lines = ["| Model | Recall High+ | False-alarm rate | Alerts /1000 | ROC-AUC | PR-AUC | Recall top 2 % | Brier | ECE |",
             "|---|---|---|---|---|---|---|---|---|"]
    for kind in [k for k in NAMES if k in set(sub.model)]:
        cells = [f"{mean.loc[kind, c]:.3f} ± {std.loc[kind, c]:.3f}" if c not in ("brier", "ece")
                 else f"{mean.loc[kind, c]:.4f}" for c in cols]
        cells[2] = f"{mean.loc[kind, 'alerts_per_1000']:.1f}"
        lines.append(f"| {NAMES[kind]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/inventory_synthetic.csv")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neg-sample", type=float, default=1.0,
                    help="share of non-flooded rows kept for training (evaluation always uses all rows)")
    ap.add_argument("--levels", choices=["recall", "budget", "area"], default="recall",
                    help="place levels by target recall (section 8), by alert budget (schema.ALERT_SHARE), "
                         "or by the same budget inside each mapped area")
    ap.add_argument("--kinds", nargs="+", default=v2.KINDS, choices=list(NAMES))
    ap.add_argument("--hgb-params", help="JSON file of gradient-boosting parameters (from flood_risk.tune)")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--models", default="models")
    ap.add_argument("--segments", default="data/segments_synthetic.csv",
                    help="synthetic segments with hidden variables, for the oracle row only")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    missing = set(RAW_FEATURES + [LABEL, "event_id", "rain_interpolated"]) - set(df.columns)
    if missing:
        raise SystemExit(f"inventory is missing columns: {sorted(missing)}")
    # Real inventories reach millions of rows: keep only what the models use, in float32.
    df = df[[c for c in RAW_FEATURES + HISTORY_FEATURES + RELATIVE_FEATURES +
             [LABEL, "flooded_true", "event_id", "segment_id", "aoi", "rain_interpolated"] if c in df.columns]].copy()
    floats = df.select_dtypes("float64").columns
    df[floats] = df[floats].astype("float32")
    synthetic = "flooded_true" in df.columns
    out, models = Path(args.out), Path(args.models)
    out.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)

    params = json.loads(Path(args.hgb_params).read_text()) if args.hgb_params else None
    res, oof = cross_validate(df, args.folds, args.seed, args.neg_sample, params, args.levels, args.kinds)
    res.to_csv(out / "cv_metrics.csv", index=False)

    rng = np.random.default_rng(args.seed + 1)
    train_all = with_forecast_error(df, rng.uniform(0, 30, len(df)), rng)
    finals = {k: v2.fit(k, train_all, args.seed, neg_sample=args.neg_sample, params=params, levels=args.levels)
              for k in ["hgb", "logistic"]}
    for k, b in finals.items():
        joblib.dump(b, models / f"v2_{k}.joblib")
    importance = figures(df, oof, res, finals["hgb"], out)
    drain = drainage_effect(df)
    oracle = oracle_false_alarm(df, args.segments)

    report = [
        "# Version 2 evaluation report",
        "",
        f"Data: `{args.data}`: {len(df)} rows, {df['event_id'].nunique()} storm events, "
        f"{df['segment_id'].nunique() if 'segment_id' in df else '?'} segments, "
        f"{int(df[LABEL].sum())} documented flooded rows ({df[LABEL].mean():.2%}).",
        "",
        ("> **Synthetic data.** These numbers test the pipeline on a simulated inventory. They are not "
         "results of the system and must not be reported as such (section 8)." if synthetic else ""),
        "",
        f"Protocol: {args.folds}-fold cross-validation grouped by storm event (a storm is never in train "
        "and test). Inside each training fold, calibration and level thresholds use out-of-fold "
        "predictions, also grouped by event. " + LEVEL_PROTOCOL[args.levels] + " Rainfall at ETA is "
        "replaced by a forecast whose error grows with lead time.",
        "",
        f"## Results against documented floods (lead {MAIN_LEAD} min)",
        "",
        fmt_table(res, "flooded", MAIN_LEAD),
        "",
    ]
    report += [f"## Event-level results (lead {MAIN_LEAD} min)", "",
               "One row per segment and storm: alert if the maximum level during the storm is High or "
               "above, flooded if the segment flooded at any time. This matches how press and "
               "civil-protection reports are dated.", "",
               fmt_table(res, "event_level", MAIN_LEAD), ""]
    if oracle:
        report += ["## Ceiling on synthetic data", "",
                   "False-alarm rate at 70% recall when ranking by the hidden true flood probability "
                   f"(no model can beat it): {oracle['flooded']:.3f} against documented floods, "
                   f"{oracle['flooded_true']:.3f} against all floods. Flooding is random at a given "
                   "probability, so part of the false alarms cannot be removed by any model.", ""]
    if synthetic:
        report += [f"## Results against all floods, including undocumented ones (lead {MAIN_LEAD} min)",
                   "", "Only possible on synthetic data: shows the bias introduced by under-reporting.", "",
                   fmt_table(res, "flooded_true", MAIN_LEAD), ""]
    report += [
        "## Lead time", "", "![lead time](lead_time.png)", "",
        "## Calibration", "", "![reliability](reliability.png)", "",
        "## Drainage factor usefulness (section 8)", "",
        f"Among documented flooded rows, rainfall intensity at flooding for the top third of estimated "
        f"drainage vulnerability: median {drain['median_rain_high']:.1f} mm/h (n={drain['n_high']}); "
        f"bottom third: {drain['median_rain_low']:.1f} mm/h (n={drain['n_low']}). One-sided Mann-Whitney "
        f"p = {drain['p_value']:.2g}.",
        "",
        "This test alone does not prove the drainage hypothesis: high-vulnerability segments are also "
        "the most impervious, and impervious roads flood at lower rainfall whatever the state of their "
        "drains. The ablation row *V2 GB without drainage proxies* is the stronger test: the proxies are "
        "useful only if removing them lowers PR-AUC or raises the false-alarm rate on held-out storms"
        + ("." if "hgb_no_drainage" in args.kinds else " (that row was not run here: add `hgb_no_drainage` to --kinds)."),
        "",
        "## Feature importance (final gradient-boosting model)", "", "![importance](importance.png)", "",
        "| Feature | Drop in PR-AUC |", "|---|---|",
        *[f"| {k} | {v:.4f} |" for k, v in importance.head(10).items()],
        "",
        "## Deployed thresholds (calibrated probability)", "",
        "| Model | " + " | ".join(LEVELS[1:]) + " |", "|---|---|---|---|",
        *[f"| {NAMES[k]} | " + " | ".join(f"{b.thresholds[l]:.3f}" for l in LEVELS[1:]) + " |"
          for k, b in finals.items()],
    ]
    (out / "v2_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n" + fmt_table(res, "flooded", MAIN_LEAD).replace("±", "+/-"))
    print("\nevent level\n" + fmt_table(res, "event_level", MAIN_LEAD).replace("±", "+/-"))
    print(f"\noracle false-alarm rate at 70% recall: {oracle}")
    print(f"drainage effect: {drain}")
    print(f"report: {out / 'v2_report.md'}")


if __name__ == "__main__":
    main()
