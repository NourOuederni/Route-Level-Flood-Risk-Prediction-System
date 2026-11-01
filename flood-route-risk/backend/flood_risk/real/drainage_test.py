"""Ablation on real data: what do drainage, history, rain and terrain each add? (FR-08, section 8)

    python -m flood_risk.real.drainage_test

Each variant of the V2 gradient boosting drops one group of variables and is scored on storms
never seen in training. The drainage group holds the measured NYC DEP catch-basin density,
"drain blocked" reports made before the storm and the dry spell; history holds past flood
reports. If removing a group lowers the score on held-out storms, the group carries information.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .. import v2
from ..features import engineer
from ..schema import HISTORY_FEATURES, LABEL, RAW_FEATURES
from ..train import budget_recall

DRAINAGE = ["drain_inlet_density", "basin_reports_prior", "dry_spell_days", "drainage_vuln", "osm_completeness"]
HISTORY = ["black_spot_count", "flood_reports_250m", "flood_reports_365d"]
RAIN = ["rain_eta_mmh", "rain_1h_mm", "rain_3h_mm", "rain_24h_mm", "rain_excess_ratio", "runoff_index", "wadi_rain"]


def variants(columns):
    drop = {"complete": [], "without drainage": DRAINAGE, "without catch-basin density": ["drain_inlet_density"],
            "without drain-blocked reports": ["basin_reports_prior"], "without history": HISTORY,
            "without rain": RAIN, "without drainage and history": DRAINAGE + HISTORY}
    return {name: [c for c in columns if c not in cols] for name, cols in drop.items()}


def main():
    params_path = Path("models/nyc/hgb_params.json")
    params = json.loads(params_path.read_text()) if params_path.exists() else None
    df = pd.read_csv("data/inventory_nyc.csv.gz",
                     usecols=lambda c: c in RAW_FEATURES + HISTORY_FEATURES + [LABEL, "event_id"])
    X, y, groups = engineer(df), df[LABEL].to_numpy(), df["event_id"].to_numpy()
    rng = np.random.default_rng(0)

    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=4).split(X, y, groups)):
        # Same training subsample as train.py (all floods, 15 % of the rest); the test fold is complete.
        tr = tr[(y[tr] == 1) | (rng.random(len(tr)) < 0.15)]
        for name, cols in variants(list(X.columns)).items():
            m = v2.make_model("hgb", cols, params=params).fit(X.iloc[tr][cols], y[tr])
            p = m.predict_proba(X.iloc[te][cols])[:, 1]
            rows.append({"fold": fold, "variant": name, "PR-AUC": average_precision_score(y[te], p),
                         "ROC-AUC": roc_auc_score(y[te], p), "recall top 2%": budget_recall(p, y[te], 0.02)})
        print(f"fold {fold + 1}/4 done", flush=True)
    res = pd.DataFrame(rows)
    Path("reports/nyc").mkdir(parents=True, exist_ok=True)
    res.to_csv("reports/nyc/drainage_test.csv", index=False)
    summary = res.groupby("variant")[["PR-AUC", "ROC-AUC", "recall top 2%"]].mean()
    summary["PR-AUC folds below complete"] = [
        int((res[res.variant == v].set_index("fold")["PR-AUC"] < res[res.variant == "complete"].set_index("fold")["PR-AUC"]).sum())
        for v in summary.index]
    print(summary.sort_values("PR-AUC", ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
