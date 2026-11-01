"""Gradient-boosting hyperparameter search with storm-grouped cross-validation.

    python -m flood_risk.tune --data data/inventory_nyc.csv.gz --out models/nyc/hgb_params.json

Each candidate is trained on every flooded row and `--neg-sample` of the others, and scored by
PR-AUC on the complete held-out storms. The best parameters go to a JSON file that
`flood_risk.train --hgb-params` reads.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from . import v2
from .features import engineer
from .schema import HISTORY_FEATURES, LABEL, RAW_FEATURES
from .train import budget_recall

GRID = {
    "learning_rate": [0.03, 0.08],
    "max_leaf_nodes": [15, 31],
    "min_samples_leaf": [40, 200],
    "max_features": [1.0, 0.5],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg-sample", type=float, default=0.15)
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()

    cols = [c for c in RAW_FEATURES + HISTORY_FEATURES + [LABEL, "event_id"]]
    df = pd.read_csv(args.data, usecols=lambda c: c in cols)
    X, y, groups = engineer(df), df[LABEL].to_numpy(), df["event_id"].to_numpy()
    rng = np.random.default_rng(0)
    folds = []
    for tr, te in GroupKFold(n_splits=args.folds).split(X, y, groups):
        folds.append((tr[(y[tr] == 1) | (rng.random(len(tr)) < args.neg_sample)], te))

    rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        pr, top2 = [], []
        for tr, te in folds:
            m = v2.make_model("hgb", list(X.columns), params=params).fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[te])[:, 1]
            pr.append(average_precision_score(y[te], p))
            top2.append(budget_recall(p, y[te], 0.02))
        rows.append({**params, "pr_auc": np.mean(pr), "pr_auc_std": np.std(pr), "recall_top2": np.mean(top2)})
        print({k: rows[-1][k] for k in [*GRID, "pr_auc", "recall_top2"]}, flush=True)

    res = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    res.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    # Pandas turns the row into floats: restore each parameter's type from the grid.
    best = {k: type(GRID[k][0])(v) for k, v in res.iloc[0][list(GRID)].items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(best, indent=2))
    print("best:", best, "| default:", {k: v2.HGB_DEFAULTS.get(k) for k in GRID})


if __name__ == "__main__":
    main()
