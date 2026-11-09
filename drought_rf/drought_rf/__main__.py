import argparse

from .pipeline import run

ap = argparse.ArgumentParser(prog="python -m drought_rf", description="Streamflow drought RF (Heldmyer et al. 2026)")
ap.add_argument("--data", help="folder with streamflow.csv, met.csv, ... (omit for synthetic demo)")
ap.add_argument("--out", default="outputs")
ap.add_argument("--leads", default="0,7,14,30", help="days ahead, comma separated; 0 = paper's nowcast")
ap.add_argument("--models", default="rf,lgbm,logistic,climatology,persistence")
ap.add_argument("--split", choices=["paper", "walk_forward"], default="paper")
ap.add_argument("--use-flow", action="store_true", help="add today's streamflow percentile as a predictor (gauged forecasting)")
ap.add_argument("--threshold", type=float, default=0.5)
a = ap.parse_args()
run(a.data, a.out, tuple(int(x) for x in a.leads.split(",")), tuple(a.models.split(",")),
    a.split, a.use_flow, a.threshold)
