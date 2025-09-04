#!/usr/bin/env python3
"""
NFL Stats Analyzer — Real Data Edition
--------------------------------------
Adds real data ingestion via nfl_data_py and a feature-importance chart.

New flags:
    --real                 Use nfl_data_py to fetch weekly player stats and schedules
    --seasons 2022 2023    Seasons to pull when --real is set (default: latest 2)
    --importance-outcome   Outcome for feature importance: team_points or point_diff (default: team_points)

Install requirements (once):
    pip install nfl_data_py scikit-learn pandas numpy matplotlib

Examples
--------
# Real data, passing focus, last two seasons
python nfl_stats_analyzer.py --real --metric PASSING --out nfl_outputs

# Real data with explicit seasons
python nfl_stats_analyzer.py --real --seasons 2023 2024 --metric RECEIVING --out nfl_outputs

# Synthetic data (no internet):
python nfl_stats_analyzer.py --make-sample --out nfl_outputs --metric RUSHING
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_sample_csv(path: str, n_seasons: int = 2, weeks: int = 12, seed: int = 42):
    rng = np.random.default_rng(seed)
    teams = ["SF", "DAL", "KC"]
    rows = []
    for s in range(2023, 2023 + n_seasons):
        for w in range(1, weeks+1):
            for t in teams:
                opp_choices = [x for x in teams if x != t]
                opp = rng.choice(opp_choices)

                # QB row
                passing_yards = max(0, rng.normal(240, 50))
                passing_tds = max(0, int(rng.normal(1.8, 1)))
                interceptions = max(0, int(rng.normal(0.6, 0.7)))
                sacks_taken = max(0, int(rng.normal(2.0, 1.0)))
                rows.append([s,w,t,opp,f"QB_{t}","QB",
                             passing_yards,passing_tds,interceptions,sacks_taken,
                             0,0,0,0,0,0,0,0])

                # RBs
                for i in range(2):
                    ry = max(0, rng.normal(55 if i==0 else 35, 25))
                    rt = max(0, int(rng.normal(0.3, 0.5)))
                    rows.append([s,w,t,opp,f"RB{i+1}_{t}","RB",
                                 0,0,0,0,ry,rt,0,0,0,0,0,0])
                # WRs (3)
                for i in range(3):
                    rec = max(0, int(rng.normal(4 if i==0 else 3, 2)))
                    tgt = max(rec, int(rec + rng.integers(0,3)))
                    ryy = max(0, rng.normal(60 if i==0 else (45 if i==1 else 35), 30))
                    rows.append([s,w,t,opp,f"WR{i+1}_{t}","WR",
                                 0,0,0,0,0,0,rec,ryy,tgt,0,0,0])

                # Estimate points
                team_base = {"SF": 26, "KC": 25, "DAL": 24}[t]
                opp_base = {"SF": 20, "KC": 21, "DAL": 22}[opp]
                # Aggregate for team-week
                tw = [(r) for r in rows if r[0]==s and r[1]==w and r[2]==t]
                total_rec_yards = sum(r[13] for r in tw)
                total_rush_yards = sum(r[10] for r in tw)
                offensive_score = 0.06*passing_yards + 1.8*passing_tds - 1.0*interceptions + 0.03*total_rush_yards + 0.035*total_rec_yards - 0.2*sacks_taken
                team_points = max(0, int(round(team_base + offensive_score + rng.normal(0,3))))
                opp_points = max(0, int(round(opp_base + rng.normal(0,5))))
                # backfill
                for r in rows:
                    if r[0]==s and r[1]==w and r[2]==t and r[16]==0 and r[17]==0:
                        r[16] = team_points
                        r[17] = opp_points

    cols = [
        "season","week","team","opponent","player","position",
        "passing_yards","passing_tds","interceptions","sacks_taken",
        "rushing_yards","rushing_tds",
        "receptions","receiving_yards","targets",
        "fumbles","team_points","opponent_points"
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, index=False)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["point_diff"] = df["team_points"] - df["opponent_points"]
    df["win"] = (df["point_diff"] > 0).astype(int)
    df["yards_per_target"] = np.where(df["targets"].fillna(0) > 0,
                                      df["receiving_yards"] / df["targets"], np.nan)
    df["rush_td_rate"] = np.where((df["rushing_yards"].fillna(0) + 1) > 0,
                                  df["rushing_tds"] / (df["rushing_yards"] + 1), np.nan)
    df["passer_eff"] = (0.05*df["passing_yards"].fillna(0)
                        + 3.0*df["passing_tds"].fillna(0)
                        - 4.0*df["interceptions"].fillna(0)
                        - 0.5*df["sacks_taken"].fillna(0))
    return df

def correlation_impact(df: pd.DataFrame, feature_cols, outcome_col: str):
    corrs = {}
    subset = df[feature_cols + [outcome_col]].dropna()
    for col in feature_cols:
        if subset[col].nunique() > 1:
            corrs[col] = subset[[col, outcome_col]].corr().iloc[0,1]
        else:
            corrs[col] = np.nan
    return pd.Series(corrs).sort_values(ascending=False)

def win_rate_by_quartile(df: pd.DataFrame, feature: str):
    d = df[[feature,"win"]].dropna().copy()
    if d.empty:
        return pd.DataFrame(columns=["quartile","feature_min","feature_max","win_rate","n"])
    d["quartile"] = pd.qcut(d[feature], q=4, labels=[1,2,3,4])
    agg = d.groupby("quartile").agg(
        win_rate=("win","mean"),
        n=("win","size"),
        feature_min=(feature,"min"),
        feature_max=(feature,"max")
    ).reset_index()
    return agg

def plot_scatter_with_fit(df, x, y, title, out_path):
    d = df[[x, y]].dropna()
    if len(d) < 5:
        print(f"[warn] Not enough data to plot {x} vs {y}")
        return
    plt.figure()
    plt.scatter(d[x], d[y], alpha=0.6)
    try:
        coeffs = np.polyfit(d[x], d[y], 1)
        xp = np.linspace(d[x].min(), d[x].max(), 100)
        yp = coeffs[0]*xp + coeffs[1]
        plt.plot(xp, yp, linewidth=2)
    except Exception as e:
        print(f"[warn] Could not fit line: {e}")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_bar(ser: pd.Series, title: str, out_path: str, top_k: int = 10):
    s = ser.dropna().head(top_k)
    plt.figure()
    plt.bar(s.index.astype(str), s.values)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Correlation with outcome")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_quartile_winrate(df_q, title, out_path):
    if df_q.empty:
        print(f("[warn] No data for quartile plot: {title}"))
        return
    plt.figure()
    plt.plot(df_q["quartile"].astype(int), df_q["win_rate"], marker="o")
    plt.title(title)
    plt.xlabel("Feature quartile (1=low, 4=high)")
    plt.ylabel("Win rate")
    plt.xticks([1,2,3,4])
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def feature_importance_chart(df, feature_cols, outcome_col, out_path):
    # Train a simple RandomForestRegressor to estimate feature importance.
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    d = df[feature_cols + [outcome_col]].dropna()
    if d.empty or d[feature_cols].shape[1] == 0:
        print("[warn] Not enough data for feature importance")
        return
    X = d[feature_cols].values
    y = d[outcome_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
    model = RandomForestRegressor(n_estimators=400, random_state=7)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure()
    plt.bar([feature_cols[i] for i in order], importances[order])
    plt.title(f"Feature importance for predicting {outcome_col} (RF, R^2={r2:.2f})")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def load_real_data(seasons):
    """
    Load weekly player stats and team scores via nfl_data_py, then align to our expected schema.
    Requires internet access on first run.
    """
    try:
        from nfl_data_py import import_weekly_data, import_schedules
    except Exception as e:
        raise SystemExit("Please install nfl_data_py: pip install nfl_data_py") from e

    weekly = import_weekly_data(seasons)
    # Ensure expected numeric columns exist (fill missing with 0)
    defaults = {
        "passing_yards": 0.0, "passing_tds": 0.0, "interceptions": 0.0, "sacks": 0.0,
        "rushing_yards": 0.0, "rushing_tds": 0.0,
        "receptions": 0.0, "receiving_yards": 0.0, "targets": 0.0, "fumbles": 0.0
    }
    for k,v in defaults.items():
        if k not in weekly.columns:
            # Try common name variants
            if k == "sacks" and "sacks_taken" in weekly.columns:
                weekly["sacks"] = weekly["sacks_taken"]
            else:
                weekly[k] = v

    # Team & opponent points from schedules, joined by season, week, team
    sched = import_schedules(seasons)
    # Build a tidy team-week score table using home/away rows
    home = sched[["season","week","home_team","away_team","home_score","away_score"]].rename(
        columns={"home_team":"team","away_team":"opponent","home_score":"team_points","away_score":"opponent_points"}
    )
    away = sched[["season","week","home_team","away_team","home_score","away_score"]].rename(
        columns={"away_team":"team","home_team":"opponent","away_score":"team_points","home_score":"opponent_points"}
    )
    tw_scores = pd.concat([home, away], ignore_index=True)

    # Map weekly data columns to our schema
    df = weekly.rename(columns={
        "player_name":"player",
        "position":"position",
        "team":"team",
        "opponent_team":"opponent",
        "sacks":"sacks_taken"
    })

    # Some datasets use 'recent_team' instead of 'team'
    if "team" not in df.columns and "recent_team" in df.columns:
        df["team"] = df["recent_team"]

    # Merge team scores
    df = df.merge(tw_scores, on=["season","week","team"], how="left")

    expected_cols = [
        "season","week","team","opponent","player","position",
        "passing_yards","passing_tds","interceptions","sacks_taken",
        "rushing_yards","rushing_tds",
        "receptions","receiving_yards","targets",
        "fumbles","team_points","opponent_points"
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[expected_cols]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, help="Path to player game CSV")
    ap.add_argument("--out", type=str, default="nfl_outputs", help="Output folder")
    ap.add_argument("--make-sample", action="store_true", help="Generate synthetic sample CSV")
    ap.add_argument("--metric", type=str, choices=["PASSING","RECEIVING","RUSHING"], default="PASSING",
                    help="Primary focus for visuals")
    ap.add_argument("--real", action="store_true", help="Fetch real data via nfl_data_py (requires internet)")
    ap.add_argument("--seasons", nargs="+", type=int, help="Seasons to fetch with --real (e.g., 2023 2024)")
    ap.add_argument("--importance-outcome", type=str, choices=["team_points","point_diff"], default="team_points")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.real:
        seasons = args.seasons if args.seasons else list(range(2023, 2025))
        print(f"[info] Loading real weekly data for seasons: {seasons}")
        df = load_real_data(seasons)
    elif args.make-sample:
        csv_path = os.path.join(args.out, "sample_nfl_games.csv")
        df = make_sample_csv(csv_path)
        print(f"[info] Sample data written to: {csv_path}")
    else:
        if not args.csv or not os.path.exists(args.csv):
            raise SystemExit("Provide --csv path, --make-sample, or --real.")
        df = pd.read_csv(args.csv)

    # Feature engineering
    df = engineer_features(df)

    # Select features per metric focus
    if args.metric == "PASSING":
        features = ["passer_eff","passing_yards","passing_tds","interceptions","sacks_taken"]
        metric_name = "Passing metrics"
    elif args.metric == "RECEIVING":
        features = ["yards_per_target","receiving_yards","receptions","targets"]
        metric_name = "Receiving metrics"
    else:
        features = ["rushing_yards","rushing_tds","rush_td_rate"]
        metric_name = "Rushing metrics"

    # Correlations vs outcomes
    for outcome in ["team_points","point_diff"]:
        corrs = correlation_impact(df, features, outcome)
        out_png = os.path.join(args.out, f"{args.metric.lower()}_{outcome}_correlations.png")
        plot_bar(corrs, f"{metric_name}: correlation vs {outcome}", out_png)
        print(f"[info] Saved: {out_png}")

    # Scatter with fit for the primary feature
    primary = {"PASSING": "passer_eff", "RECEIVING": "yards_per_target", "RUSHING": "rushing_yards"}[args.metric]
    out_scatter = os.path.join(args.out, f"{args.metric.lower()}_{primary}_vs_points.png")
    plot_scatter_with_fit(df, primary, "team_points", f"{primary} vs team_points", out_scatter)
    print(f"[info] Saved: {out_scatter}")

    # Win rate by quartile
    wr_q = win_rate_by_quartile(df, primary)
    out_wr = os.path.join(args.out, f"{args.metric.lower()}_{primary}_winrate_quartiles.png")
    plot_quartile_winrate(wr_q, f"Win rate by {primary} quartiles", out_wr)
    print(f"[info] Saved: {out_wr}")

    # NEW: Feature importance chart (RandomForest on chosen outcome)
    importance_outcome = args.importance_outcome
    out_imp = os.path.join(args.out, f"{args.metric.lower()}_{importance_outcome}_feature_importance.png")
    try:
        feature_importance_chart(df, features, importance_outcome, out_imp)
        print(f"[info] Saved: {out_imp}")
    except Exception as e:
        print(f"[warn] Feature importance failed: {e}")

    # Impact report CSV
    report_rows = []
    for outcome in ["team_points","point_diff"]:
        corrs = correlation_impact(df, features, outcome)
        for k,v in corrs.items():
            report_rows.append({"metric_focus": args.metric, "feature": k, "outcome": outcome, "correlation": v})
    report_df = pd.DataFrame(report_rows)
    report_path = os.path.join(args.out, f"{args.metric.lower()}_impact_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"[info] Saved report: {report_path}")

if __name__ == "__main__":
    main()
