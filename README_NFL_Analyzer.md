# NFL Stats Analyzer (Python)

This is a mid-level technical project that loads player game logs, engineers features, and visualizes how player statistics relate to team outcomes such as **team points**, **point differential**, and **win rate**.

## What's included
- **nfl_stats_analyzer.py** — standalone script
- **nfl_analyzer_outputs/sample_nfl_games.csv** — synthetic dataset for quick testing
- Example charts saved to **nfl_analyzer_outputs/**

## Quickstart

**Option A: Use the sample dataset** (no internet required)
```bash
python nfl_stats_analyzer.py --csv nfl_analyzer_outputs/sample_nfl_games.csv --out nfl_analyzer_outputs --metric PASSING
```

**Option B: Generate a new synthetic dataset**
```bash
python nfl_stats_analyzer.py --make-sample --out nfl_analyzer_outputs --metric RECEIVING
```

This will produce:
- Correlation bar charts (e.g., `passing_team_points_correlations.png`)
- Scatter + fit plots (e.g., `passing_passer_eff_vs_points.png`)
- Win-rate-by-quartile plots
- A CSV **impact report** summarizing correlations

## Using real data
If you have a CSV with columns like `season, week, team, opponent, player, position, passing_yards, passing_tds, interceptions, sacks_taken, rushing_yards, rushing_tds, receptions, receiving_yards, targets, fumbles, team_points, opponent_points`, point the script to it with `--csv`.

> Tip: You can extend the script to pull data via libraries like `nfl_data_py` or to add advanced metrics (EPA/play) and model-based feature importance (e.g., random forests).

## Notes
- Plots are built with **matplotlib** (one chart per figure; no specific styles/colors set).
- The script favors **clear, explainable metrics** for resume readability (correlations, simple linear fits, and quartile win rates).