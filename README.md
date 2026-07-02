# Thiago Engine

Thiago Engine is a local Streamlit app for building fair probabilities and value signals from historical soccer data. The app does not use Pinnacle, paid odds APIs, or scraping. It uses only your own match history data and Monte Carlo simulations.

## Installation

1. Open a terminal in this project folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Live schedule and Kalshi integration

The app supports an optional World Cup schedule API or a schedule upload file. If you enable `Use live World Cup schedule API`, enter a compatible schedule API URL in the sidebar.

You can also provide an optional Kalshi market API URL and API token to fetch live market implied probabilities. If live pricing is unavailable, use the manual Kalshi implied probability input instead.

## CSV data format

The historical matches CSV must include these columns:

- `date` (ISO format like `2024-03-18`)
- `home_team`
- `away_team`
- `home_goals`
- `away_goals`
- `neutral` (`yes` / `no` / `true` / `false`)
- `competition`

Optional advanced event and team metrics are strongly encouraged. If present, the app will use them to improve strength estimates:

- `home_shots`, `away_shots`
- `home_shots_on_target`, `away_shots_on_target`
- `home_corners`, `away_corners`
- `home_fouls`, `away_fouls`
- `home_possession`, `away_possession`
- `home_pass_accuracy`, `away_pass_accuracy`
- `home_xg`, `away_xg`
- `home_goal_scorers`, `away_goal_scorers`

If no CSV is uploaded, the app uses `data/demo_historical_matches.csv`.

All match choices are selected from dropdowns, and all market inputs are chosen with sliders or menu selection. There is no need to type team names manually.

## How it works

Thiago builds fair probabilities from historical match data using:

- recency weighting with a half-life
- attack and defense strength estimates
- simple Elo-style ratings
- home advantage adjustments
- Poisson Monte Carlo simulations

The app compares Thiago fair probabilities to Kalshi prices and computes an edge.

## Understanding market flow

- Markets move based on expected goals, team quality, lineup news, and when information becomes public.
- Regulation win and advance lines are most sensitive to match balance and relative team strength.
- Totals and BTTS markets often react later to starting lineups, injuries, and weather.
- Correct score markets are narrow; a strong edge there should be handled cautiously.
- Home advantage and neutral-site settings change the model's goal expectations and shift probabilities.

Use the app as a quantitative reference point, not a final decision.

## Important notes

- This is a model-only tool, not a guaranteed betting system.
- Do not place wagers automatically.
- The demo data is for testing only and is not strong enough for serious decisions.
- Stronger historical data and calibration improve the model.
- Use the manual team adjustment sliders to reflect lineup quality, fitness, or tactical changes.
- The model confidence score is based on historical match volume and weighted sample size.
- Always sanity-check lineups, injuries, and news before acting on any edge.

## Disclaimer

This code is for educational and research purposes only. There is no guarantee of profit.
