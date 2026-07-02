import sys
from pathlib import Path
import importlib.util

import streamlit as st
import pandas as pd

def load_model_engine():
    current_dir = Path(__file__).resolve().parent
    candidates = [
        current_dir,
        current_dir.parent,
        current_dir / "src",
        current_dir.parent / "src",
        Path.cwd(),
        Path.cwd().parent,
    ]

    searched_paths = []
    for base in candidates:
        for path in [
            base / "model_engine.py",
            base / "thiago" / "model_engine.py",
            base / "src" / "model_engine.py",
            base / "src" / "thiago" / "model_engine.py",
        ]:
            if path.exists():
                spec = importlib.util.spec_from_file_location("model_engine", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            searched_paths.append(path)

    for path in current_dir.rglob("model_engine.py"):
        spec = importlib.util.spec_from_file_location("model_engine", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    searched_text = "\n".join(str(path) for path in searched_paths)
    raise FileNotFoundError(
        "model_engine.py could not be found in the deployment. Searched these locations:\n" + searched_text
    )

model_engine = load_model_engine()


MARKET_OPTIONS = [
    "Team A advance",
    "Team A regulation win",
    "Team A -1.5",
    "Team B regulation win",
    "Team B -1.5",
    "Draw",
    "Over 2.5 goals",
    "Under 2.5 goals",
    "BTTS Yes",
    "BTTS No",
    "Team A over 1.5 goals",
    "Team B over 0.5 goals",
    "Correct score",
]


def format_percentage(value):
    return f"{value * 100:.2f}%"


def verdict_from_edge(edge_pct):
    if edge_pct <= 0:
        return "PASS"
    if edge_pct <= 1.5:
        return "PASS"
    if edge_pct <= 3:
        return "WATCH / SMALL ONLY"
    if edge_pct <= 5:
        return "PLAYABLE"
    return "STRONG CONSIDERATION"


def main():
    st.set_page_config(page_title="Thiago Engine", layout="wide")
    st.title("Thiago Engine")
    st.write(
        "Build fair soccer probabilities from your historical data, compare to Kalshi prices, and identify potential edges."
    )

    st.sidebar.header("Model controls")
    half_life_days = st.sidebar.slider(
        "Recency half-life (days)", min_value=30, max_value=365, value=90, step=10
    )
    home_advantage_goals = st.sidebar.slider(
        "Home advantage (goals)", min_value=0.0, max_value=1.0, value=0.25, step=0.05
    )
    draw_tiebreaker = st.sidebar.slider(
        "Draw advance strength", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="How strongly a draw favors the higher Elo team in an advance scenario."
    )
    team_a_adjustment = st.sidebar.slider(
        "Team A adjustment", min_value=-0.25, max_value=0.25, value=0.0, step=0.01,
        help="Adjust expected goals up or down for Team A based on lineup or news."
    )
    team_b_adjustment = st.sidebar.slider(
        "Team B adjustment", min_value=-0.25, max_value=0.25, value=0.0, step=0.01,
        help="Adjust expected goals up or down for Team B based on lineup or news."
    )
    n_sims = st.sidebar.number_input(
        "Monte Carlo simulations", min_value=1000, max_value=20000, value=5000, step=500
    )

    uploaded_file = st.file_uploader("Upload historical_matches.csv", type=["csv"])
    if uploaded_file is not None:
        try:
            matches_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Unable to read upload: {exc}")
            return
    else:
        matches_df = pd.read_csv("data/demo_historical_matches.csv")

    try:
        matches_df = model_engine.load_matches(matches_df)
    except Exception as exc:
        st.error(f"Error loading matches: {exc}")
        return

    if matches_df.empty:
        st.warning("The historical data is empty. Upload a file with matches.")
        return

    advanced_columns = [
        "home_xg",
        "away_xg",
        "home_shots",
        "away_shots",
        "home_possession",
        "away_possession",
        "home_pass_accuracy",
        "away_pass_accuracy",
        "home_corners",
        "away_corners",
    ]
    has_advanced = any(col in matches_df.columns for col in advanced_columns)
    if has_advanced:
        st.success("Advanced match event stats detected. The model will use these to refine team attack/defense strength.")
    else:
        st.info("Upload advanced match stats if available (xG, shots, possession, pass accuracy, corners, fouls) for a stronger model.")

    teams = sorted(set(matches_df["home_team"]).union(matches_df["away_team"]))
    if len(teams) < 2:
        st.warning("Not enough unique teams in data.")
        return

    st.write(f"Loaded {len(matches_df)} historical matches.")
    with st.expander("Preview historical data"):
        st.dataframe(matches_df.head(10))

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", teams, index=0)
    with col2:
        team_b = st.selectbox("Team B", teams, index=1 if len(teams) > 1 else 0)

    if team_a == team_b:
        st.warning("Please choose two different teams for Team A and Team B.")
        return

    neutral = st.radio("Neutral site?", ["Yes", "No"]) == "Yes"
    kalshi_price_cents = st.slider(
        "Kalshi YES price (cents)", min_value=0.0, max_value=100.0, value=50.0, step=0.5
    )
    market = st.selectbox("Select market", MARKET_OPTIONS)

    model = model_engine.build_team_model(
        matches_df, half_life_days=half_life_days, home_advantage_goals=home_advantage_goals
    )
    simulation = model_engine.simulate_match(
        model,
        team_a,
        team_b,
        neutral,
        n_sims=n_sims,
        draw_in_advance_to_fav=draw_tiebreaker,
        team_a_adjustment=team_a_adjustment,
        team_b_adjustment=team_b_adjustment,
        random_seed=42,
    )

    if market == "Correct score":
        score_choices = [entry["score"] for entry in simulation["most_likely_scores"]]
        if score_choices:
            selected_score = st.selectbox("Choose correct score", score_choices)
        else:
            selected_score = None
    else:
        selected_score = None

    kalshi_prob = float(min(max(kalshi_price_cents / 100.0, 0.0), 1.0))
    if market == "Team A advance":
        fair_prob = simulation["advance_a"]
    elif market == "Team A regulation win":
        fair_prob = simulation["regulation_win_a"]
    elif market == "Team B regulation win":
        fair_prob = simulation["regulation_win_b"]
    elif market == "Draw":
        fair_prob = simulation["regulation_draw"]
    elif market == "Team A -1.5":
        fair_prob = simulation["team_a_minus_1_5"]
    elif market == "Team B -1.5":
        fair_prob = simulation["team_b_minus_1_5"]
    elif market == "Over 2.5 goals":
        fair_prob = simulation["over_2_5"]
    elif market == "Under 2.5 goals":
        fair_prob = simulation["under_2.5"]
    elif market == "Over 3.5 goals":
        fair_prob = simulation["over_3_5"]
    elif market == "Under 1.5 goals":
        fair_prob = simulation["under_1_5"]
    elif market == "BTTS Yes":
        fair_prob = simulation["btts_yes"]
    elif market == "BTTS No":
        fair_prob = simulation["btts_no"]
    elif market == "Team A over 1.5 goals":
        fair_prob = simulation["team_a_over_1.5"]
    elif market == "Team B over 0.5 goals":
        fair_prob = simulation["team_b_over_0.5"]
    elif market == "Correct score" and selected_score is not None:
        fair_prob = simulation["correct_score_probs"].get(
            tuple(int(x) for x in selected_score.split("-")), 0.0
        )
    else:
        fair_prob = 0.0

    fair_decimal = model_engine.fair_decimal(fair_prob)
    edge = model_engine.compare_to_kalshi(fair_prob, kalshi_prob)
    edge_pct = edge * 100.0
    verdict = verdict_from_edge(edge_pct)
    insight = model_engine.market_insight(market, simulation, team_a, team_b, neutral)

    st.header("Thiago value output")
    st.metric("Thiago fair probability", format_percentage(fair_prob))
    st.metric("Fair decimal odds", fair_decimal)
    st.metric("Kalshi probability", format_percentage(kalshi_prob))
    st.metric("Edge (percentage points)", f"{edge_pct:+.2f}")
    st.write(f"**Verdict:** {verdict}")
    st.info(
        "This is a model-only edge. Always do lineup, injury, and news sanity checks before placing any wager."
    )
    st.markdown("**Market insight:**")
    st.write(insight)

    st.subheader("Match forecast summary")
    st.write(
        f"Estimated goals: {team_a} {simulation['expected_goals'][team_a]:.2f} vs "
        f"{team_b} {simulation['expected_goals'][team_b]:.2f}"
    )
    st.write(f"Simulated average total goals: {simulation['average_total_goals']:.2f}")

    if simulation["most_likely_scores"]:
        st.subheader("Most likely scores")
        st.table(
            pd.DataFrame(simulation["most_likely_scores"]).assign(
                probability=lambda df: df["probability"].map("{:.2%}".format)
            )
        )

    st.subheader("Match systems overview")
    st.write(
        f"Result path: {simulation['system_summary']['result']} | "
        f"Total goals path: {simulation['system_summary']['totals']} | "
        f"BTTS path: {simulation['system_summary']['btts']}"
    )
    st.write(
        f"Corner estimate: {simulation['system_summary']['corner_total']} total corners, "
        f"{simulation['system_summary']['corner_advantage']} expected to win the corner count."
    )
    st.write(
        f"Likely first goal side: {simulation['system_summary']['likely_first_goal']} | "
        f"Goal bias: {simulation['system_summary']['goal_bias']}"
    )
    st.write(f"Goal scorer bias: {simulation['system_summary']['goalscorer_bias']}")

    st.subheader("Kalshi-style basket summary")
    st.write(
        "Markets aligned with the model: "
        f"{', '.join(simulation['basket'].get('aligned_markets', [])) or 'None'}"
    )
    st.write(
        "Markets moderately aligned: "
        f"{', '.join(simulation['basket'].get('moderate_markets', [])) or 'None'}"
    )
    st.write(
        "Markets to monitor only: "
        f"{', '.join(simulation['basket'].get('watch_markets', [])) or 'None'}"
    )

    st.subheader("Model diagnostics")
    diagnostics = {
        "Team A Elo": simulation["model_stats"]["elo_a"],
        "Team B Elo": simulation["model_stats"]["elo_b"],
        "Home advantage (goals)": simulation["model_stats"]["home_advantage_goals"],
        "Simulations": simulation["n_sims"],
        "Team A expected goals": simulation["expected_goals"][team_a],
        "Team B expected goals": simulation["expected_goals"][team_b],
        "Team A average simulated goals": simulation["average_goals_a"],
        "Team B average simulated goals": simulation["average_goals_b"],
        "Model confidence": simulation["model_confidence"],
        "Team A weighted matches": simulation["team_a_history"]["weighted_matches"],
        "Team B weighted matches": simulation["team_b_history"]["weighted_matches"],
        "Team A adjustment": team_a_adjustment,
        "Team B adjustment": team_b_adjustment,
    }
    st.json(diagnostics)

    monte = simulation.get("monte_carlo", {})
    if monte:
        st.subheader("Monte Carlo diagnostics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Shared goal factor", f"{simulation['shared_lambda']:.3f}")
            st.metric("Goal correlation", f"{simulation['goal_correlation']:.3f}")
            st.metric("Draw rate", f"{monte['draw_rate']:.1%}")
            st.metric("Clean sheet A rate", f"{monte['clean_sheet_a_rate']:.1%}")
            st.metric("Clean sheet B rate", f"{monte['clean_sheet_b_rate']:.1%}")
        with col2:
            st.metric("Avg goals A", f"{monte['avg_goals_a']:.2f}")
            st.metric("Avg goals B", f"{monte['avg_goals_b']:.2f}")
            st.metric("Avg total goals", f"{monte['avg_total_goals']:.2f}")
            st.metric("Std goals A", f"{monte['std_goals_a']:.2f}")
            st.metric("Std goals B", f"{monte['std_goals_b']:.2f}")

        with st.expander("Goal distribution probabilities"):
            st.write("Team A goal probabilities")
            st.json(monte["team_a_goal_probs"])
            st.write("Team B goal probabilities")
            st.json(monte["team_b_goal_probs"])
            st.write("Total goals probabilities")
            st.json(monte["total_goal_probs"])

        with st.expander("Margin probabilities"):
            st.json(monte["margin_probs"])

    st.sidebar.write("## Notes")
    st.sidebar.write(
        "The demo dataset is small. Better results require a broader and richer historical sample."
    )


if __name__ == "__main__":
    main()
