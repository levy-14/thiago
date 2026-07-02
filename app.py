import sys
from pathlib import Path
from io import StringIO
import importlib.util

import streamlit as st
import pandas as pd

def load_model_engine():
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    sys_path_dirs = [Path(p) for p in sys.path if p and Path(p).exists()]
    candidates = []

    # Common deployment roots and repository layouts
    candidates.extend([
        current_dir,
        current_dir.parent,
        current_dir / "src",
        current_dir.parent / "src",
        current_dir / "thiago",
        current_dir.parent / "thiago",
        Path.cwd(),
        Path.cwd().parent,
    ])

    # Include all parent folders to catch nested deploy roots
    candidates.extend(current_dir.parents)
    candidates.extend(Path.cwd().resolve().parents)
    candidates.extend(sys_path_dirs)

    # Preserve order and remove duplicates
    ordered_candidates = []
    for path in candidates:
        if path not in ordered_candidates:
            ordered_candidates.append(path)

    searched_paths = []
    for base in ordered_candidates:
        for path in [
            base / "model_engine.py",
            base / "thiago" / "model_engine.py",
            base / "src" / "model_engine.py",
            base / "src" / "thiago" / "model_engine.py",
        ]:
            if path.exists():
                sys.path.insert(0, str(path.parent))
                spec = importlib.util.spec_from_file_location("model_engine", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            searched_paths.append(path)

    # Recursive fallback so nested layouts can still be discovered
    for base in ordered_candidates:
        if base.exists():
            for path in base.rglob("model_engine.py"):
                sys.path.insert(0, str(path.parent))
                spec = importlib.util.spec_from_file_location("model_engine", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    searched_text = "\n".join(str(path) for path in searched_paths[:200])
    debug_dirs = "\n".join(f"{path} (exists={path.exists()})" for path in ordered_candidates[:50])
    raise FileNotFoundError(
        "model_engine.py could not be found in the deployment. Searched these locations:\n"
        f"{searched_text}\n\n"
        "Candidate search directories:\n"
        f"{debug_dirs}\n"
        "If you are deploying to Streamlit Cloud, ensure that `model_engine.py` is included in the app bundle and lives alongside `app.py` or under a discovered source path."
    )

_loaded_model_engine = None


def get_model_engine():
    global _loaded_model_engine
    if _loaded_model_engine is None:
        _loaded_model_engine = load_model_engine()
    return _loaded_model_engine


def _render_loader_diagnostics():
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    candidates = [
        current_dir,
        current_dir.parent,
        Path.cwd(),
        Path.cwd().parent,
    ]
    candidate_info = []
    for candidate in candidates:
        if candidate.exists():
            candidate_info.append(
                f"{candidate}: exists, items={len(list(candidate.iterdir()))}"
            )
        else:
            candidate_info.append(f"{candidate}: does not exist")
    return "\n".join(candidate_info)


MARKET_OPTIONS = [
    "Team A advance",
    "Team A regulation win",
    "Team A -1.5",
    "Team B regulation win",
    "Team B -1.5",
    "Draw",
    "Over 2.5 goals",
    "Under 2.5 goals",
    "Over 3.5 goals",
    "Under 1.5 goals",
    "BTTS Yes",
    "BTTS No",
    "Team A over 1.5 goals",
    "Team B over 0.5 goals",
    "Correct score",
]

DEFAULT_DEMO_CSV = """date,home_team,away_team,home_goals,away_goals,neutral,competition,home_shots,away_shots,home_shots_on_target,away_shots_on_target,home_corners,away_corners,home_fouls,away_fouls,home_possession,away_possession,home_pass_accuracy,away_pass_accuracy,home_xg,away_xg,home_goal_scorers,away_goal_scorers
2023-10-05,Spain,Germany,2,1,no,UEFA Nations League,14,9,6,4,7,3,8,12,59,41,86,79,1.9,1.1,Alvarez,Musiala
2023-10-08,France,England,1,2,no,UEFA Nations League,13,16,5,7,6,5,11,10,52,48,84,82,1.3,1.7,Benzema,Kane,Mount
2023-10-11,Brazil,Argentina,2,0,no,World Cup Qualifier,18,7,8,2,9,2,6,14,62,38,88,76,2.4,0.7,Richarlison,Messi
2023-10-14,Italy,Portugal,0,1,no,European Championship Qualifier,10,12,3,5,4,6,9,11,47,53,79,80,0.6,1.0,Ronaldo,
2023-10-17,USA,Canada,3,1,no,CONCACAF Gold Cup Qualifier,17,10,7,4,8,4,11,13,58,42,81,77,2.1,0.9,Brady,Weah
2023-10-20,Colombia,Uruguay,1,0,no,World Cup Qualifier,11,8,5,3,4,4,12,10,54,46,78,80,1.0,0.6,Borre,
2023-10-23,Spain,Portugal,3,2,no,UEFA Nations League,20,15,10,6,10,5,9,14,61,39,87,78,2.7,1.9,Alvarez,Silva
2023-10-26,France,Belgium,0,1,no,UEFA Nations League,12,14,4,5,6,6,10,11,53,47,82,81,0.8,1.3,DeBruyne,
2023-10-29,Brazil,Spain,1,1,yes,International Friendly,16,13,6,5,8,5,8,9,55,45,85,80,1.4,1.1,Neymar,Alvarez
2023-11-02,England,Germany,2,2,no,UEFA Nations League,18,17,9,8,7,6,12,12,57,43,86,84,2.2,2.0,Kane,Musiala
"""


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


def build_market_summary(simulation, kalshi_prob=None):
    market_map = {
        "Team A advance": "advance_a",
        "Team A regulation win": "regulation_win_a",
        "Team A -1.5": "team_a_minus_1_5",
        "Team B regulation win": "regulation_win_b",
        "Team B -1.5": "team_b_minus_1_5",
        "Draw": "regulation_draw",
        "Over 2.5 goals": "over_2_5",
        "Under 2.5 goals": "under_2.5",
        "Over 3.5 goals": "over_3_5",
        "Under 1.5 goals": "under_1_5",
        "BTTS Yes": "btts_yes",
        "BTTS No": "btts_no",
        "Team A over 1.5 goals": "team_a_over_1.5",
        "Team B over 0.5 goals": "team_b_over_0.5",
    }
    rows = []
    for market, key in market_map.items():
        probability = float(simulation.get(key, 0.0))
        row = {
            "Market": market,
            "Thiago fair probability": probability,
            "Thiago fair decimal": format_percentage(probability),
            "Thiago fair odds": model_engine.fair_decimal(probability),
        }
        if kalshi_prob is not None:
            edge = model_engine.compare_to_kalshi(probability, kalshi_prob)
            row["Edge vs Kalshi"] = f"{edge * 100:+.2f}%"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="Thiago Engine", layout="wide")
    st.title("Thiago Engine")
    st.write(
        "Build fair soccer probabilities from your historical data, compare to Kalshi prices, and identify potential edges."
    )

    # Fixed model settings: match selection drives the output, not manual tuning.
    half_life_days = 90
    home_advantage_goals = 0.25
    draw_tiebreaker = 0.5
    team_a_adjustment = 0.0
    team_b_adjustment = 0.0
    kalshi_price_cents = 50.0
    n_sims = 5000

    st.sidebar.write("Select a match and a market on the main screen. Model tuning is fixed for all games.")

    uploaded_file = st.file_uploader("Upload historical_matches.csv", type=["csv"])
    app_dir = Path(__file__).resolve().parent
    demo_data_path = app_dir / "data" / "demo_historical_matches.csv"

    if uploaded_file is not None:
        try:
            matches_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Unable to read upload: {exc}")
            return
    else:
        if demo_data_path.exists() and demo_data_path.is_file():
            try:
                matches_df = pd.read_csv(demo_data_path)
            except Exception as exc:
                st.error(f"Unable to read demo dataset: {exc}")
                return
        else:
            try:
                matches_df = pd.read_csv(StringIO(DEFAULT_DEMO_CSV))
                st.info("Using built-in demo dataset because the deployed data file is unavailable.")
            except Exception as exc:
                st.error(f"Unable to load built-in demo dataset: {exc}")
                return

    try:
        model_engine = get_model_engine()
    except FileNotFoundError as exc:
        st.error("Unable to load the model engine module.")
        st.code(str(exc))
        st.write("Streamlit deployment may be mounting a different source root than expected.")
        st.write("Current loader diagnostics:")
        st.code(_render_loader_diagnostics())
        return
    except Exception as exc:
        st.error(f"Unexpected model engine load failure: {exc}")
        return

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

    if matches_df.shape[0] == 0:
        st.warning("No matches available in the uploaded dataset.")
        return

    matches_df = matches_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    game_choices = [
        f"{row['date'].date()} — {row['home_team']} vs {row['away_team']} ({row['competition']})"
        for _, row in matches_df.iterrows()
    ]

    st.write(f"Loaded {len(matches_df)} historical matches.")
    with st.expander("Preview historical data"):
        st.dataframe(matches_df.head(10))

    selected_game = st.selectbox("Select a match", game_choices, index=0)
    selected_idx = game_choices.index(selected_game)
    match_row = matches_df.loc[selected_idx]
    team_a = match_row["home_team"]
    team_b = match_row["away_team"]
    neutral = bool(match_row["neutral"])
    st.markdown(
        f"**Selected match:** {match_row['date'].date()} — {team_a} vs {team_b} | "
        f"{match_row['competition']} | {'Neutral' if neutral else 'Home advantage'}"
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
    edge = model_engine.compare_to_kalshi(fair_prob, kalshi_price_cents / 100.0)
    edge_pct = edge * 100.0
    verdict = verdict_from_edge(edge_pct)
    insight = model_engine.market_insight(market, simulation, team_a, team_b, neutral)

    st.header("Thiago value output")
    st.metric("Thiago fair probability", format_percentage(fair_prob))
    st.metric("Fair decimal odds", fair_decimal)
    st.metric("Kalshi probability", format_percentage(kalshi_price_cents / 100.0))
    st.metric("Edge (percentage points)", f"{edge_pct:+.2f}")
    st.write(f"**Verdict:** {verdict}")
    st.info(
        "This is a model-only edge. Always do lineup, injury, and news sanity checks before placing any wager."
    )
    st.markdown("**Market insight:**")
    st.write(insight)

    st.subheader("All market fair odds")
    market_table = build_market_summary(simulation, kalshi_prob)
    st.dataframe(market_table)

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
