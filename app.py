from pathlib import Path
from io import StringIO
from collections import Counter

import math
import numpy as np
import streamlit as st
import pandas as pd

COMPETITION_WEIGHTS = {
    "friendly": 0.6,
    "international friendly": 0.6,
    "world cup qualifier": 1.0,
    "euro qualifier": 1.0,
    "nations league": 0.9,
    "gold cup": 0.95,
    "championship": 0.85,
    "league cup": 0.8,
}

OPTIONAL_NUMERIC_COLUMNS = [
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
    "home_possession",
    "away_possession",
    "home_pass_accuracy",
    "away_pass_accuracy",
    "home_xg",
    "away_xg",
]


def competition_weight(competition):
    key = str(competition).strip().lower()
    return float(COMPETITION_WEIGHTS.get(key, 0.8))


def load_matches(df):
    if isinstance(df, pd.DataFrame):
        matches = df.copy()
    else:
        raise ValueError("Expected a pandas DataFrame for load_matches")

    missing = [col for col in [
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "neutral",
        "competition",
    ] if col not in matches.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    if matches["date"].isna().any():
        raise ValueError("All rows must have a valid date in the 'date' column.")

    matches["home_goals"] = pd.to_numeric(matches["home_goals"], errors="coerce")
    matches["away_goals"] = pd.to_numeric(matches["away_goals"], errors="coerce")
    if matches[["home_goals", "away_goals"]].isna().any().any():
        raise ValueError("All rows must have numeric values for home_goals and away_goals.")

    matches["neutral"] = matches["neutral"].astype(str).str.lower().isin([
        "yes", "true", "1", "y", "t"
    ])

    for col in OPTIONAL_NUMERIC_COLUMNS:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce").fillna(0.0)

    return matches


def _recency_weight(date, last_date, half_life_days):
    days = (last_date - date).days
    return 0.5 ** (days / half_life_days)


def build_team_model(matches, half_life_days=90, home_advantage_goals=0.25):
    matches = load_matches(matches)
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")

    last_date = matches["date"].max()
    matches = matches.sort_values("date").reset_index(drop=True)
    matches["competition_weight"] = matches["competition"].astype(str).str.lower().apply(competition_weight)
    matches["weight"] = matches.apply(
        lambda row: _recency_weight(row["date"], last_date, half_life_days) * row["competition_weight"],
        axis=1,
    )

    total_weight = matches["weight"].sum()
    avg_home_goals = (matches["home_goals"] * matches["weight"]).sum() / total_weight
    avg_away_goals = (matches["away_goals"] * matches["weight"]).sum() / total_weight
    avg_neutral_goals = (avg_home_goals + avg_away_goals) / 2

    teams = sorted(set(matches["home_team"]).union(matches["away_team"]))
    stats = {
        "home_attack": {team: 1.0 for team in teams},
        "away_attack": {team: 1.0 for team in teams},
        "home_defense": {team: 1.0 for team in teams},
        "away_defense": {team: 1.0 for team in teams},
        "elo": {team: 1500.0 for team in teams},
        "team_stats": {
            team: {
                "raw_matches": 0,
                "weighted_matches": 0.0,
                "goals_for": 0.0,
                "goals_against": 0.0,
            }
            for team in teams
        },
    }

    attack_home_sum = Counter()
    attack_away_sum = Counter()
    defense_home_sum = Counter()
    defense_away_sum = Counter()
    weight_home = Counter()
    weight_away = Counter()

    team_xg_home = Counter()
    team_xg_away = Counter()
    team_goal_eff_home = Counter()
    team_goal_eff_away = Counter()
    team_shot_ratio_home = Counter()
    team_shot_ratio_away = Counter()
    team_possession_home = Counter()
    team_possession_away = Counter()
    team_pass_acc_home = Counter()
    team_pass_acc_away = Counter()

    for _, row in matches.iterrows():
        weight = row["weight"]
        home = row["home_team"]
        away = row["away_team"]

        attack_home_sum[home] += row["home_goals"] * weight
        attack_away_sum[away] += row["away_goals"] * weight

        defense_home_sum[home] += row["away_goals"] * weight
        defense_away_sum[away] += row["home_goals"] * weight

        weight_home[home] += weight
        weight_away[away] += weight

        stats["team_stats"][home]["raw_matches"] += 1
        stats["team_stats"][home]["weighted_matches"] += weight
        stats["team_stats"][home]["goals_for"] += row["home_goals"]
        stats["team_stats"][home]["goals_against"] += row["away_goals"]

        stats["team_stats"][away]["raw_matches"] += 1
        stats["team_stats"][away]["weighted_matches"] += weight
        stats["team_stats"][away]["goals_for"] += row["away_goals"]
        stats["team_stats"][away]["goals_against"] += row["home_goals"]

        if "home_xg" in matches.columns and "away_xg" in matches.columns:
            team_xg_home[home] += row["home_xg"] * weight
            team_xg_away[away] += row["away_xg"] * weight
            team_goal_eff_home[home] += (row["home_goals"] / max(row["home_xg"], 0.6)) * weight
            team_goal_eff_away[away] += (row["away_goals"] / max(row["away_xg"], 0.6)) * weight
        if "home_shots" in matches.columns and "away_shots" in matches.columns:
            team_shot_ratio_home[home] += (row["home_shots"] / max(row["away_shots"], 1)) * weight
            team_shot_ratio_away[away] += (row["away_shots"] / max(row["home_shots"], 1)) * weight
        if "home_possession" in matches.columns and "away_possession" in matches.columns:
            team_possession_home[home] += row["home_possession"] * weight
            team_possession_away[away] += row["away_possession"] * weight
        if "home_pass_accuracy" in matches.columns and "away_pass_accuracy" in matches.columns:
            team_pass_acc_home[home] += row["home_pass_accuracy"] * weight
            team_pass_acc_away[away] += row["away_pass_accuracy"] * weight

    recent_period = pd.Timedelta(days=90)
    recent_matches = matches[matches["date"] >= last_date - recent_period]

    def _recent_factors(team, home=True):
        subset = recent_matches[recent_matches["home_team"] == team] if home else recent_matches[recent_matches["away_team"] == team]
        if subset.empty:
            return 1.0, 1.0
        if home:
            avg_scored = subset["home_goals"].mean()
            avg_conceded = subset["away_goals"].mean()
            attack_norm = avg_scored / max(avg_home_goals, 0.1)
            defense_norm = avg_conceded / max(avg_away_goals, 0.1)
        else:
            avg_scored = subset["away_goals"].mean()
            avg_conceded = subset["home_goals"].mean()
            attack_norm = avg_scored / max(avg_away_goals, 0.1)
            defense_norm = avg_conceded / max(avg_home_goals, 0.1)
        return attack_norm, defense_norm

    for team in teams:
        if weight_home[team] > 0:
            base_home_attack = max(0.3, (attack_home_sum[team] / weight_home[team]) / avg_home_goals)
            base_home_defense = max(0.3, (defense_home_sum[team] / weight_home[team]) / avg_away_goals)
        else:
            base_home_attack = 1.0
            base_home_defense = 1.0
        if weight_away[team] > 0:
            base_away_attack = max(0.3, (attack_away_sum[team] / weight_away[team]) / avg_away_goals)
            base_away_defense = max(0.3, (defense_away_sum[team] / weight_away[team]) / avg_home_goals)
        else:
            base_away_attack = 1.0
            base_away_defense = 1.0

        raw_matches = max(1, stats["team_stats"][team]["raw_matches"])
        stats["team_stats"][team]["avg_goals_for"] = float(stats["team_stats"][team]["goals_for"] / raw_matches)
        stats["team_stats"][team]["avg_goals_against"] = float(stats["team_stats"][team]["goals_against"] / raw_matches)
        stats["team_stats"][team]["goal_diff"] = float(stats["team_stats"][team]["avg_goals_for"] - stats["team_stats"][team]["avg_goals_against"])

        if weight_home[team] > 0:
            xg_factor = 1.0
            shot_factor = 1.0
            possession_factor = 1.0
            pass_factor = 1.0
            if team_xg_home[team] > 0:
                xg_factor = max(0.7, (team_xg_home[team] / weight_home[team]) / max(0.7, stats.get("avg_home_xg", 1.0)))
            if team_shot_ratio_home[team] > 0:
                shot_factor = max(0.7, team_shot_ratio_home[team] / max(0.7, stats.get("avg_shot_ratio", 1.0)))
            if team_possession_home[team] > 0:
                possession_factor = max(0.7, (team_possession_home[team] / weight_home[team]) / 50.0)
            if team_pass_acc_home[team] > 0:
                pass_factor = max(0.7, (team_pass_acc_home[team] / weight_home[team]) / 80.0)
            efficiency_factor = 1.0
            if team_xg_home[team] > 0:
                efficiency_factor = max(0.7, min(1.3, (attack_home_sum[team] / max(team_xg_home[team], 1e-6))))
            conversion_factor = 1.0
            if team_goal_eff_home[team] > 0:
                conversion_factor = max(0.75, min(1.25, team_goal_eff_home[team] / max(stats.get("avg_home_conversion", 1.0), 0.7)))
            form_attack, form_defense = _recent_factors(team, home=True)
            stats["home_attack"][team] = (
                base_home_attack * 0.6
                + form_attack * 0.25
                + 0.1 * (xg_factor - 1.0)
            ) * efficiency_factor * conversion_factor
            stats["home_defense"][team] = base_home_defense * 0.8 + form_defense * 0.2
        if weight_away[team] > 0:
            xg_factor = 1.0
            shot_factor = 1.0
            possession_factor = 1.0
            pass_factor = 1.0
            if team_xg_away[team] > 0:
                xg_factor = max(0.7, (team_xg_away[team] / weight_away[team]) / max(0.7, stats.get("avg_away_xg", 1.0)))
            if team_shot_ratio_away[team] > 0:
                shot_factor = max(0.7, team_shot_ratio_away[team] / max(0.7, stats.get("avg_shot_ratio", 1.0)))
            if team_possession_away[team] > 0:
                possession_factor = max(0.7, (team_possession_away[team] / weight_away[team]) / 50.0)
            if team_pass_acc_away[team] > 0:
                pass_factor = max(0.7, (team_pass_acc_away[team] / weight_away[team]) / 80.0)
            efficiency_factor = 1.0
            if team_xg_away[team] > 0:
                efficiency_factor = max(0.7, min(1.3, (attack_away_sum[team] / max(team_xg_away[team], 1e-6))))
            conversion_factor = 1.0
            if team_goal_eff_away[team] > 0:
                conversion_factor = max(0.75, min(1.25, team_goal_eff_away[team] / max(stats.get("avg_away_conversion", 1.0), 0.7)))
            form_attack, form_defense = _recent_factors(team, home=False)
            stats["away_attack"][team] = (
                base_away_attack * 0.6
                + form_attack * 0.25
                + 0.1 * (xg_factor - 1.0)
            ) * efficiency_factor * conversion_factor
            stats["away_defense"][team] = base_away_defense * 0.8 + form_defense * 0.2

    if "home_xg" in matches.columns and "away_xg" in matches.columns:
        stats["avg_home_xg"] = (matches["home_xg"] * matches["weight"]).sum() / total_weight
        stats["avg_away_xg"] = (matches["away_xg"] * matches["weight"]).sum() / total_weight
        stats["avg_home_conversion"] = (
            ((matches["home_goals"] / matches["home_xg"].replace(0, 0.6)) * matches["weight"]) 
        ).sum() / total_weight
        stats["avg_away_conversion"] = (
            ((matches["away_goals"] / matches["away_xg"].replace(0, 0.6)) * matches["weight"]) 
        ).sum() / total_weight
    if "home_shots" in matches.columns and "away_shots" in matches.columns:
        stats["avg_shot_ratio"] = (
            (matches["home_shots"] / matches["away_shots"].replace(0, 1)) * matches["weight"]
        ).sum() / total_weight
    if "home_possession" in matches.columns and "away_possession" in matches.columns:
        stats["avg_home_possession"] = (matches["home_possession"] * matches["weight"]).sum() / total_weight
        stats["avg_away_possession"] = (matches["away_possession"] * matches["weight"]).sum() / total_weight
    if "home_pass_accuracy" in matches.columns and "away_pass_accuracy" in matches.columns:
        stats["avg_home_pass_accuracy"] = (matches["home_pass_accuracy"] * matches["weight"]).sum() / total_weight
        stats["avg_away_pass_accuracy"] = (matches["away_pass_accuracy"] * matches["weight"]).sum() / total_weight

    recent_period = pd.Timedelta(days=90)
    recent_matches = matches[matches["date"] >= last_date - recent_period]

    elo = {team: 1500.0 for team in teams}
    k_factor = 25.0
    home_elo_bias = 50.0
    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_score = row["home_goals"]
        away_score = row["away_goals"]
        weight = row["weight"]
        result = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
        expected = 1 / (1 + 10 ** (((elo[away] - (elo[home] + home_elo_bias)) / 400)))
        delta = k_factor * weight * (result - expected)
        elo[home] += delta
        elo[away] -= delta

    stats["elo"] = elo
    stats["avg_home_goals"] = avg_home_goals
    stats["avg_away_goals"] = avg_away_goals
    stats["avg_neutral_goals"] = avg_neutral_goals
    stats["home_advantage_goals"] = home_advantage_goals
    stats["teams"] = teams
    stats["base_elo"] = 1500.0
    stats["data_confidence"] = float(min(1.0, total_weight / 50.0))
    stats["data_confidence_flag"] = (
        "Strong" if total_weight >= 50 else "Moderate" if total_weight >= 20 else "Weak"
    )
    return stats


def expected_goals(model, team_a, team_b, neutral, team_a_adjustment=0.0, team_b_adjustment=0.0):
    if team_a not in model["teams"] or team_b not in model["teams"]:
        raise ValueError("Teams must exist in the historical dataset.")

    if neutral:
        attack_a = np.mean([model["home_attack"][team_a], model["away_attack"][team_a]])
        defense_b = np.mean([model["home_defense"][team_b], model["away_defense"][team_b]])
        attack_b = np.mean([model["home_attack"][team_b], model["away_attack"][team_b]])
        defense_a = np.mean([model["home_defense"][team_a], model["away_defense"][team_a]])
        base = model["avg_neutral_goals"]
        lambda_a = base * attack_a * defense_b
        lambda_b = base * attack_b * defense_a
    else:
        attack_a = model["home_attack"][team_a]
        defense_b = model["away_defense"][team_b]
        attack_b = model["away_attack"][team_b]
        defense_a = model["home_defense"][team_a]
        lambda_a = model["avg_home_goals"] * attack_a * defense_b + model["home_advantage_goals"]
        lambda_b = model["avg_away_goals"] * attack_b * defense_a

    elo_diff = model["elo"][team_a] - model["elo"][team_b]
    elo_adjust = math.tanh(elo_diff / 400) * 0.15
    lambda_a *= 1 + elo_adjust
    lambda_b *= 1 - elo_adjust
    lambda_a *= 1.0 + team_a_adjustment
    lambda_b *= 1.0 + team_b_adjustment
    lambda_a = max(0.05, float(lambda_a))
    lambda_b = max(0.05, float(lambda_b))
    return lambda_a, lambda_b


def _score_array_to_probabilities(scores):
    counter = Counter(scores)
    total = sum(counter.values())
    if total == 0:
        return {}
    return {score: count / total for score, count in counter.items()}


def _bucket_probabilities(values, max_bucket):
    counts = Counter(min(int(x), max_bucket) for x in values)
    total = len(values)
    result = {}
    for i in range(max_bucket):
        result[str(i)] = float(counts[i] / total)
    result[f"{max_bucket}+"] = float(counts[max_bucket] / total)
    return result


def simulate_match(
    model,
    team_a,
    team_b,
    neutral,
    n_sims=5000,
    draw_in_advance_to_fav=0.5,
    team_a_adjustment=0.0,
    team_b_adjustment=0.0,
    random_seed=None,
):
    if random_seed is not None:
        np.random.seed(random_seed)

    lambda_a, lambda_b = expected_goals(
        model,
        team_a,
        team_b,
        neutral,
        team_a_adjustment=team_a_adjustment,
        team_b_adjustment=team_b_adjustment,
    )
    shared_lambda = max(0.0, min(0.15 * min(lambda_a, lambda_b), 0.4, min(lambda_a, lambda_b) - 0.05))
    if shared_lambda > 0.0:
        shared = np.random.poisson(lam=shared_lambda, size=n_sims)
        goals_a = np.random.poisson(lam=np.clip(lambda_a - shared_lambda, 0.05, None), size=n_sims) + shared
        goals_b = np.random.poisson(lam=np.clip(lambda_b - shared_lambda, 0.05, None), size=n_sims) + shared
    else:
        goals_a = np.random.poisson(lam=lambda_a, size=n_sims)
        goals_b = np.random.poisson(lam=lambda_b, size=n_sims)

    wins_a = goals_a > goals_b
    wins_b = goals_b > goals_a
    draws = goals_a == goals_b
    total_goals = goals_a + goals_b

    prob_win_a = float(np.mean(wins_a))
    prob_draw = float(np.mean(draws))
    prob_win_b = float(np.mean(wins_b))
    prob_team_a_minus_1_5 = float(np.mean(goals_a >= goals_b + 2))
    prob_team_b_minus_1_5 = float(np.mean(goals_b >= goals_a + 2))
    prob_team_a_over_1_5 = float(np.mean(goals_a >= 2))
    prob_team_b_over_0_5 = float(np.mean(goals_b >= 1))
    prob_over_2_5 = float(np.mean(total_goals > 2.5))
    prob_under_2_5 = float(np.mean(total_goals <= 2.5))
    prob_over_3_5 = float(np.mean(total_goals > 3.5))
    prob_under_1_5 = float(np.mean(total_goals <= 1.5))
    prob_btts_yes = float(np.mean((goals_a >= 1) & (goals_b >= 1)))
    prob_btts_no = 1.0 - prob_btts_yes

    elo_diff = model["elo"][team_a] - model["elo"][team_b]
    draw_tiebreaker = float(np.clip(draw_in_advance_to_fav, 0.0, 1.0))
    draw_advantage = 0.5 + 0.5 * math.tanh(elo_diff / 200) * draw_tiebreaker
    draw_advantage = float(np.clip(draw_advantage, 0.01, 0.99))
    prob_advance_a = prob_win_a + prob_draw * draw_advantage
    prob_advance_b = 1.0 - prob_advance_a

    score_tuples = list(zip(goals_a, goals_b))
    score_probs = _score_array_to_probabilities(score_tuples)
    top_scores = sorted(score_probs.items(), key=lambda item: item[1], reverse=True)[:10]
    most_likely_scores = [
        {"score": f"{a}-{b}", "probability": float(prob)}
        for (a, b), prob in top_scores
    ]

    margin = goals_a - goals_b
    margin_probs = {
        f"{team_a} +3+": float(np.mean(margin >= 3)),
        f"{team_a} +2": float(np.mean(margin == 2)),
        f"{team_a} +1": float(np.mean(margin == 1)),
        "Draw": float(np.mean(margin == 0)),
        f"{team_b} +1": float(np.mean(margin == -1)),
        f"{team_b} +2": float(np.mean(margin == -2)),
        f"{team_b} +3+": float(np.mean(margin <= -3)),
    }

    team_a_goal_probs = _bucket_probabilities(goals_a, 3)
    team_b_goal_probs = _bucket_probabilities(goals_b, 3)
    total_goal_probs = _bucket_probabilities(total_goals, 4)

    goal_correlation = float(
        np.corrcoef(goals_a, goals_b)[0, 1] if len(goals_a) > 1 else 0.0
    )
    corner_total = float(max(6.0, 6.0 + 2.0 * (lambda_a + lambda_b)))
    corner_share_a = float(lambda_a / max(lambda_a + lambda_b, 0.01))
    corner_advantage = (
        f"{team_a}" if corner_share_a > 0.55 else f"{team_b}" if corner_share_a < 0.45 else "Even"
    )
    likely_first_goal = (
        team_a if lambda_a > lambda_b + 0.1 else team_b if lambda_b > lambda_a + 0.1 else "Either team"
    )
    goal_bias = (
        team_a if lambda_a > lambda_b else team_b if lambda_b > lambda_a else "Balanced"
    )
    totals_path = "Over 2.5" if prob_over_2_5 >= 0.5 else "Under 2.5"
    btts_path = "BTTS Yes" if prob_btts_yes >= 0.5 else "BTTS No"
    if prob_win_a > prob_win_b and prob_win_a > prob_draw:
        result_path = f"{team_a} win"
    elif prob_win_b > prob_win_a and prob_win_b > prob_draw:
        result_path = f"{team_b} win"
    else:
        result_path = "Draw"

    team_a_data = model["team_stats"][team_a]
    team_b_data = model["team_stats"][team_b]
    confidence = float(
        min(
            1.0,
            math.sqrt(team_a_data["weighted_matches"] * team_b_data["weighted_matches"]) / 25.0,
        )
    )

    basket = {
        "aligned_markets": [],
        "moderate_markets": [],
        "watch_markets": [],
    }

    if prob_win_a >= 0.55:
        basket["aligned_markets"].append("Team A regulation win")
    elif prob_win_a >= 0.45:
        basket["moderate_markets"].append("Team A regulation win")
    else:
        basket["watch_markets"].append("Team A regulation win")

    if prob_win_b >= 0.55:
        basket["aligned_markets"].append("Team B regulation win")
    elif prob_win_b >= 0.45:
        basket["moderate_markets"].append("Team B regulation win")
    else:
        basket["watch_markets"].append("Team B regulation win")

    if prob_draw >= 0.30:
        basket["aligned_markets"].append("Draw")
    elif prob_draw >= 0.20:
        basket["moderate_markets"].append("Draw")
    else:
        basket["watch_markets"].append("Draw")

    if prob_advance_a >= 0.55:
        basket["aligned_markets"].append("Team A advance")
    elif prob_advance_a >= 0.45:
        basket["moderate_markets"].append("Team A advance")
    else:
        basket["watch_markets"].append("Team A advance")

    if prob_over_2_5 >= 0.55:
        basket["aligned_markets"].append("Over 2.5 goals")
    elif prob_over_2_5 >= 0.45:
        basket["moderate_markets"].append("Over 2.5 goals")
    else:
        basket["watch_markets"].append("Over 2.5 goals")

    if prob_under_2_5 >= 0.55:
        basket["aligned_markets"].append("Under 2.5 goals")
    elif prob_under_2_5 >= 0.45:
        basket["moderate_markets"].append("Under 2.5 goals")
    else:
        basket["watch_markets"].append("Under 2.5 goals")

    if prob_btts_yes >= 0.55:
        basket["aligned_markets"].append("BTTS Yes")
    elif prob_btts_yes >= 0.45:
        basket["moderate_markets"].append("BTTS Yes")
    else:
        basket["watch_markets"].append("BTTS Yes")

    results = {
        "lambda_a": float(lambda_a),
        "lambda_b": float(lambda_b),
        "shared_lambda": float(shared_lambda),
        "goal_correlation": goal_correlation,
        "expected_goals": {
            team_a: float(lambda_a),
            team_b: float(lambda_b),
        },
        "monte_carlo": {
            "n_sims": int(n_sims),
            "seed": random_seed,
            "team_a_goal_probs": team_a_goal_probs,
            "team_b_goal_probs": team_b_goal_probs,
            "total_goal_probs": total_goal_probs,
            "margin_probs": margin_probs,
            "avg_goals_a": float(np.mean(goals_a)),
            "avg_goals_b": float(np.mean(goals_b)),
            "avg_total_goals": float(np.mean(total_goals)),
            "std_goals_a": float(np.std(goals_a)),
            "std_goals_b": float(np.std(goals_b)),
            "std_total_goals": float(np.std(total_goals)),
            "draw_rate": prob_draw,
            "clean_sheet_a_rate": float(np.mean(goals_b == 0)),
            "clean_sheet_b_rate": float(np.mean(goals_a == 0)),
        },
        "model_confidence": confidence,
        "team_a_history": team_a_data,
        "team_b_history": team_b_data,
        "system_summary": {
            "result": result_path,
            "totals": totals_path,
            "btts": btts_path,
            "corner_total": round(corner_total, 1),
            "corner_advantage": corner_advantage,
            "likely_first_goal": likely_first_goal,
            "goal_bias": goal_bias,
            "goalscorer_bias": f"More likely from {goal_bias} based on attack strength."
        },
        "basket": basket,
        "regulation_win_a": prob_win_a,
        "regulation_draw": prob_draw,
        "regulation_win_b": prob_win_b,
        "advance_a": prob_advance_a,
        "advance_b": prob_advance_b,
        "team_a_minus_1_5": prob_team_a_minus_1_5,
        "team_b_minus_1_5": prob_team_b_minus_1_5,
        "over_2_5": prob_over_2_5,
        "under_2.5": prob_under_2_5,
        "over_3_5": prob_over_3_5,
        "under_1_5": prob_under_1_5,
        "btts_yes": prob_btts_yes,
        "btts_no": prob_btts_no,
        "team_a_over_1.5": prob_team_a_over_1_5,
        "team_b_over_0.5": prob_team_b_over_0_5,
        "correct_score_probs": score_probs,
        "most_likely_scores": most_likely_scores,
        "average_goals_a": float(np.mean(goals_a)),
        "average_goals_b": float(np.mean(goals_b)),
        "average_total_goals": float(np.mean(total_goals)),
        "model_stats": {
            "elo_a": float(model["elo"][team_a]),
            "elo_b": float(model["elo"][team_b]),
            "home_advantage_goals": float(model["home_advantage_goals"])
        },
        "n_sims": int(n_sims),
    }

    return results


def market_insight(market, simulation, team_a, team_b, neutral):
    text = []
    exp_a = simulation["expected_goals"][team_a]
    exp_b = simulation["expected_goals"][team_b]
    total = exp_a + exp_b
    is_home = not neutral

    if market == "Team A advance":
        text.append(
            f"The model sees {team_a} with an advance probability of {simulation['advance_a']:.1%}. "
            f"If a draw is likely, the higher Elo team benefits from the tiebreaker strength."
        )
    elif market == "Team A regulation win":
        text.append(
            f"The model estimates {team_a} wins in regulation about {simulation['regulation_win_a']:.1%} of simulations. "
            "Regulation markets often respond to expected goal differentials and relative team momentum."
        )
    elif market == "Team A -1.5":
        text.append(
            f"The team A -1.5 probability is based on the chance {team_a} scores 2+ goals, currently {simulation['team_a_minus_1_5']:.1%}. "
            "This market is usually more attractive when team A has both offensive strength and a weaker away defense."
        )
    elif market == "Over 2.5 goals":
        text.append(
            f"Expected total goals are {total:.2f}. Over 2.5 is estimated at {simulation['over_2_5']:.1%}. "
            "Totals markets typically move on attacking form and lineup risk."
        )
    elif market == "Under 2.5 goals":
        text.append(
            f"Expected total goals are {total:.2f}. Under 2.5 is estimated at {simulation['under_2.5']:.1%}. "
            "Low-scoring markets are likely when both defenses are strong and goal expectation is below 2.5."
        )
    elif market == "BTTS Yes":
        text.append(
            f"Both teams score in {simulation['btts_yes']:.1%} of simulated matches. "
            "BTTS flow often depends on whether both lineups are likely to attack and concede."
        )
    elif market == "BTTS No":
        text.append(
            f"No BTTS is estimated at {simulation['btts_no']:.1%}. "
            "This is generally stronger when both teams are defensive or a low open play tempo is expected."
        )
    elif market == "Team A over 1.5 goals":
        text.append(
            f"The model estimates {team_a} scores at least 2 goals in {simulation['team_a_over_1.5']:.1%} of simulations. "
            "This market is sensitive to team A's scoring form and the opponent's defensive shape."
        )
    elif market == "Team B over 0.5 goals":
        text.append(
            f"The model estimates {team_b} scores at least 1 goal in {simulation['team_b_over_0.5']:.1%} of simulations. "
            "A reliable line if the underdog can create enough chances or if team A is not fully defensively compact."
        )
    elif market == "Correct score":
        top = simulation["most_likely_scores"][0] if simulation["most_likely_scores"] else None
        if top:
            text.append(
                f"The most likely scoreline is {top['score']} at {top['probability']:.1%}. "
                "Correct score markets are very narrow, so a strong edge should be treated cautiously."
            )
        else:
            text.append("No clear correct score probability is available from the simulation.")
    else:
        text.append("The model is comparing fair probability to market price. Use this as a context check.")

    if is_home:
        text.append("Home advantage is active, so the home team is expected to score slightly more.")
    else:
        text.append("Neutral conditions remove the home bias and rely more on team quality and Elo difference.")

    return " ".join(text)


def fair_decimal(prob):
    if prob <= 0:
        return float("inf")
    return round(1.0 / prob, 3)


def compare_to_kalshi(fair_prob, kalshi_prob):
    return fair_prob - kalshi_prob


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
2023-10-08,France,England,1,2,no,UEFA Nations League,13,16,5,7,6,5,11,10,52,48,84,82,1.3,1.7,Benzema,"Kane,Mount"
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
            "Thiago fair odds": fair_decimal(probability),
        }
        if kalshi_prob is not None:
            edge = compare_to_kalshi(probability, kalshi_prob)
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
        matches_df = load_matches(matches_df)
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
    st.info(
        "The model uses the full dataset to train, but the interface focuses on the selected match and recent team form. "
        "Full historical data is hidden behind the preview section."
    )
    with st.expander("Preview full historical dataset"):
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

    relevant_recent = matches_df[
        (matches_df["home_team"].isin([team_a, team_b]))
        | (matches_df["away_team"].isin([team_a, team_b]))
    ].sort_values("date", ascending=False).reset_index(drop=True)
    if not relevant_recent.empty:
        with st.expander("Show recent matches for the selected teams"):
            st.write(
                "These are the most recent games for the selected home and away teams. "
                "Older match history is hidden by default to keep the analysis focused."
            )
            st.dataframe(relevant_recent.head(10))

    market = st.selectbox("Select market", MARKET_OPTIONS)

    model = build_team_model(
        matches_df, half_life_days=half_life_days, home_advantage_goals=home_advantage_goals
    )
    simulation = simulate_match(
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
        fair_prob = simulation["team_a_over_1_5"]
    elif market == "Team B over 0.5 goals":
        fair_prob = simulation["team_b_over_0_5"]
    elif market == "Correct score" and selected_score is not None:
        fair_prob = simulation["correct_score_probs"].get(
            tuple(int(x) for x in selected_score.split("-")), 0.0
        )
    else:
        fair_prob = 0.0

    fair_decimal_value = fair_decimal(fair_prob)
    edge = compare_to_kalshi(fair_prob, kalshi_price_cents / 100.0)
    edge_pct = edge * 100.0
    verdict = verdict_from_edge(edge_pct)
    insight = market_insight(market, simulation, team_a, team_b, neutral)

    st.header("Thiago value output")
    st.metric("Thiago fair probability", format_percentage(fair_prob))
    st.metric("Fair decimal odds", fair_decimal_value)
    st.metric("Kalshi probability", format_percentage(kalshi_price_cents / 100.0))
    st.metric("Edge (percentage points)", f"{edge_pct:+.2f}")
    st.write(f"**Verdict:** {verdict}")
    st.info(
        "This is a model-only edge. Always do lineup, injury, and news sanity checks before placing any wager."
    )
    st.markdown("**Market insight:**")
    st.write(insight)

    st.subheader("All market fair odds")
    market_table = build_market_summary(simulation, kalshi_price_cents / 100.0)
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
