import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

DATA_FOLDER = Path(r"D:\wc data")

MODEL_PATH = (
    DATA_FOLDER /
    "best_match_prediction_model.joblib"
)

TEAM_DATA_PATH = (
    DATA_FOLDER /
    "team_latest_elo.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

HOME_ELO_ADVANTAGE = 100.0


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model was not found:\n{MODEL_PATH}"
    )

if not TEAM_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Team Elo data was not found:\n{TEAM_DATA_PATH}"
    )


# ============================================================
# 4. LOAD MODEL
# ============================================================

model_bundle = joblib.load(MODEL_PATH)

required_bundle_keys = [
    "model",
    "feature_columns",
    "class_names"
]

missing_bundle_keys = [
    key
    for key in required_bundle_keys
    if key not in model_bundle
]

if missing_bundle_keys:
    raise ValueError(
        "The saved model bundle is missing these items:\n"
        f"{missing_bundle_keys}"
    )

model = model_bundle["model"]

feature_columns = model_bundle[
    "feature_columns"
]

class_names = model_bundle[
    "class_names"
]


# ============================================================
# 5. LOAD LATEST TEAM DATA
# ============================================================

team_data = pd.read_csv(
    TEAM_DATA_PATH
)

required_team_columns = [
    "team",
    "latest_elo",
    "recent_win_rate_5",
    "recent_points_per_game_5",
    "recent_goals_for_5",
    "recent_goals_against_5",
    "recent_goal_difference_5"
]

missing_team_columns = [
    column
    for column in required_team_columns
    if column not in team_data.columns
]

if missing_team_columns:
    raise ValueError(
        "The team Elo file is missing these columns:\n"
        f"{missing_team_columns}"
    )

team_data["team"] = (
    team_data["team"]
    .astype(str)
    .str.strip()
)

if team_data["team"].duplicated().any():

    duplicate_teams = (
        team_data.loc[
            team_data["team"].duplicated(),
            "team"
        ]
        .tolist()
    )

    raise ValueError(
        "Duplicate team names were found:\n"
        f"{duplicate_teams}"
    )

team_data = team_data.set_index(
    "team"
)


# ============================================================
# 6. HELPER FUNCTION: TEAM CHECK
# ============================================================

def validate_teams(team_a, team_b):
    """
    Confirm that both teams exist in the latest Elo table.
    """

    if team_a == team_b:
        raise ValueError(
            "A team cannot play against itself."
        )

    if team_a not in team_data.index:
        raise ValueError(
            f"Team not found: {team_a}\n\n"
            "Check the spelling exactly as it appears "
            "in team_latest_elo.csv."
        )

    if team_b not in team_data.index:
        raise ValueError(
            f"Team not found: {team_b}\n\n"
            "Check the spelling exactly as it appears "
            "in team_latest_elo.csv."
        )


# ============================================================
# 7. HELPER FUNCTION: ELO EXPECTATION
# ============================================================

def calculate_expected_home_result(
    home_elo,
    away_elo,
    neutral=True
):
    """
    Calculate the expected Elo result for the home-listed team.

    The result ranges between 0 and 1.
    """

    if neutral:
        home_advantage = 0.0
    else:
        home_advantage = HOME_ELO_ADVANTAGE

    adjusted_home_elo = (
        home_elo +
        home_advantage
    )

    expected_home_result = 1.0 / (
        1.0 +
        10.0 ** (
            (
                away_elo -
                adjusted_home_elo
            ) / 400.0
        )
    )

    return expected_home_result


# ============================================================
# 8. BUILD PRE-MATCH FEATURES
# ============================================================

def build_match_features(
    home_team,
    away_team,
    neutral=True
):
    """
    Build the exact features required by the trained model.

    For neutral matches, no home advantage is applied.
    """

    validate_teams(
        home_team,
        away_team
    )

    home = team_data.loc[
        home_team
    ]

    away = team_data.loc[
        away_team
    ]

    home_elo = float(
        home["latest_elo"]
    )

    away_elo = float(
        away["latest_elo"]
    )

    if neutral:
        home_advantage_flag = 0
    else:
        home_advantage_flag = 1

    elo_difference = (
        home_elo -
        away_elo
    )

    adjusted_elo_difference = (
        home_elo +
        home_advantage_flag *
        HOME_ELO_ADVANTAGE -
        away_elo
    )

    expected_home_elo_result = (
        calculate_expected_home_result(
            home_elo=home_elo,
            away_elo=away_elo,
            neutral=neutral
        )
    )

    expected_away_elo_result = (
        1.0 -
        expected_home_elo_result
    )

    home_win_rate = float(
        home["recent_win_rate_5"]
    )

    away_win_rate = float(
        away["recent_win_rate_5"]
    )

    home_points_per_game = float(
        home["recent_points_per_game_5"]
    )

    away_points_per_game = float(
        away["recent_points_per_game_5"]
    )

    home_goals_for = float(
        home["recent_goals_for_5"]
    )

    away_goals_for = float(
        away["recent_goals_for_5"]
    )

    home_goals_against = float(
        home["recent_goals_against_5"]
    )

    away_goals_against = float(
        away["recent_goals_against_5"]
    )

    home_goal_difference = float(
        home["recent_goal_difference_5"]
    )

    away_goal_difference = float(
        away["recent_goal_difference_5"]
    )

    feature_row = {

        # Match setting
        "neutral":
            int(neutral),

        "home_advantage":
            home_advantage_flag,

        # Elo strength
        "home_elo":
            home_elo,

        "away_elo":
            away_elo,

        "elo_difference":
            elo_difference,

        "adjusted_elo_difference":
            adjusted_elo_difference,

        "expected_home_elo_result":
            expected_home_elo_result,

        "expected_away_elo_result":
            expected_away_elo_result,

        # Win-rate form
        "home_win_rate_5":
            home_win_rate,

        "away_win_rate_5":
            away_win_rate,

        "win_rate_difference_5":
            home_win_rate -
            away_win_rate,

        # Points-per-game form
        "home_points_per_game_5":
            home_points_per_game,

        "away_points_per_game_5":
            away_points_per_game,

        "points_per_game_difference_5":
            home_points_per_game -
            away_points_per_game,

        # Scoring form
        "home_goals_for_5":
            home_goals_for,

        "away_goals_for_5":
            away_goals_for,

        "goals_for_difference_5":
            home_goals_for -
            away_goals_for,

        # Defensive form
        "home_goals_against_5":
            home_goals_against,

        "away_goals_against_5":
            away_goals_against,

        "goals_against_difference_5":
            home_goals_against -
            away_goals_against,

        # Goal-difference form
        "home_goal_difference_5":
            home_goal_difference,

        "away_goal_difference_5":
            away_goal_difference,

        "recent_goal_difference_gap_5":
            home_goal_difference -
            away_goal_difference
    }

    match_features = pd.DataFrame(
        [feature_row]
    )

    missing_features = [
        column
        for column in feature_columns
        if column not in match_features.columns
    ]

    if missing_features:
        raise ValueError(
            "These trained-model features could not be created:\n"
            f"{missing_features}"
        )

    # Preserve the exact feature order used during training.
    match_features = match_features[
        feature_columns
    ]

    return match_features


# ============================================================
# 9. GET MODEL CLASS ORDER
# ============================================================

def get_model_classes():
    """
    Return the target-class order used by the classifier.
    """

    if hasattr(model, "classes_"):
        return model.classes_

    if (
        hasattr(model, "named_steps") and
        "model" in model.named_steps and
        hasattr(
            model.named_steps["model"],
            "classes_"
        )
    ):
        return (
            model
            .named_steps["model"]
            .classes_
        )

    raise AttributeError(
        "The trained model does not expose class labels."
    )


# ============================================================
# 10. RAW ORIENTATION-BASED PREDICTION
# ============================================================

def get_raw_probabilities(
    home_team,
    away_team,
    neutral=True
):
    """
    Predict one match orientation without printing a report.

    Target labels:
    0 = Away Win
    1 = Draw
    2 = Home Win
    """

    match_features = build_match_features(
        home_team=home_team,
        away_team=away_team,
        neutral=neutral
    )

    probabilities = model.predict_proba(
        match_features
    )[0]

    model_classes = get_model_classes()

    probability_by_class = {}

    for class_value, probability in zip(
        model_classes,
        probabilities
    ):
        probability_by_class[
            int(class_value)
        ] = float(probability)

    expected_classes = {
        0,
        1,
        2
    }

    if set(probability_by_class) != expected_classes:
        raise ValueError(
            "Unexpected model classes were found:\n"
            f"{sorted(probability_by_class)}"
        )

    return {
        "away_win":
            probability_by_class[0],

        "draw":
            probability_by_class[1],

        "home_win":
            probability_by_class[2]
    }


# ============================================================
# 11. REGULAR HOME-VENUE PREDICTION
# ============================================================

def predict_home_match(
    home_team,
    away_team
):
    """
    Predict a match where the first team has home advantage.
    """

    probabilities = get_raw_probabilities(
        home_team=home_team,
        away_team=away_team,
        neutral=False
    )

    home_win_probability = (
        probabilities["home_win"]
    )

    draw_probability = (
        probabilities["draw"]
    )

    away_win_probability = (
        probabilities["away_win"]
    )

    result_probabilities = {
        f"{home_team} win":
            home_win_probability,

        "Draw":
            draw_probability,

        f"{away_team} win":
            away_win_probability
    }

    most_likely_result = max(
        result_probabilities,
        key=result_probabilities.get
    )

    print("\n" + "=" * 70)
    print("HOME-VENUE MATCH PREDICTION")
    print("=" * 70)

    print(
        f"\nMatch: {home_team} vs {away_team}"
    )

    print(
        f"Venue: {home_team} home venue"
    )

    print("\nTeam strength:")

    print(
        f"{home_team} Elo:",
        round(
            float(
                team_data.loc[
                    home_team,
                    "latest_elo"
                ]
            ),
            2
        )
    )

    print(
        f"{away_team} Elo:",
        round(
            float(
                team_data.loc[
                    away_team,
                    "latest_elo"
                ]
            ),
            2
        )
    )

    print("\nProbabilities:")

    print(
        f"{home_team} win:",
        f"{home_win_probability:.2%}"
    )

    print(
        "Draw:",
        f"{draw_probability:.2%}"
    )

    print(
        f"{away_team} win:",
        f"{away_win_probability:.2%}"
    )

    print(
        "\nMost likely result:",
        most_likely_result
    )

    return {
        "home_team":
            home_team,

        "away_team":
            away_team,

        "home_win_probability":
            home_win_probability,

        "draw_probability":
            draw_probability,

        "away_win_probability":
            away_win_probability,

        "most_likely_result":
            most_likely_result
    }


# ============================================================
# 12. SYMMETRICAL NEUTRAL-MATCH PREDICTION
# ============================================================

def predict_neutral_match(
    team_a,
    team_b,
    show_details=True
):
    """
    Predict a neutral match in both orientations.

    Orientation 1:
    Team A is stored as the home-listed team.

    Orientation 2:
    Team B is stored as the home-listed team.

    Equivalent outcomes are averaged to reduce artificial
    home/away-label bias.
    """

    validate_teams(
        team_a,
        team_b
    )

    # --------------------------------------------------------
    # Orientation 1: Team A listed first
    # --------------------------------------------------------

    prediction_ab = get_raw_probabilities(
        home_team=team_a,
        away_team=team_b,
        neutral=True
    )

    # --------------------------------------------------------
    # Orientation 2: Team B listed first
    # --------------------------------------------------------

    prediction_ba = get_raw_probabilities(
        home_team=team_b,
        away_team=team_a,
        neutral=True
    )

    # --------------------------------------------------------
    # Average equivalent outcomes
    # --------------------------------------------------------

    team_a_win_probability = (
        prediction_ab["home_win"] +
        prediction_ba["away_win"]
    ) / 2.0

    team_b_win_probability = (
        prediction_ab["away_win"] +
        prediction_ba["home_win"]
    ) / 2.0

    draw_probability = (
        prediction_ab["draw"] +
        prediction_ba["draw"]
    ) / 2.0

    # Normalize to guarantee that the total is exactly 1.
    total_probability = (
        team_a_win_probability +
        draw_probability +
        team_b_win_probability
    )

    if total_probability <= 0:
        raise ValueError(
            "Invalid probability total was produced."
        )

    team_a_win_probability /= total_probability
    draw_probability /= total_probability
    team_b_win_probability /= total_probability

    result_probabilities = {
        f"{team_a} win":
            team_a_win_probability,

        "Draw":
            draw_probability,

        f"{team_b} win":
            team_b_win_probability
    }

    most_likely_result = max(
        result_probabilities,
        key=result_probabilities.get
    )

    if show_details:

        print("\n" + "=" * 70)
        print(
            "SYMMETRICAL NEUTRAL-MATCH PREDICTION"
        )
        print("=" * 70)

        print(
            f"\nMatch: {team_a} vs {team_b}"
        )

        print("Venue: Neutral")

        print("\nTeam strength:")

        print(
            f"{team_a} Elo:",
            round(
                float(
                    team_data.loc[
                        team_a,
                        "latest_elo"
                    ]
                ),
                2
            )
        )

        print(
            f"{team_b} Elo:",
            round(
                float(
                    team_data.loc[
                        team_b,
                        "latest_elo"
                    ]
                ),
                2
            )
        )

        print("\nFirst model orientation:")

        print(
            f"{team_a} win:",
            f"{prediction_ab['home_win']:.2%}"
        )

        print(
            "Draw:",
            f"{prediction_ab['draw']:.2%}"
        )

        print(
            f"{team_b} win:",
            f"{prediction_ab['away_win']:.2%}"
        )

        print("\nReversed model orientation:")

        print(
            f"{team_a} win:",
            f"{prediction_ba['away_win']:.2%}"
        )

        print(
            "Draw:",
            f"{prediction_ba['draw']:.2%}"
        )

        print(
            f"{team_b} win:",
            f"{prediction_ba['home_win']:.2%}"
        )

        print("\nFinal averaged probabilities:")

        print(
            f"{team_a} win:",
            f"{team_a_win_probability:.2%}"
        )

        print(
            "Draw:",
            f"{draw_probability:.2%}"
        )

        print(
            f"{team_b} win:",
            f"{team_b_win_probability:.2%}"
        )

        print(
            "\nMost likely result:",
            most_likely_result
        )

    return {
        "team_a":
            team_a,

        "team_b":
            team_b,

        "team_a_win_probability":
            team_a_win_probability,

        "draw_probability":
            draw_probability,

        "team_b_win_probability":
            team_b_win_probability,

        "most_likely_result":
            most_likely_result,

        "first_orientation":
            prediction_ab,

        "reversed_orientation":
            prediction_ba
    }


# ============================================================
# 13. SIMPLE TEAM-LIST FUNCTION
# ============================================================

def show_available_teams():
    """
    Display every available team alphabetically.
    """

    available_teams = sorted(
        team_data.index.tolist()
    )

    print("\n" + "=" * 70)
    print("AVAILABLE TEAMS")
    print("=" * 70)

    for team in available_teams:
        print(team)


# ============================================================
# 14. TEST PREDICTION
# ============================================================

if __name__ == "__main__":

    prediction = predict_neutral_match(
        team_a="Argentina",
        team_b="France",
        show_details=True
    )
