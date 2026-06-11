import json
import joblib
import math
import numpy as np

from pathlib import Path

from predict_match import (
    build_match_features,
    get_raw_probabilities,
    team_data
)


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

DATA_FOLDER = Path(r"D:\wc data")

GOAL_MODEL_PATH = (
    DATA_FOLDER /
    "best_goal_prediction_models.joblib"
)

BEST_WEIGHT_PATH = (
    DATA_FOLDER /
    "best_hybrid_weights.json"
)


# ============================================================
# 2. HOST-COUNTRY INFORMATION
# ============================================================

# Team names must match team_latest_elo.csv.
HOST_TEAM_COUNTRY = {
    "Canada": "Canada",
    "Mexico": "Mexico",
    "United States": "United States"
}

# Accepted alternative country spellings.
COUNTRY_ALIASES = {
    "usa": "United States",
    "u.s.a.": "United States",
    "us": "United States",
    "u.s.": "United States",
    "united states": "United States",
    "united states of america": "United States",

    "canada": "Canada",

    "mexico": "Mexico",
    "méxico": "Mexico"
}


# ============================================================
# 3. SCORE-MODEL SETTINGS
# ============================================================

MAX_GOALS = 10

# Fallback weights are used only when the weight-tuning JSON
# file has not yet been created.
DEFAULT_CLASSIFIER_WEIGHT = 0.60
DEFAULT_POISSON_WEIGHT = 0.40


# ============================================================
# 4. LOAD HYBRID WEIGHTS
# ============================================================

def load_hybrid_weights():
    """
    Load tuned hybrid weights when available.

    Otherwise use:
        Random Forest = 60%
        Poisson       = 40%
    """

    if not BEST_WEIGHT_PATH.exists():

        print(
            "\nWarning: best_hybrid_weights.json was not found."
        )

        print(
            "Using fallback weights: "
            "Classifier 60%, Poisson 40%."
        )

        return {
            "classifier_weight":
                DEFAULT_CLASSIFIER_WEIGHT,

            "poisson_weight":
                DEFAULT_POISSON_WEIGHT,

            "source":
                "Fallback values"
        }

    with open(
        BEST_WEIGHT_PATH,
        "r",
        encoding="utf-8"
    ) as input_file:

        weight_data = json.load(
            input_file
        )

    classifier_weight = float(
        weight_data["classifier_weight"]
    )

    poisson_weight = float(
        weight_data["poisson_weight"]
    )

    total_weight = (
        classifier_weight +
        poisson_weight
    )

    if total_weight <= 0:
        raise ValueError(
            "Hybrid weights must have a positive total."
        )

    classifier_weight /= total_weight
    poisson_weight /= total_weight

    return {
        "classifier_weight":
            classifier_weight,

        "poisson_weight":
            poisson_weight,

        "source":
            str(BEST_WEIGHT_PATH)
    }


HYBRID_WEIGHTS = load_hybrid_weights()

CLASSIFIER_WEIGHT = (
    HYBRID_WEIGHTS[
        "classifier_weight"
    ]
)

POISSON_WEIGHT = (
    HYBRID_WEIGHTS[
        "poisson_weight"
    ]
)


# ============================================================
# 5. LOAD GOAL MODELS
# ============================================================

if not GOAL_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Goal-model file was not found:\n"
        f"{GOAL_MODEL_PATH}"
    )

goal_model_bundle = joblib.load(
    GOAL_MODEL_PATH
)

required_goal_keys = [
    "home_goal_model",
    "away_goal_model",
    "feature_columns"
]

missing_goal_keys = [
    key
    for key in required_goal_keys
    if key not in goal_model_bundle
]

if missing_goal_keys:
    raise ValueError(
        "The goal-model bundle is missing:\n"
        f"{missing_goal_keys}"
    )

home_goal_model = (
    goal_model_bundle[
        "home_goal_model"
    ]
)

away_goal_model = (
    goal_model_bundle[
        "away_goal_model"
    ]
)

goal_feature_columns = (
    goal_model_bundle[
        "feature_columns"
    ]
)

MIN_EXPECTED_GOALS = (
    goal_model_bundle.get(
        "minimum_expected_goals",
        0.05
    )
)

MAX_EXPECTED_GOALS = (
    goal_model_bundle.get(
        "maximum_expected_goals",
        6.00
    )
)


# ============================================================
# 6. TEAM VALIDATION
# ============================================================

def validate_teams(team_a, team_b):
    """
    Confirm that both teams exist in the team-strength file.
    """

    if team_a == team_b:
        raise ValueError(
            "A team cannot play against itself."
        )

    if team_a not in team_data.index:
        raise ValueError(
            f"Team was not found: {team_a}"
        )

    if team_b not in team_data.index:
        raise ValueError(
            f"Team was not found: {team_b}"
        )


# ============================================================
# 7. NORMALIZE VENUE COUNTRY
# ============================================================

def normalize_country_name(
    venue_country
):
    """
    Convert alternative country spellings into the canonical
    country names used by the simulator.
    """

    if venue_country is None:
        return None

    normalized_text = (
        str(venue_country)
        .strip()
        .lower()
    )

    return COUNTRY_ALIASES.get(
        normalized_text,
        str(venue_country).strip()
    )


# ============================================================
# 8. DETERMINE VENUE ADVANTAGE
# ============================================================

def determine_venue_type(
    team_a,
    team_b,
    venue_country
):
    """
    Determine whether a 2026 host team is playing inside its
    own country.

    Possible outputs:
        team_a_home
        team_b_home
        neutral
    """

    normalized_country = (
        normalize_country_name(
            venue_country
        )
    )

    team_a_country = (
        HOST_TEAM_COUNTRY.get(
            team_a
        )
    )

    team_b_country = (
        HOST_TEAM_COUNTRY.get(
            team_b
        )
    )

    team_a_is_home = (
        team_a_country is not None
        and
        team_a_country ==
        normalized_country
    )

    team_b_is_home = (
        team_b_country is not None
        and
        team_b_country ==
        normalized_country
    )

    if (
        team_a_is_home
        and
        not team_b_is_home
    ):
        return "team_a_home"

    if (
        team_b_is_home
        and
        not team_a_is_home
    ):
        return "team_b_home"

    return "neutral"


# ============================================================
# 9. VENUE DESCRIPTION
# ============================================================

def get_venue_description(
    team_a,
    team_b,
    venue_country,
    venue_type
):
    """
    Return a readable venue explanation.
    """

    normalized_country = (
        normalize_country_name(
            venue_country
        )
    )

    if venue_type == "team_a_home":

        return (
            f"{team_a} host-country advantage "
            f"in {normalized_country}"
        )

    if venue_type == "team_b_home":

        return (
            f"{team_b} host-country advantage "
            f"in {normalized_country}"
        )

    if normalized_country is None:

        return "Neutral venue"

    return (
        f"Neutral for both teams "
        f"in {normalized_country}"
    )


# ============================================================
# 10. CLASSIFIER PROBABILITIES
# ============================================================

def get_venue_aware_classifier_probabilities(
    team_a,
    team_b,
    venue_type
):
    """
    Return probabilities in the original fixture order:

        Team A win
        Draw
        Team B win

    Neutral matches use two-orientation averaging.

    Host-country matches use one true home-match orientation.
    """

    # --------------------------------------------------------
    # Neutral match
    # --------------------------------------------------------

    if venue_type == "neutral":

        prediction_ab = get_raw_probabilities(
            home_team=team_a,
            away_team=team_b,
            neutral=True
        )

        prediction_ba = get_raw_probabilities(
            home_team=team_b,
            away_team=team_a,
            neutral=True
        )

        team_a_win_probability = (
            prediction_ab["home_win"]
            +
            prediction_ba["away_win"]
        ) / 2.0

        draw_probability = (
            prediction_ab["draw"]
            +
            prediction_ba["draw"]
        ) / 2.0

        team_b_win_probability = (
            prediction_ab["away_win"]
            +
            prediction_ba["home_win"]
        ) / 2.0

    # --------------------------------------------------------
    # Team A is playing at home
    # --------------------------------------------------------

    elif venue_type == "team_a_home":

        prediction = get_raw_probabilities(
            home_team=team_a,
            away_team=team_b,
            neutral=False
        )

        team_a_win_probability = (
            prediction["home_win"]
        )

        draw_probability = (
            prediction["draw"]
        )

        team_b_win_probability = (
            prediction["away_win"]
        )

    # --------------------------------------------------------
    # Team B is playing at home
    # --------------------------------------------------------

    elif venue_type == "team_b_home":

        # Team B must internally be entered as the home team.
        prediction = get_raw_probabilities(
            home_team=team_b,
            away_team=team_a,
            neutral=False
        )

        # Convert back to original Team A / Team B order.
        team_a_win_probability = (
            prediction["away_win"]
        )

        draw_probability = (
            prediction["draw"]
        )

        team_b_win_probability = (
            prediction["home_win"]
        )

    else:
        raise ValueError(
            f"Unknown venue type: {venue_type}"
        )

    total_probability = (
        team_a_win_probability
        +
        draw_probability
        +
        team_b_win_probability
    )

    return {
        "team_a_win_probability":
            team_a_win_probability /
            total_probability,

        "draw_probability":
            draw_probability /
            total_probability,

        "team_b_win_probability":
            team_b_win_probability /
            total_probability
    }


# ============================================================
# 11. CLIP EXPECTED GOALS
# ============================================================

def clip_expected_goals(value):
    """
    Keep expected goals within the limits used during training.
    """

    return float(
        np.clip(
            value,
            MIN_EXPECTED_GOALS,
            MAX_EXPECTED_GOALS
        )
    )


# ============================================================
# 12. VENUE-AWARE EXPECTED GOALS
# ============================================================

def get_venue_aware_expected_goals(
    team_a,
    team_b,
    venue_type
):
    """
    Return expected goals in the original fixture order.

    Neutral matches:
        Predict both orientations and average them.

    Host matches:
        Use the host as the true home team.
    """

    # --------------------------------------------------------
    # Neutral match
    # --------------------------------------------------------

    if venue_type == "neutral":

        features_ab = build_match_features(
            home_team=team_a,
            away_team=team_b,
            neutral=True
        )

        features_ab = features_ab[
            goal_feature_columns
        ]

        ab_home_expected = (
            clip_expected_goals(
                home_goal_model.predict(
                    features_ab
                )[0]
            )
        )

        ab_away_expected = (
            clip_expected_goals(
                away_goal_model.predict(
                    features_ab
                )[0]
            )
        )

        features_ba = build_match_features(
            home_team=team_b,
            away_team=team_a,
            neutral=True
        )

        features_ba = features_ba[
            goal_feature_columns
        ]

        ba_home_expected = (
            clip_expected_goals(
                home_goal_model.predict(
                    features_ba
                )[0]
            )
        )

        ba_away_expected = (
            clip_expected_goals(
                away_goal_model.predict(
                    features_ba
                )[0]
            )
        )

        team_a_expected_goals = (
            ab_home_expected
            +
            ba_away_expected
        ) / 2.0

        team_b_expected_goals = (
            ab_away_expected
            +
            ba_home_expected
        ) / 2.0

    # --------------------------------------------------------
    # Team A is the true home team
    # --------------------------------------------------------

    elif venue_type == "team_a_home":

        features = build_match_features(
            home_team=team_a,
            away_team=team_b,
            neutral=False
        )

        features = features[
            goal_feature_columns
        ]

        team_a_expected_goals = (
            clip_expected_goals(
                home_goal_model.predict(
                    features
                )[0]
            )
        )

        team_b_expected_goals = (
            clip_expected_goals(
                away_goal_model.predict(
                    features
                )[0]
            )
        )

    # --------------------------------------------------------
    # Team B is the true home team
    # --------------------------------------------------------

    elif venue_type == "team_b_home":

        features = build_match_features(
            home_team=team_b,
            away_team=team_a,
            neutral=False
        )

        features = features[
            goal_feature_columns
        ]

        internal_home_expected = (
            clip_expected_goals(
                home_goal_model.predict(
                    features
                )[0]
            )
        )

        internal_away_expected = (
            clip_expected_goals(
                away_goal_model.predict(
                    features
                )[0]
            )
        )

        # Convert back to original fixture order.
        team_a_expected_goals = (
            internal_away_expected
        )

        team_b_expected_goals = (
            internal_home_expected
        )

    else:
        raise ValueError(
            f"Unknown venue type: {venue_type}"
        )

    return {
        "team_a_expected_goals":
            team_a_expected_goals,

        "team_b_expected_goals":
            team_b_expected_goals
    }


# ============================================================
# 13. POISSON PROBABILITY
# ============================================================

def poisson_probability(
    goals,
    expected_goals
):
    """
    Probability of scoring exactly a specified number of goals.
    """

    if expected_goals <= 0:
        raise ValueError(
            "Expected goals must be positive."
        )

    log_probability = (
        -expected_goals
        +
        goals * math.log(
            expected_goals
        )
        -
        math.lgamma(
            goals + 1
        )
    )

    return math.exp(
        log_probability
    )


# ============================================================
# 14. CREATE SCORE MATRIX
# ============================================================

def create_poisson_score_matrix(
    team_a_expected_goals,
    team_b_expected_goals,
    maximum_goals=MAX_GOALS
):
    """
    Build the probability matrix for all scorelines.
    """

    team_a_probabilities = np.array([
        poisson_probability(
            goals,
            team_a_expected_goals
        )
        for goals in range(
            maximum_goals + 1
        )
    ])

    team_b_probabilities = np.array([
        poisson_probability(
            goals,
            team_b_expected_goals
        )
        for goals in range(
            maximum_goals + 1
        )
    ])

    score_matrix = np.outer(
        team_a_probabilities,
        team_b_probabilities
    )

    score_matrix /= score_matrix.sum()

    return score_matrix


# ============================================================
# 15. SCORE-MATRIX OUTCOME PROBABILITIES
# ============================================================

def calculate_matrix_outcome_probabilities(
    score_matrix
):
    """
    Calculate Team A win, draw and Team B win probabilities.
    """

    team_a_win_probability = 0.0
    draw_probability = 0.0
    team_b_win_probability = 0.0

    number_of_scores = (
        score_matrix.shape[0]
    )

    for team_a_goals in range(
        number_of_scores
    ):

        for team_b_goals in range(
            number_of_scores
        ):

            probability = score_matrix[
                team_a_goals,
                team_b_goals
            ]

            if team_a_goals > team_b_goals:

                team_a_win_probability += (
                    probability
                )

            elif team_a_goals < team_b_goals:

                team_b_win_probability += (
                    probability
                )

            else:

                draw_probability += (
                    probability
                )

    return {
        "team_a_win_probability":
            team_a_win_probability,

        "draw_probability":
            draw_probability,

        "team_b_win_probability":
            team_b_win_probability
    }


# ============================================================
# 16. BLEND MODEL PROBABILITIES
# ============================================================

def blend_outcome_probabilities(
    classifier_probabilities,
    poisson_probabilities
):
    """
    Blend the classifier and Poisson probabilities using the
    tuned hybrid weights.
    """

    blended_team_a_win = (
        CLASSIFIER_WEIGHT
        *
        classifier_probabilities[
            "team_a_win_probability"
        ]
        +
        POISSON_WEIGHT
        *
        poisson_probabilities[
            "team_a_win_probability"
        ]
    )

    blended_draw = (
        CLASSIFIER_WEIGHT
        *
        classifier_probabilities[
            "draw_probability"
        ]
        +
        POISSON_WEIGHT
        *
        poisson_probabilities[
            "draw_probability"
        ]
    )

    blended_team_b_win = (
        CLASSIFIER_WEIGHT
        *
        classifier_probabilities[
            "team_b_win_probability"
        ]
        +
        POISSON_WEIGHT
        *
        poisson_probabilities[
            "team_b_win_probability"
        ]
    )

    total_probability = (
        blended_team_a_win
        +
        blended_draw
        +
        blended_team_b_win
    )

    return {
        "team_a_win_probability":
            blended_team_a_win /
            total_probability,

        "draw_probability":
            blended_draw /
            total_probability,

        "team_b_win_probability":
            blended_team_b_win /
            total_probability
    }


# ============================================================
# 17. ADJUST SCORE MATRIX
# ============================================================

def adjust_score_matrix(
    original_score_matrix,
    target_probabilities
):
    """
    Reweight individual scorelines so that total win, draw and
    loss probabilities equal the hybrid probabilities.
    """

    adjusted_matrix = (
        original_score_matrix.copy()
    )

    original_probabilities = (
        calculate_matrix_outcome_probabilities(
            original_score_matrix
        )
    )

    epsilon = 1e-12

    team_a_scale = (
        target_probabilities[
            "team_a_win_probability"
        ]
        /
        max(
            original_probabilities[
                "team_a_win_probability"
            ],
            epsilon
        )
    )

    draw_scale = (
        target_probabilities[
            "draw_probability"
        ]
        /
        max(
            original_probabilities[
                "draw_probability"
            ],
            epsilon
        )
    )

    team_b_scale = (
        target_probabilities[
            "team_b_win_probability"
        ]
        /
        max(
            original_probabilities[
                "team_b_win_probability"
            ],
            epsilon
        )
    )

    number_of_scores = (
        adjusted_matrix.shape[0]
    )

    for team_a_goals in range(
        number_of_scores
    ):

        for team_b_goals in range(
            number_of_scores
        ):

            if team_a_goals > team_b_goals:

                adjusted_matrix[
                    team_a_goals,
                    team_b_goals
                ] *= team_a_scale

            elif team_a_goals < team_b_goals:

                adjusted_matrix[
                    team_a_goals,
                    team_b_goals
                ] *= team_b_scale

            else:

                adjusted_matrix[
                    team_a_goals,
                    team_b_goals
                ] *= draw_scale

    adjusted_matrix /= (
        adjusted_matrix.sum()
    )

    return adjusted_matrix


# ============================================================
# 18. SAMPLE A SCORELINE
# ============================================================

def sample_scoreline(
    score_matrix,
    random_generator
):
    """
    Randomly select one scoreline from the probability matrix.
    """

    flat_probabilities = (
        score_matrix.flatten()
    )

    selected_position = (
        random_generator.choice(
            len(flat_probabilities),
            p=flat_probabilities
        )
    )

    team_a_goals, team_b_goals = (
        np.unravel_index(
            selected_position,
            score_matrix.shape
        )
    )

    return (
        int(team_a_goals),
        int(team_b_goals)
    )


# ============================================================
# 19. MOST LIKELY SCORELINES
# ============================================================

def get_most_likely_scorelines(
    score_matrix,
    number_of_results=10
):
    """
    Return the most likely exact scorelines.
    """

    scorelines = []

    number_of_scores = (
        score_matrix.shape[0]
    )

    for team_a_goals in range(
        number_of_scores
    ):

        for team_b_goals in range(
            number_of_scores
        ):

            scorelines.append({
                "team_a_goals":
                    team_a_goals,

                "team_b_goals":
                    team_b_goals,

                "probability":
                    float(
                        score_matrix[
                            team_a_goals,
                            team_b_goals
                        ]
                    )
            })

    scorelines = sorted(
        scorelines,
        key=lambda item:
            item["probability"],
        reverse=True
    )

    return scorelines[
        :number_of_results
    ]


# ============================================================
# 20. PENALTY-SHOOTOUT SIMULATION
# ============================================================

def simulate_penalty_shootout(
    team_a,
    team_b,
    team_a_win_probability,
    team_b_win_probability,
    random_generator
):
    """
    Select the shootout winner from relative non-draw strength.
    """

    non_draw_total = (
        team_a_win_probability
        +
        team_b_win_probability
    )

    if non_draw_total <= 0:

        team_a_penalty_probability = 0.5

    else:

        team_a_penalty_probability = (
            team_a_win_probability
            /
            non_draw_total
        )

    if (
        random_generator.random()
        <
        team_a_penalty_probability
    ):
        return team_a

    return team_b


# ============================================================
# 21. COMPLETE VENUE-AWARE MATCH SIMULATION
# ============================================================

def simulate_match(
    team_a,
    team_b,
    venue_country=None,
    knockout=False,
    random_seed=None,
    show_details=True
):
    """
    Simulate one World Cup match.

    Parameters
    ----------
    team_a:
        First team in the official fixture.

    team_b:
        Second team in the official fixture.

    venue_country:
        "Canada", "Mexico", "United States", or an alias such
        as "USA".

        A host advantage is applied only when one of the three
        host teams plays in its own country.

    knockout:
        False: a draw is allowed.
        True: a tied match receives a shootout winner.

    random_seed:
        None gives a different result on each run.
        An integer gives a repeatable result.
    """

    validate_teams(
        team_a,
        team_b
    )

    normalized_venue_country = (
        normalize_country_name(
            venue_country
        )
    )

    venue_type = determine_venue_type(
        team_a=team_a,
        team_b=team_b,
        venue_country=normalized_venue_country
    )

    venue_description = (
        get_venue_description(
            team_a=team_a,
            team_b=team_b,
            venue_country=normalized_venue_country,
            venue_type=venue_type
        )
    )

    random_generator = (
        np.random.default_rng(
            random_seed
        )
    )

    # --------------------------------------------------------
    # Random Forest probabilities
    # --------------------------------------------------------

    classifier_probabilities = (
        get_venue_aware_classifier_probabilities(
            team_a=team_a,
            team_b=team_b,
            venue_type=venue_type
        )
    )

    # --------------------------------------------------------
    # Poisson expected goals
    # --------------------------------------------------------

    expected_goals = (
        get_venue_aware_expected_goals(
            team_a=team_a,
            team_b=team_b,
            venue_type=venue_type
        )
    )

    team_a_expected_goals = (
        expected_goals[
            "team_a_expected_goals"
        ]
    )

    team_b_expected_goals = (
        expected_goals[
            "team_b_expected_goals"
        ]
    )

    # --------------------------------------------------------
    # Poisson score matrix
    # --------------------------------------------------------

    poisson_score_matrix = (
        create_poisson_score_matrix(
            team_a_expected_goals,
            team_b_expected_goals
        )
    )

    poisson_probabilities = (
        calculate_matrix_outcome_probabilities(
            poisson_score_matrix
        )
    )

    # --------------------------------------------------------
    # Hybrid probabilities
    # --------------------------------------------------------

    hybrid_probabilities = (
        blend_outcome_probabilities(
            classifier_probabilities,
            poisson_probabilities
        )
    )

    hybrid_score_matrix = (
        adjust_score_matrix(
            poisson_score_matrix,
            hybrid_probabilities
        )
    )

    # --------------------------------------------------------
    # Generate one scoreline
    # --------------------------------------------------------

    team_a_goals, team_b_goals = (
        sample_scoreline(
            hybrid_score_matrix,
            random_generator
        )
    )

    winner = None
    decided_by = "Regular time"

    if team_a_goals > team_b_goals:

        winner = team_a

    elif team_b_goals > team_a_goals:

        winner = team_b

    elif knockout:

        winner = simulate_penalty_shootout(
            team_a=team_a,
            team_b=team_b,

            team_a_win_probability=(
                hybrid_probabilities[
                    "team_a_win_probability"
                ]
            ),

            team_b_win_probability=(
                hybrid_probabilities[
                    "team_b_win_probability"
                ]
            ),

            random_generator=random_generator
        )

        decided_by = "Penalty shootout"

    most_likely_scorelines = (
        get_most_likely_scorelines(
            hybrid_score_matrix,
            number_of_results=10
        )
    )

    # --------------------------------------------------------
    # Display report
    # --------------------------------------------------------

    if show_details:

        print("\n" + "=" * 78)
        print(
            "VENUE-AWARE HYBRID WORLD CUP "
            "MATCH SIMULATION"
        )
        print("=" * 78)

        print(
            f"\nMatch: {team_a} vs {team_b}"
        )

        print(
            "Venue country:",
            normalized_venue_country
            if normalized_venue_country
            else "Not specified"
        )

        print(
            "Venue treatment:",
            venue_description
        )

        print("\nTeam Elo ratings:")

        print(
            f"{team_a}:",
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
            f"{team_b}:",
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

        print("\nHybrid weights:")

        print(
            "Random Forest:",
            f"{CLASSIFIER_WEIGHT:.0%}"
        )

        print(
            "Poisson:",
            f"{POISSON_WEIGHT:.0%}"
        )

        print(
            "Weight source:",
            HYBRID_WEIGHTS["source"]
        )

        print("\nExpected goals:")

        print(
            f"{team_a}:",
            round(
                team_a_expected_goals,
                3
            )
        )

        print(
            f"{team_b}:",
            round(
                team_b_expected_goals,
                3
            )
        )

        print("\nRandom Forest probabilities:")

        print(
            f"{team_a} win:",
            f"{classifier_probabilities['team_a_win_probability']:.2%}"
        )

        print(
            "Draw:",
            f"{classifier_probabilities['draw_probability']:.2%}"
        )

        print(
            f"{team_b} win:",
            f"{classifier_probabilities['team_b_win_probability']:.2%}"
        )

        print("\nPoisson probabilities:")

        print(
            f"{team_a} win:",
            f"{poisson_probabilities['team_a_win_probability']:.2%}"
        )

        print(
            "Draw:",
            f"{poisson_probabilities['draw_probability']:.2%}"
        )

        print(
            f"{team_b} win:",
            f"{poisson_probabilities['team_b_win_probability']:.2%}"
        )

        print("\nFinal hybrid probabilities:")

        print(
            f"{team_a} win:",
            f"{hybrid_probabilities['team_a_win_probability']:.2%}"
        )

        print(
            "Draw:",
            f"{hybrid_probabilities['draw_probability']:.2%}"
        )

        print(
            f"{team_b} win:",
            f"{hybrid_probabilities['team_b_win_probability']:.2%}"
        )

        print("\nMost likely scorelines:")

        for position, scoreline in enumerate(
            most_likely_scorelines,
            start=1
        ):

            print(
                f"{position:>2}. "
                f"{team_a} "
                f"{scoreline['team_a_goals']}-"
                f"{scoreline['team_b_goals']} "
                f"{team_b}: "
                f"{scoreline['probability']:.2%}"
            )

        print("\nSimulated result:")

        print(
            f"{team_a} "
            f"{team_a_goals}-"
            f"{team_b_goals} "
            f"{team_b}"
        )

        if knockout:

            print(
                "Winner:",
                winner
            )

            print(
                "Decision:",
                decided_by
            )

        elif winner is None:

            print("Outcome: Draw")

        else:

            print(
                "Winner:",
                winner
            )

    return {
        "team_a":
            team_a,

        "team_b":
            team_b,

        "venue_country":
            normalized_venue_country,

        "venue_type":
            venue_type,

        "venue_description":
            venue_description,

        "team_a_expected_goals":
            team_a_expected_goals,

        "team_b_expected_goals":
            team_b_expected_goals,

        "team_a_win_probability":
            hybrid_probabilities[
                "team_a_win_probability"
            ],

        "draw_probability":
            hybrid_probabilities[
                "draw_probability"
            ],

        "team_b_win_probability":
            hybrid_probabilities[
                "team_b_win_probability"
            ],

        "team_a_goals":
            team_a_goals,

        "team_b_goals":
            team_b_goals,

        "winner":
            winner,

        "decided_by":
            decided_by,

        "score_matrix":
            hybrid_score_matrix
    }


# ============================================================
# 22. TEST MATCH
# ============================================================

if __name__ == "__main__":

    result = simulate_match(
        team_a="Mexico",
        team_b="Argentina",

        # Mexico receives host-country advantage here.
        venue_country="Mexico",

        knockout=False,

        # None = different simulation each run.
        # Integer = repeatable simulation.
        random_seed=None,

        show_details=True
    )
