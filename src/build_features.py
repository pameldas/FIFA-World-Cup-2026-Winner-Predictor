import pandas as pd
from pathlib import Path
from collections import defaultdict, deque


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

DATA_FOLDER = Path(r"D:\wc data")

INPUT_PATH = DATA_FOLDER / "results.csv"
FEATURES_OUTPUT_PATH = DATA_FOLDER / "ml_features.csv"
ELO_OUTPUT_PATH = DATA_FOLDER / "team_latest_elo.csv"

DATA_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DATE CONFIGURATION
# ============================================================

# Matches from 1980–1999 are used only to initialize Elo
# ratings and recent-form history.
WARMUP_START = pd.Timestamp("1980-01-01")

# Matches from 2000 onward are saved in the ML feature dataset.
MODEL_START = pd.Timestamp("2000-01-01")

# Exclude matches after this date to avoid World Cup leakage.
CUTOFF_DATE = pd.Timestamp("2026-06-10")


# ============================================================
# 3. ELO AND FORM CONFIGURATION
# ============================================================

INITIAL_ELO = 1500.0

# Applied only when a match is not played at a neutral venue.
HOME_ELO_ADVANTAGE = 100.0

# Controls how strongly each result changes Elo ratings.
K_FACTOR = 30.0

# Number of previous matches used for recent form.
FORM_WINDOW = 5


# ============================================================
# 4. LOAD THE DATASET
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Dataset was not found at:\n{INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

print("=" * 80)
print("WORLD CUP ML FEATURE ENGINEERING")
print("=" * 80)

print("\nDataset loaded successfully.")
print("Original dataset shape:", df.shape)


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"The following required columns are missing:\n"
        f"{missing_columns}"
    )


# ============================================================
# 6. CLEAN THE DATASET
# ============================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["home_score"] = pd.to_numeric(
    df["home_score"],
    errors="coerce"
)

df["away_score"] = pd.to_numeric(
    df["away_score"],
    errors="coerce"
)

# Remove rows containing unusable match information.
df = df.dropna(
    subset=[
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score"
    ]
)

# Use only matches from 1980 until the prediction cutoff.
df = df[
    (df["date"] >= WARMUP_START) &
    (df["date"] <= CUTOFF_DATE)
].copy()

# Remove duplicate matches and sort chronologically.
df = (
    df
    .drop_duplicates()
    .sort_values(
        by=["date"],
        ascending=True
    )
    .reset_index(drop=True)
)

df["home_score"] = df["home_score"].astype(int)
df["away_score"] = df["away_score"].astype(int)

print("\n" + "=" * 80)
print("CLEANED MATCH DATA")
print("=" * 80)

print("Matches from 1980 to cutoff:", len(df))
print("Starting date:", df["date"].min())
print("Ending date:", df["date"].max())
print("Unique home teams:", df["home_team"].nunique())
print("Unique away teams:", df["away_team"].nunique())


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def convert_to_boolean(value):
    """
    Convert the neutral-venue column safely into True or False.
    """

    if isinstance(value, bool):
        return value

    text_value = str(value).strip().lower()

    return text_value in {
        "true",
        "1",
        "yes",
        "y"
    }


def calculate_form_stats(team, form_history):
    """
    Calculate recent statistics using the team's previous
    five matches.

    The current match is not included. This prevents data leakage.
    """

    previous_matches = list(form_history[team])

    number_of_matches = len(previous_matches)

    if number_of_matches == 0:
        return {
            "matches": 0,
            "win_rate": 0.33,
            "draw_rate": 0.34,
            "loss_rate": 0.33,
            "points_per_game": 1.00,
            "goals_for": 1.20,
            "goals_against": 1.20,
            "goal_difference": 0.00
        }

    total_points = sum(
        match["points"]
        for match in previous_matches
    )

    total_wins = sum(
        match["win"]
        for match in previous_matches
    )

    total_draws = sum(
        match["draw"]
        for match in previous_matches
    )

    total_losses = sum(
        match["loss"]
        for match in previous_matches
    )

    total_goals_for = sum(
        match["goals_for"]
        for match in previous_matches
    )

    total_goals_against = sum(
        match["goals_against"]
        for match in previous_matches
    )

    return {
        "matches": number_of_matches,

        "win_rate":
            total_wins / number_of_matches,

        "draw_rate":
            total_draws / number_of_matches,

        "loss_rate":
            total_losses / number_of_matches,

        "points_per_game":
            total_points / number_of_matches,

        "goals_for":
            total_goals_for / number_of_matches,

        "goals_against":
            total_goals_against / number_of_matches,

        "goal_difference":
            (
                total_goals_for -
                total_goals_against
            ) / number_of_matches
    }


def calculate_expected_home_result(
    home_elo,
    away_elo,
    neutral
):
    """
    Calculate the expected Elo result for the home team.

    Expected value ranges between 0 and 1.
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


def determine_match_result(
    home_score,
    away_score
):
    """
    Return result information for one match.

    Target:
    0 = Away Win
    1 = Draw
    2 = Home Win
    """

    if home_score > away_score:
        return {
            "result": "Home Win",
            "target": 2,
            "actual_home_elo_result": 1.0,
            "home_points": 3,
            "away_points": 0,
            "home_win": 1,
            "home_draw": 0,
            "home_loss": 0,
            "away_win": 0,
            "away_draw": 0,
            "away_loss": 1
        }

    if home_score < away_score:
        return {
            "result": "Away Win",
            "target": 0,
            "actual_home_elo_result": 0.0,
            "home_points": 0,
            "away_points": 3,
            "home_win": 0,
            "home_draw": 0,
            "home_loss": 1,
            "away_win": 1,
            "away_draw": 0,
            "away_loss": 0
        }

    return {
        "result": "Draw",
        "target": 1,
        "actual_home_elo_result": 0.5,
        "home_points": 1,
        "away_points": 1,
        "home_win": 0,
        "home_draw": 1,
        "home_loss": 0,
        "away_win": 0,
        "away_draw": 1,
        "away_loss": 0
    }


# ============================================================
# 8. INITIALIZE TEAM RECORDS
# ============================================================

# Every new team begins with an Elo rating of 1500.
elo_ratings = defaultdict(
    lambda: INITIAL_ELO
)

# Store each team's previous five matches.
form_history = defaultdict(
    lambda: deque(maxlen=FORM_WINDOW)
)

# Count the number of matches processed for each team.
matches_processed = defaultdict(int)

# Store the ML feature rows.
feature_rows = []


# ============================================================
# 9. PROCESS MATCHES IN CHRONOLOGICAL ORDER
# ============================================================

print("\n" + "=" * 80)
print("CALCULATING ELO RATINGS AND RECENT FORM")
print("=" * 80)

for row in df.itertuples(index=False):

    match_date = row.date

    home_team = row.home_team
    away_team = row.away_team

    home_score = int(row.home_score)
    away_score = int(row.away_score)

    neutral = convert_to_boolean(
        row.neutral
    )

    # --------------------------------------------------------
    # Get ratings before the current match
    # --------------------------------------------------------

    home_elo_before = elo_ratings[home_team]
    away_elo_before = elo_ratings[away_team]

    # --------------------------------------------------------
    # Get recent form before the current match
    # --------------------------------------------------------

    home_form = calculate_form_stats(
        home_team,
        form_history
    )

    away_form = calculate_form_stats(
        away_team,
        form_history
    )

    if neutral:
        home_advantage_flag = 0
    else:
        home_advantage_flag = 1

    adjusted_home_elo = (
        home_elo_before +
        home_advantage_flag *
        HOME_ELO_ADVANTAGE
    )

    elo_difference = (
        home_elo_before -
        away_elo_before
    )

    adjusted_elo_difference = (
        adjusted_home_elo -
        away_elo_before
    )

    # --------------------------------------------------------
    # Determine the actual result
    # --------------------------------------------------------

    result_information = determine_match_result(
        home_score,
        away_score
    )

    result_text = result_information["result"]
    target = result_information["target"]

    actual_home_elo_result = (
        result_information[
            "actual_home_elo_result"
        ]
    )

    # --------------------------------------------------------
    # Calculate expected Elo result
    # --------------------------------------------------------

    expected_home_elo_result = (
        calculate_expected_home_result(
            home_elo_before,
            away_elo_before,
            neutral
        )
    )

    expected_away_elo_result = (
        1.0 -
        expected_home_elo_result
    )

    # --------------------------------------------------------
    # Save ML rows only from January 1, 2000 onward
    # --------------------------------------------------------

    if match_date >= MODEL_START:

        feature_rows.append({

            # Basic match information
            "date": match_date,
            "year": match_date.year,
            "month": match_date.month,

            "home_team": home_team,
            "away_team": away_team,

            "home_score": home_score,
            "away_score": away_score,

            "tournament": row.tournament,
            "city": row.city,
            "country": row.country,

            "neutral": neutral,
            "home_advantage":
                home_advantage_flag,

            # Pre-match Elo features
            "home_elo":
                round(home_elo_before, 4),

            "away_elo":
                round(away_elo_before, 4),

            "elo_difference":
                round(elo_difference, 4),

            "adjusted_elo_difference":
                round(
                    adjusted_elo_difference,
                    4
                ),

            "expected_home_elo_result":
                round(
                    expected_home_elo_result,
                    6
                ),

            "expected_away_elo_result":
                round(
                    expected_away_elo_result,
                    6
                ),

            # Number of matches available for form
            "home_previous_matches_5":
                home_form["matches"],

            "away_previous_matches_5":
                away_form["matches"],

            # Win-rate features
            "home_win_rate_5":
                round(
                    home_form["win_rate"],
                    4
                ),

            "away_win_rate_5":
                round(
                    away_form["win_rate"],
                    4
                ),

            "win_rate_difference_5":
                round(
                    home_form["win_rate"] -
                    away_form["win_rate"],
                    4
                ),

            # Draw-rate features
            "home_draw_rate_5":
                round(
                    home_form["draw_rate"],
                    4
                ),

            "away_draw_rate_5":
                round(
                    away_form["draw_rate"],
                    4
                ),

            "draw_rate_difference_5":
                round(
                    home_form["draw_rate"] -
                    away_form["draw_rate"],
                    4
                ),

            # Loss-rate features
            "home_loss_rate_5":
                round(
                    home_form["loss_rate"],
                    4
                ),

            "away_loss_rate_5":
                round(
                    away_form["loss_rate"],
                    4
                ),

            "loss_rate_difference_5":
                round(
                    home_form["loss_rate"] -
                    away_form["loss_rate"],
                    4
                ),

            # Points-per-game features
            "home_points_per_game_5":
                round(
                    home_form[
                        "points_per_game"
                    ],
                    4
                ),

            "away_points_per_game_5":
                round(
                    away_form[
                        "points_per_game"
                    ],
                    4
                ),

            "points_per_game_difference_5":
                round(
                    home_form[
                        "points_per_game"
                    ] -
                    away_form[
                        "points_per_game"
                    ],
                    4
                ),

            # Goals scored features
            "home_goals_for_5":
                round(
                    home_form["goals_for"],
                    4
                ),

            "away_goals_for_5":
                round(
                    away_form["goals_for"],
                    4
                ),

            "goals_for_difference_5":
                round(
                    home_form["goals_for"] -
                    away_form["goals_for"],
                    4
                ),

            # Goals conceded features
            "home_goals_against_5":
                round(
                    home_form[
                        "goals_against"
                    ],
                    4
                ),

            "away_goals_against_5":
                round(
                    away_form[
                        "goals_against"
                    ],
                    4
                ),

            "goals_against_difference_5":
                round(
                    home_form[
                        "goals_against"
                    ] -
                    away_form[
                        "goals_against"
                    ],
                    4
                ),

            # Goal-difference form features
            "home_goal_difference_5":
                round(
                    home_form[
                        "goal_difference"
                    ],
                    4
                ),

            "away_goal_difference_5":
                round(
                    away_form[
                        "goal_difference"
                    ],
                    4
                ),

            "recent_goal_difference_gap_5":
                round(
                    home_form[
                        "goal_difference"
                    ] -
                    away_form[
                        "goal_difference"
                    ],
                    4
                ),

            # Prediction target
            "result": result_text,

            # 0 = Away Win
            # 1 = Draw
            # 2 = Home Win
            "target": target
        })

    # --------------------------------------------------------
    # Update Elo ratings after feature extraction
    # --------------------------------------------------------

    elo_change = K_FACTOR * (
        actual_home_elo_result -
        expected_home_elo_result
    )

    elo_ratings[home_team] = (
        home_elo_before +
        elo_change
    )

    elo_ratings[away_team] = (
        away_elo_before -
        elo_change
    )

    matches_processed[home_team] += 1
    matches_processed[away_team] += 1

    # --------------------------------------------------------
    # Update recent form after feature extraction
    # --------------------------------------------------------

    form_history[home_team].append({
        "points":
            result_information[
                "home_points"
            ],

        "win":
            result_information[
                "home_win"
            ],

        "draw":
            result_information[
                "home_draw"
            ],

        "loss":
            result_information[
                "home_loss"
            ],

        "goals_for":
            home_score,

        "goals_against":
            away_score
    })

    form_history[away_team].append({
        "points":
            result_information[
                "away_points"
            ],

        "win":
            result_information[
                "away_win"
            ],

        "draw":
            result_information[
                "away_draw"
            ],

        "loss":
            result_information[
                "away_loss"
            ],

        "goals_for":
            away_score,

        "goals_against":
            home_score
    })


# ============================================================
# 10. CREATE THE ML FEATURE DATASET
# ============================================================

features_df = pd.DataFrame(
    feature_rows
)

if features_df.empty:
    raise ValueError(
        "No ML feature rows were generated."
    )

features_df = (
    features_df
    .sort_values(
        by=["date"],
        ascending=True
    )
    .reset_index(drop=True)
)

features_df.to_csv(
    FEATURES_OUTPUT_PATH,
    index=False
)


# ============================================================
# 11. CREATE THE LATEST ELO TABLE
# ============================================================

elo_rows = []

for team, rating in elo_ratings.items():

    current_form = calculate_form_stats(
        team,
        form_history
    )

    elo_rows.append({
        "team": team,

        "latest_elo":
            round(rating, 2),

        "matches_processed":
            matches_processed[team],

        "recent_win_rate_5":
            round(
                current_form[
                    "win_rate"
                ],
                4
            ),

        "recent_points_per_game_5":
            round(
                current_form[
                    "points_per_game"
                ],
                4
            ),

        "recent_goals_for_5":
            round(
                current_form[
                    "goals_for"
                ],
                4
            ),

        "recent_goals_against_5":
            round(
                current_form[
                    "goals_against"
                ],
                4
            ),

        "recent_goal_difference_5":
            round(
                current_form[
                    "goal_difference"
                ],
                4
            )
    })

elo_df = pd.DataFrame(
    elo_rows
)

elo_df = (
    elo_df
    .sort_values(
        by="latest_elo",
        ascending=False
    )
    .reset_index(drop=True)
)

elo_df.to_csv(
    ELO_OUTPUT_PATH,
    index=False
)


# ============================================================
# 12. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 80)

print("\nML feature dataset shape:")
print(features_df.shape)

print("\nML dataset starting date:")
print(features_df["date"].min())

print("\nML dataset ending date:")
print(features_df["date"].max())

print("\nNumber of unique teams in ML dataset:")
ml_teams = pd.concat(
    [
        features_df["home_team"],
        features_df["away_team"]
    ],
    ignore_index=True
)

print(ml_teams.nunique())

print("\nMissing values in feature dataset:")
print(features_df.isnull().sum().sum())

print("\nDuplicate rows in feature dataset:")
print(features_df.duplicated().sum())

print("\nTarget meaning:")
print("0 = Away Win")
print("1 = Draw")
print("2 = Home Win")

print("\nTarget distribution:")
print(
    features_df["target"]
    .value_counts()
    .sort_index()
)

print("\nTarget percentages:")
print(
    features_df["target"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print("\nFirst ML feature row:")
print(
    features_df
    .head(1)
    .to_string(index=False)
)

print("\nLast ML feature row:")
print(
    features_df
    .tail(1)
    .to_string(index=False)
)


# ============================================================
# 13. DISPLAY TOP ELO TEAMS
# ============================================================

print("\n" + "=" * 80)
print("TOP 20 ELO-RATED TEAMS")
print("=" * 80)

# Exclude teams with very few matches.
top_teams = elo_df[
    elo_df["matches_processed"] >= 20
].head(20)

print(
    top_teams.to_string(index=False)
)


# ============================================================
# 14. CONFIRM SAVED FILES
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED SUCCESSFULLY")
print("=" * 80)

print("\nML feature dataset:")
print(FEATURES_OUTPUT_PATH)

print("\nLatest Elo and recent-form table:")
print(ELO_OUTPUT_PATH)

print("\nDate structure:")
print("1980–1999: Elo and form warm-up only")
print("2000–2026: ML feature dataset")
print("After 2026-06-10: Excluded")
