r"""
FIFA WORLD CUP 2026 — COMPLETE STANDALONE TOURNAMENT ENGINE
===========================================================

Runs the full tournament in one file:

    Group stage (72 matches)
    Round of 32
    Round of 16
    Quarter-finals
    Semi-finals
    Third-place play-off
    Final

The script does NOT import any of your earlier custom .py files.

Required files in D:\wc data
----------------------------
best_match_prediction_model.joblib
best_goal_prediction_models.joblib
best_hybrid_weights.json
team_latest_elo.csv

Required packages
-----------------
pandas
numpy
matplotlib
scikit-learn
joblib
lxml

Install once from Command Prompt:

    py -m pip install pandas numpy matplotlib scikit-learn joblib lxml

Outputs
-------
D:\wc data\world_cup_2026_complete_results\

Notes
-----
1. The eight best third-placed teams advance.
2. The script tries to download FIFA Annex C's official third-place
   allocation table on first use and caches it locally.
3. If the Annex C table cannot be downloaded, a valid deterministic
   fallback assignment is produced and clearly marked in the output.
4. Fair-play cards are not simulated. Latest Elo is used only as the
   final deterministic group-ranking proxy after all score-based
   tiebreakers have failed.
5. This is one stochastic simulation, not a guaranteed forecast.
"""

from __future__ import annotations

import difflib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS
# =============================================================================

DATA_FOLDER = Path(r"D:\wc data")

CLASSIFIER_MODEL_PATH = (
    DATA_FOLDER / "best_match_prediction_model.joblib"
)

GOAL_MODEL_PATH = (
    DATA_FOLDER / "best_goal_prediction_models.joblib"
)

HYBRID_WEIGHT_PATH = (
    DATA_FOLDER / "best_hybrid_weights.json"
)

TEAM_DATA_PATH = (
    DATA_FOLDER / "team_latest_elo.csv"
)

OUTPUT_FOLDER = (
    DATA_FOLDER / "world_cup_2026_complete_results"
)

GROUP_IMAGE_FOLDER = (
    OUTPUT_FOLDER / "group_tables"
)

ROUND_IMAGE_FOLDER = (
    OUTPUT_FOLDER / "round_images"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

GROUP_IMAGE_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

ROUND_IMAGE_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

ANNEX_C_CACHE_PATH = (
    DATA_FOLDER / "annex_c_third_place_mapping.csv"
)


# =============================================================================
# 2. REPRODUCIBLE SEEDS
# =============================================================================

GROUP_STAGE_SEED = 2026
ROUND_OF_32_SEED = 32026
ROUND_OF_16_SEED = 16026
QUARTER_FINAL_SEED = 826
SEMI_FINAL_SEED = 426
THIRD_PLACE_SEED = 326
FINAL_SEED = 10426


# =============================================================================
# 3. MATCH-MODEL SETTINGS
# =============================================================================

HOME_ELO_ADVANTAGE = 100.0
MAX_GOALS = 10

EXTRA_TIME_MINUTE_FACTOR = 1.0 / 3.0
EXTRA_TIME_SCORING_FACTOR = 0.85

HOST_TEAM_COUNTRY = {
    "Canada": "Canada",
    "Mexico": "Mexico",
    "United States": "United States",
    "USA": "United States",
}

COUNTRY_ALIASES = {
    "canada": "Canada",
    "mexico": "Mexico",
    "méxico": "Mexico",
    "united states": "United States",
    "united states of america": "United States",
    "usa": "United States",
    "u.s.a.": "United States",
    "us": "United States",
    "u.s.": "United States",
}


# =============================================================================
# 4. GROUPS
# =============================================================================

GROUPS = {
    "A": [
        "Mexico",
        "South Africa",
        "Korea Republic",
        "Czechia",
    ],
    "B": [
        "Canada",
        "Bosnia and Herzegovina",
        "Qatar",
        "Switzerland",
    ],
    "C": [
        "Brazil",
        "Morocco",
        "Haiti",
        "Scotland",
    ],
    "D": [
        "United States",
        "Paraguay",
        "Australia",
        "Türkiye",
    ],
    "E": [
        "Germany",
        "Curaçao",
        "Côte d'Ivoire",
        "Ecuador",
    ],
    "F": [
        "Netherlands",
        "Japan",
        "Sweden",
        "Tunisia",
    ],
    "G": [
        "Belgium",
        "Egypt",
        "IR Iran",
        "New Zealand",
    ],
    "H": [
        "Spain",
        "Cabo Verde",
        "Saudi Arabia",
        "Uruguay",
    ],
    "I": [
        "France",
        "Senegal",
        "Iraq",
        "Norway",
    ],
    "J": [
        "Argentina",
        "Algeria",
        "Austria",
        "Jordan",
    ],
    "K": [
        "Portugal",
        "Congo DR",
        "Uzbekistan",
        "Colombia",
    ],
    "L": [
        "England",
        "Croatia",
        "Ghana",
        "Panama",
    ],
}


# =============================================================================
# 5. DISPLAY-NAME TO DATASET-NAME CANDIDATES
# =============================================================================

MODEL_NAME_CANDIDATES = {
    "Mexico": ["Mexico"],
    "South Africa": ["South Africa"],
    "Korea Republic": ["South Korea", "Korea Republic"],
    "Czechia": ["Czech Republic", "Czechia"],

    "Canada": ["Canada"],
    "Bosnia and Herzegovina": [
        "Bosnia-Herzegovina",
        "Bosnia and Herzegovina",
    ],
    "Qatar": ["Qatar"],
    "Switzerland": ["Switzerland"],

    "Brazil": ["Brazil"],
    "Morocco": ["Morocco"],
    "Haiti": ["Haiti"],
    "Scotland": ["Scotland"],

    "United States": ["United States", "USA"],
    "Paraguay": ["Paraguay"],
    "Australia": ["Australia"],
    "Türkiye": ["Turkey", "Türkiye", "Turkiye"],

    "Germany": ["Germany"],
    "Curaçao": ["Curaçao", "Curacao"],
    "Côte d'Ivoire": [
        "Ivory Coast",
        "Côte d'Ivoire",
        "Cote d'Ivoire",
    ],
    "Ecuador": ["Ecuador"],

    "Netherlands": ["Netherlands"],
    "Japan": ["Japan"],
    "Sweden": ["Sweden"],
    "Tunisia": ["Tunisia"],

    "Belgium": ["Belgium"],
    "Egypt": ["Egypt"],
    "IR Iran": ["Iran", "IR Iran"],
    "New Zealand": ["New Zealand"],

    "Spain": ["Spain"],
    "Cabo Verde": ["Cape Verde", "Cabo Verde"],
    "Saudi Arabia": ["Saudi Arabia"],
    "Uruguay": ["Uruguay"],

    "France": ["France"],
    "Senegal": ["Senegal"],
    "Iraq": ["Iraq"],
    "Norway": ["Norway"],

    "Argentina": ["Argentina"],
    "Algeria": ["Algeria"],
    "Austria": ["Austria"],
    "Jordan": ["Jordan"],

    "Portugal": ["Portugal"],
    "Congo DR": [
        "DR Congo",
        "Congo DR",
        "Democratic Republic of the Congo",
    ],
    "Uzbekistan": ["Uzbekistan"],
    "Colombia": ["Colombia"],

    "England": ["England"],
    "Croatia": ["Croatia"],
    "Ghana": ["Ghana"],
    "Panama": ["Panama"],
}


# =============================================================================
# 6. GROUP-STAGE FIXTURES
# =============================================================================

# Each tuple:
# match_id, date, group, team_a, team_b, stadium, venue_country

GROUP_FIXTURES = [
    # Group A
    (1,  "2026-06-11", "A", "Mexico", "South Africa", "Mexico City Stadium", "Mexico"),
    (2,  "2026-06-11", "A", "Korea Republic", "Czechia", "Estadio Guadalajara", "Mexico"),
    (25, "2026-06-18", "A", "Czechia", "South Africa", "Atlanta Stadium", "United States"),
    (28, "2026-06-18", "A", "Mexico", "Korea Republic", "Estadio Guadalajara", "Mexico"),
    (53, "2026-06-24", "A", "Czechia", "Mexico", "Mexico City Stadium", "Mexico"),
    (54, "2026-06-24", "A", "South Africa", "Korea Republic", "Estadio Monterrey", "Mexico"),

    # Group B
    (3,  "2026-06-12", "B", "Canada", "Bosnia and Herzegovina", "Toronto Stadium", "Canada"),
    (5,  "2026-06-13", "B", "Qatar", "Switzerland", "San Francisco Bay Area Stadium", "United States"),
    (26, "2026-06-18", "B", "Switzerland", "Bosnia and Herzegovina", "Los Angeles Stadium", "United States"),
    (27, "2026-06-18", "B", "Canada", "Qatar", "BC Place Vancouver", "Canada"),
    (51, "2026-06-24", "B", "Switzerland", "Canada", "BC Place Vancouver", "Canada"),
    (52, "2026-06-24", "B", "Bosnia and Herzegovina", "Qatar", "Seattle Stadium", "United States"),

    # Group C
    (6,  "2026-06-13", "C", "Brazil", "Morocco", "New York New Jersey Stadium", "United States"),
    (7,  "2026-06-13", "C", "Haiti", "Scotland", "Boston Stadium", "United States"),
    (30, "2026-06-19", "C", "Scotland", "Morocco", "Boston Stadium", "United States"),
    (31, "2026-06-19", "C", "Brazil", "Haiti", "Philadelphia Stadium", "United States"),
    (49, "2026-06-24", "C", "Scotland", "Brazil", "Miami Stadium", "United States"),
    (50, "2026-06-24", "C", "Morocco", "Haiti", "Atlanta Stadium", "United States"),

    # Group D
    (4,  "2026-06-12", "D", "United States", "Paraguay", "Los Angeles Stadium", "United States"),
    (8,  "2026-06-13", "D", "Australia", "Türkiye", "BC Place Vancouver", "Canada"),
    (29, "2026-06-19", "D", "United States", "Australia", "Seattle Stadium", "United States"),
    (32, "2026-06-19", "D", "Türkiye", "Paraguay", "San Francisco Bay Area Stadium", "United States"),
    (59, "2026-06-25", "D", "Türkiye", "United States", "Los Angeles Stadium", "United States"),
    (60, "2026-06-25", "D", "Paraguay", "Australia", "San Francisco Bay Area Stadium", "United States"),

    # Group E
    (9,  "2026-06-14", "E", "Germany", "Curaçao", "Houston Stadium", "United States"),
    (11, "2026-06-14", "E", "Côte d'Ivoire", "Ecuador", "Philadelphia Stadium", "United States"),
    (34, "2026-06-20", "E", "Germany", "Côte d'Ivoire", "Toronto Stadium", "Canada"),
    (35, "2026-06-20", "E", "Ecuador", "Curaçao", "Kansas City Stadium", "United States"),
    (55, "2026-06-25", "E", "Curaçao", "Côte d'Ivoire", "Philadelphia Stadium", "United States"),
    (56, "2026-06-25", "E", "Ecuador", "Germany", "New York New Jersey Stadium", "United States"),

    # Group F
    (10, "2026-06-14", "F", "Netherlands", "Japan", "Dallas Stadium", "United States"),
    (12, "2026-06-14", "F", "Sweden", "Tunisia", "Estadio Monterrey", "Mexico"),
    (33, "2026-06-20", "F", "Netherlands", "Sweden", "Houston Stadium", "United States"),
    (36, "2026-06-20", "F", "Tunisia", "Japan", "Estadio Monterrey", "Mexico"),
    (57, "2026-06-25", "F", "Japan", "Sweden", "Dallas Stadium", "United States"),
    (58, "2026-06-25", "F", "Tunisia", "Netherlands", "Kansas City Stadium", "United States"),

    # Group G
    (14, "2026-06-15", "G", "Belgium", "Egypt", "Seattle Stadium", "United States"),
    (16, "2026-06-15", "G", "IR Iran", "New Zealand", "Los Angeles Stadium", "United States"),
    (38, "2026-06-21", "G", "Belgium", "IR Iran", "Los Angeles Stadium", "United States"),
    (40, "2026-06-21", "G", "New Zealand", "Egypt", "BC Place Vancouver", "Canada"),
    (61, "2026-06-26", "G", "Belgium", "New Zealand", "Seattle Stadium", "United States"),
    (62, "2026-06-26", "G", "IR Iran", "Egypt", "BC Place Vancouver", "Canada"),

    # Group H
    (13, "2026-06-15", "H", "Spain", "Cabo Verde", "Atlanta Stadium", "United States"),
    (15, "2026-06-15", "H", "Saudi Arabia", "Uruguay", "Miami Stadium", "United States"),
    (37, "2026-06-21", "H", "Spain", "Saudi Arabia", "Atlanta Stadium", "United States"),
    (39, "2026-06-21", "H", "Uruguay", "Cabo Verde", "Miami Stadium", "United States"),
    (63, "2026-06-26", "H", "Spain", "Uruguay", "Mexico City Stadium", "Mexico"),
    (64, "2026-06-26", "H", "Cabo Verde", "Saudi Arabia", "Estadio Guadalajara", "Mexico"),

    # Group I
    (17, "2026-06-16", "I", "France", "Senegal", "New York New Jersey Stadium", "United States"),
    (18, "2026-06-16", "I", "Iraq", "Norway", "Boston Stadium", "United States"),
    (41, "2026-06-22", "I", "Norway", "Senegal", "New York New Jersey Stadium", "United States"),
    (42, "2026-06-22", "I", "France", "Iraq", "Philadelphia Stadium", "United States"),
    (65, "2026-06-26", "I", "France", "Norway", "Boston Stadium", "United States"),
    (66, "2026-06-26", "I", "Senegal", "Iraq", "Miami Stadium", "United States"),

    # Group J
    (19, "2026-06-16", "J", "Argentina", "Algeria", "Kansas City Stadium", "United States"),
    (20, "2026-06-16", "J", "Austria", "Jordan", "San Francisco Bay Area Stadium", "United States"),
    (43, "2026-06-22", "J", "Argentina", "Austria", "Dallas Stadium", "United States"),
    (44, "2026-06-22", "J", "Jordan", "Algeria", "San Francisco Bay Area Stadium", "United States"),
    (69, "2026-06-27", "J", "Jordan", "Argentina", "Dallas Stadium", "United States"),
    (70, "2026-06-27", "J", "Algeria", "Austria", "Kansas City Stadium", "United States"),

    # Group K
    (21, "2026-06-17", "K", "Portugal", "Congo DR", "Houston Stadium", "United States"),
    (24, "2026-06-17", "K", "Uzbekistan", "Colombia", "Mexico City Stadium", "Mexico"),
    (47, "2026-06-23", "K", "Portugal", "Uzbekistan", "Houston Stadium", "United States"),
    (48, "2026-06-23", "K", "Colombia", "Congo DR", "Estadio Guadalajara", "Mexico"),
    (71, "2026-06-27", "K", "Colombia", "Portugal", "Miami Stadium", "United States"),
    (72, "2026-06-27", "K", "Congo DR", "Uzbekistan", "Atlanta Stadium", "United States"),

    # Group L
    (22, "2026-06-17", "L", "England", "Croatia", "Dallas Stadium", "United States"),
    (23, "2026-06-17", "L", "Ghana", "Panama", "Toronto Stadium", "Canada"),
    (45, "2026-06-23", "L", "England", "Ghana", "Boston Stadium", "United States"),
    (46, "2026-06-23", "L", "Panama", "Croatia", "Toronto Stadium", "Canada"),
    (67, "2026-06-27", "L", "Panama", "England", "New York New Jersey Stadium", "United States"),
    (68, "2026-06-27", "L", "Croatia", "Ghana", "Philadelphia Stadium", "United States"),
]


# =============================================================================
# 7. OFFICIAL KNOCKOUT PATH
# =============================================================================

ROUND_OF_32_MATCHES = [
    (73, "2026-06-28", "2A", "2B", "Los Angeles Stadium", "United States"),
    (74, "2026-06-29", "1E", "THIRD_1E", "Boston Stadium", "United States"),
    (75, "2026-06-29", "1F", "2C", "Estadio Monterrey", "Mexico"),
    (76, "2026-06-29", "1C", "2F", "Houston Stadium", "United States"),
    (77, "2026-06-30", "1I", "THIRD_1I", "New York New Jersey Stadium", "United States"),
    (78, "2026-06-30", "2E", "2I", "Dallas Stadium", "United States"),
    (79, "2026-06-30", "1A", "THIRD_1A", "Mexico City Stadium", "Mexico"),
    (80, "2026-07-01", "1L", "THIRD_1L", "Atlanta Stadium", "United States"),
    (81, "2026-07-01", "1D", "THIRD_1D", "San Francisco Bay Area Stadium", "United States"),
    (82, "2026-07-01", "1G", "THIRD_1G", "Seattle Stadium", "United States"),
    (83, "2026-07-02", "2K", "2L", "Toronto Stadium", "Canada"),
    (84, "2026-07-02", "1H", "2J", "Los Angeles Stadium", "United States"),
    (85, "2026-07-02", "1B", "THIRD_1B", "BC Place Vancouver", "Canada"),
    (86, "2026-07-03", "1J", "2H", "Miami Stadium", "United States"),
    (87, "2026-07-03", "1K", "THIRD_1K", "Kansas City Stadium", "United States"),
    (88, "2026-07-03", "2D", "2G", "Dallas Stadium", "United States"),
]

ROUND_OF_16_MATCHES = [
    (89, "2026-07-04", "W74", "W77", "Philadelphia Stadium", "United States"),
    (90, "2026-07-04", "W73", "W75", "Houston Stadium", "United States"),
    (91, "2026-07-05", "W76", "W78", "New York New Jersey Stadium", "United States"),
    (92, "2026-07-05", "W79", "W80", "Mexico City Stadium", "Mexico"),
    (93, "2026-07-06", "W83", "W84", "Dallas Stadium", "United States"),
    (94, "2026-07-06", "W81", "W82", "Seattle Stadium", "United States"),
    (95, "2026-07-07", "W86", "W88", "Atlanta Stadium", "United States"),
    (96, "2026-07-07", "W85", "W87", "BC Place Vancouver", "Canada"),
]

QUARTER_FINAL_MATCHES = [
    (97, "2026-07-09", "W89", "W90", "Boston Stadium", "United States"),
    (98, "2026-07-10", "W93", "W94", "Los Angeles Stadium", "United States"),
    (99, "2026-07-11", "W91", "W92", "Miami Stadium", "United States"),
    (100, "2026-07-11", "W95", "W96", "Kansas City Stadium", "United States"),
]

SEMI_FINAL_MATCHES = [
    (101, "2026-07-14", "W97", "W98", "Dallas Stadium", "United States"),
    (102, "2026-07-15", "W99", "W100", "Atlanta Stadium", "United States"),
]

THIRD_PLACE_MATCHES = [
    (103, "2026-07-18", "L101", "L102", "Miami Stadium", "United States"),
]

FINAL_MATCHES = [
    (104, "2026-07-19", "W101", "W102", "New York New Jersey Stadium", "United States"),
]


# =============================================================================
# 8. THIRD-PLACE ASSIGNMENT SETTINGS
# =============================================================================

THIRD_PLACE_WINNER_COLUMNS = [
    "1A",
    "1B",
    "1D",
    "1E",
    "1G",
    "1I",
    "1K",
    "1L",
]

ALLOWED_THIRD_GROUPS = {
    "1A": set("CEFHI"),
    "1B": set("EFGIJ"),
    "1D": set("BEFIJ"),
    "1E": set("ABCDF"),
    "1G": set("AEHIJ"),
    "1I": set("CDFGH"),
    "1K": set("DEIJL"),
    "1L": set("EHIJK"),
}

ANNEX_C_TABLE_URL = (
    "https://en.wikipedia.org/wiki/"
    "2026_FIFA_World_Cup_knockout_stage"
)


# =============================================================================
# 9. LOAD MODEL FILES
# =============================================================================

def check_required_files() -> None:
    required_paths = [
        CLASSIFIER_MODEL_PATH,
        GOAL_MODEL_PATH,
        HYBRID_WEIGHT_PATH,
        TEAM_DATA_PATH,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        missing_text = "\n".join(
            str(path)
            for path in missing_paths
        )

        raise FileNotFoundError(
            "Required files are missing:\n"
            f"{missing_text}"
        )


check_required_files()

classifier_bundle = joblib.load(
    CLASSIFIER_MODEL_PATH
)

goal_bundle = joblib.load(
    GOAL_MODEL_PATH
)

with open(
    HYBRID_WEIGHT_PATH,
    "r",
    encoding="utf-8",
) as weight_file:
    weight_information = json.load(
        weight_file
    )

CLASSIFIER_WEIGHT = float(
    weight_information.get(
        "classifier_weight",
        0.0,
    )
)

POISSON_WEIGHT = float(
    weight_information.get(
        "poisson_weight",
        1.0,
    )
)

weight_total = (
    CLASSIFIER_WEIGHT
    +
    POISSON_WEIGHT
)

if weight_total <= 0:
    raise ValueError(
        "Hybrid model weights must have a positive total."
    )

CLASSIFIER_WEIGHT /= weight_total
POISSON_WEIGHT /= weight_total

classifier_model = classifier_bundle[
    "model"
]

classifier_feature_columns = classifier_bundle[
    "feature_columns"
]

home_goal_model = goal_bundle[
    "home_goal_model"
]

away_goal_model = goal_bundle[
    "away_goal_model"
]

goal_feature_columns = goal_bundle[
    "feature_columns"
]

MIN_EXPECTED_GOALS = float(
    goal_bundle.get(
        "minimum_expected_goals",
        0.05,
    )
)

MAX_EXPECTED_GOALS = float(
    goal_bundle.get(
        "maximum_expected_goals",
        6.0,
    )
)

team_data = pd.read_csv(
    TEAM_DATA_PATH
)

required_team_columns = {
    "team",
    "latest_elo",
    "recent_win_rate_5",
    "recent_points_per_game_5",
    "recent_goals_for_5",
    "recent_goals_against_5",
    "recent_goal_difference_5",
}

if not required_team_columns.issubset(
    team_data.columns
):
    missing_columns = sorted(
        required_team_columns
        -
        set(team_data.columns)
    )

    raise ValueError(
        "team_latest_elo.csv is missing columns:\n"
        f"{missing_columns}"
    )

team_data["team"] = (
    team_data["team"]
    .astype(str)
    .str.strip()
)

team_data = team_data.set_index(
    "team"
)


# =============================================================================
# 10. RESOLVE DISPLAY NAMES
# =============================================================================

def resolve_model_team_name(
    display_name: str
) -> str:
    available_names = set(
        team_data.index.astype(str)
    )

    candidates = MODEL_NAME_CANDIDATES.get(
        display_name,
        [display_name],
    )

    for candidate in candidates:
        if candidate in available_names:
            return candidate

    close_matches = difflib.get_close_matches(
        display_name,
        sorted(available_names),
        n=8,
        cutoff=0.45,
    )

    raise ValueError(
        f"\nCould not find dataset name for: {display_name}\n"
        f"Tried: {candidates}\n"
        f"Closest names: {close_matches}\n"
        "Update MODEL_NAME_CANDIDATES near the top of the file."
    )


DISPLAY_TO_MODEL = {}

for group_teams in GROUPS.values():
    for display_team in group_teams:
        DISPLAY_TO_MODEL[
            display_team
        ] = resolve_model_team_name(
            display_team
        )

MODEL_TO_DISPLAY = {
    model_name: display_name
    for display_name, model_name
    in DISPLAY_TO_MODEL.items()
}


# =============================================================================
# 11. MATCH-FEATURE ENGINE
# =============================================================================

def normalize_country_name(
    venue_country: str | None
) -> str | None:
    if venue_country is None:
        return None

    original_text = str(
        venue_country
    ).strip()

    return COUNTRY_ALIASES.get(
        original_text.lower(),
        original_text,
    )


def calculate_expected_home_result(
    home_elo: float,
    away_elo: float,
    neutral: bool,
) -> float:
    home_advantage = (
        0.0
        if neutral
        else HOME_ELO_ADVANTAGE
    )

    adjusted_home_elo = (
        home_elo
        +
        home_advantage
    )

    return 1.0 / (
        1.0
        +
        10.0 ** (
            (
                away_elo
                -
                adjusted_home_elo
            )
            /
            400.0
        )
    )


def build_match_features(
    home_team: str,
    away_team: str,
    neutral: bool,
) -> pd.DataFrame:
    if home_team == away_team:
        raise ValueError(
            "A team cannot play against itself."
        )

    if home_team not in team_data.index:
        raise ValueError(
            f"Team not found in Elo table: {home_team}"
        )

    if away_team not in team_data.index:
        raise ValueError(
            f"Team not found in Elo table: {away_team}"
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

    home_advantage_flag = (
        0
        if neutral
        else 1
    )

    expected_home = calculate_expected_home_result(
        home_elo=home_elo,
        away_elo=away_elo,
        neutral=neutral,
    )

    home_win_rate = float(
        home["recent_win_rate_5"]
    )

    away_win_rate = float(
        away["recent_win_rate_5"]
    )

    home_points = float(
        home["recent_points_per_game_5"]
    )

    away_points = float(
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
        "neutral": int(neutral),
        "home_advantage": home_advantage_flag,

        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_difference": (
            home_elo
            -
            away_elo
        ),
        "adjusted_elo_difference": (
            home_elo
            +
            home_advantage_flag
            *
            HOME_ELO_ADVANTAGE
            -
            away_elo
        ),
        "expected_home_elo_result": expected_home,
        "expected_away_elo_result": (
            1.0
            -
            expected_home
        ),

        "home_win_rate_5": home_win_rate,
        "away_win_rate_5": away_win_rate,
        "win_rate_difference_5": (
            home_win_rate
            -
            away_win_rate
        ),

        "home_points_per_game_5": home_points,
        "away_points_per_game_5": away_points,
        "points_per_game_difference_5": (
            home_points
            -
            away_points
        ),

        "home_goals_for_5": home_goals_for,
        "away_goals_for_5": away_goals_for,
        "goals_for_difference_5": (
            home_goals_for
            -
            away_goals_for
        ),

        "home_goals_against_5": home_goals_against,
        "away_goals_against_5": away_goals_against,
        "goals_against_difference_5": (
            home_goals_against
            -
            away_goals_against
        ),

        "home_goal_difference_5": home_goal_difference,
        "away_goal_difference_5": away_goal_difference,
        "recent_goal_difference_gap_5": (
            home_goal_difference
            -
            away_goal_difference
        ),
    }

    all_required_columns = list(
        dict.fromkeys(
            list(
                classifier_feature_columns
            )
            +
            list(
                goal_feature_columns
            )
        )
    )

    missing_features = [
        column
        for column in all_required_columns
        if column not in feature_row
    ]

    if missing_features:
        raise ValueError(
            "This script cannot create the following saved-model "
            f"features:\n{missing_features}"
        )

    return pd.DataFrame(
        [feature_row]
    )


def get_classifier_classes():
    if hasattr(
        classifier_model,
        "classes_",
    ):
        return classifier_model.classes_

    if (
        hasattr(
            classifier_model,
            "named_steps",
        )
        and
        "model"
        in classifier_model.named_steps
        and
        hasattr(
            classifier_model.named_steps["model"],
            "classes_",
        )
    ):
        return (
            classifier_model
            .named_steps["model"]
            .classes_
        )

    raise AttributeError(
        "Classifier class labels could not be found."
    )


def get_raw_classifier_probabilities(
    home_team: str,
    away_team: str,
    neutral: bool,
) -> dict:
    features = build_match_features(
        home_team=home_team,
        away_team=away_team,
        neutral=neutral,
    )

    model_input = features[
        classifier_feature_columns
    ]

    probabilities = (
        classifier_model
        .predict_proba(
            model_input
        )[0]
    )

    class_values = get_classifier_classes()

    probability_by_class = {
        int(class_value): float(probability)
        for class_value, probability
        in zip(
            class_values,
            probabilities,
        )
    }

    return {
        "away_win": probability_by_class[0],
        "draw": probability_by_class[1],
        "home_win": probability_by_class[2],
    }


def determine_venue_type(
    team_a: str,
    team_b: str,
    venue_country: str | None,
) -> str:
    normalized_country = normalize_country_name(
        venue_country
    )

    team_a_home_country = (
        HOST_TEAM_COUNTRY.get(
            team_a
        )
    )

    team_b_home_country = (
        HOST_TEAM_COUNTRY.get(
            team_b
        )
    )

    if (
        team_a_home_country
        ==
        normalized_country
        and
        team_b_home_country
        !=
        normalized_country
    ):
        return "team_a_home"

    if (
        team_b_home_country
        ==
        normalized_country
        and
        team_a_home_country
        !=
        normalized_country
    ):
        return "team_b_home"

    return "neutral"


def get_venue_aware_classifier_probabilities(
    team_a: str,
    team_b: str,
    venue_type: str,
) -> dict:
    if venue_type == "neutral":
        prediction_ab = get_raw_classifier_probabilities(
            home_team=team_a,
            away_team=team_b,
            neutral=True,
        )

        prediction_ba = get_raw_classifier_probabilities(
            home_team=team_b,
            away_team=team_a,
            neutral=True,
        )

        team_a_win = (
            prediction_ab["home_win"]
            +
            prediction_ba["away_win"]
        ) / 2.0

        draw = (
            prediction_ab["draw"]
            +
            prediction_ba["draw"]
        ) / 2.0

        team_b_win = (
            prediction_ab["away_win"]
            +
            prediction_ba["home_win"]
        ) / 2.0

    elif venue_type == "team_a_home":
        prediction = get_raw_classifier_probabilities(
            home_team=team_a,
            away_team=team_b,
            neutral=False,
        )

        team_a_win = prediction["home_win"]
        draw = prediction["draw"]
        team_b_win = prediction["away_win"]

    elif venue_type == "team_b_home":
        prediction = get_raw_classifier_probabilities(
            home_team=team_b,
            away_team=team_a,
            neutral=False,
        )

        team_a_win = prediction["away_win"]
        draw = prediction["draw"]
        team_b_win = prediction["home_win"]

    else:
        raise ValueError(
            f"Unknown venue type: {venue_type}"
        )

    total = (
        team_a_win
        +
        draw
        +
        team_b_win
    )

    return {
        "team_a_win_probability": team_a_win / total,
        "draw_probability": draw / total,
        "team_b_win_probability": team_b_win / total,
    }


def clip_expected_goals(
    value: float
) -> float:
    return float(
        np.clip(
            value,
            MIN_EXPECTED_GOALS,
            MAX_EXPECTED_GOALS,
        )
    )


def get_venue_aware_expected_goals(
    team_a: str,
    team_b: str,
    venue_type: str,
) -> dict:
    if venue_type == "neutral":
        features_ab = build_match_features(
            home_team=team_a,
            away_team=team_b,
            neutral=True,
        )

        features_ba = build_match_features(
            home_team=team_b,
            away_team=team_a,
            neutral=True,
        )

        ab_home = clip_expected_goals(
            home_goal_model.predict(
                features_ab[
                    goal_feature_columns
                ]
            )[0]
        )

        ab_away = clip_expected_goals(
            away_goal_model.predict(
                features_ab[
                    goal_feature_columns
                ]
            )[0]
        )

        ba_home = clip_expected_goals(
            home_goal_model.predict(
                features_ba[
                    goal_feature_columns
                ]
            )[0]
        )

        ba_away = clip_expected_goals(
            away_goal_model.predict(
                features_ba[
                    goal_feature_columns
                ]
            )[0]
        )

        team_a_expected = (
            ab_home
            +
            ba_away
        ) / 2.0

        team_b_expected = (
            ab_away
            +
            ba_home
        ) / 2.0

    elif venue_type == "team_a_home":
        features = build_match_features(
            home_team=team_a,
            away_team=team_b,
            neutral=False,
        )

        team_a_expected = clip_expected_goals(
            home_goal_model.predict(
                features[
                    goal_feature_columns
                ]
            )[0]
        )

        team_b_expected = clip_expected_goals(
            away_goal_model.predict(
                features[
                    goal_feature_columns
                ]
            )[0]
        )

    elif venue_type == "team_b_home":
        features = build_match_features(
            home_team=team_b,
            away_team=team_a,
            neutral=False,
        )

        internal_home = clip_expected_goals(
            home_goal_model.predict(
                features[
                    goal_feature_columns
                ]
            )[0]
        )

        internal_away = clip_expected_goals(
            away_goal_model.predict(
                features[
                    goal_feature_columns
                ]
            )[0]
        )

        team_a_expected = internal_away
        team_b_expected = internal_home

    else:
        raise ValueError(
            f"Unknown venue type: {venue_type}"
        )

    return {
        "team_a_expected_goals": float(
            team_a_expected
        ),
        "team_b_expected_goals": float(
            team_b_expected
        ),
    }


# =============================================================================
# 12. POISSON SCORE ENGINE
# =============================================================================

def poisson_probability(
    goals: int,
    expected_goals: float,
) -> float:
    log_probability = (
        -expected_goals
        +
        goals
        *
        math.log(
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


def create_poisson_score_matrix(
    team_a_expected_goals: float,
    team_b_expected_goals: float,
) -> np.ndarray:
    team_a_probabilities = np.array([
        poisson_probability(
            goals,
            team_a_expected_goals,
        )
        for goals in range(
            MAX_GOALS + 1
        )
    ])

    team_b_probabilities = np.array([
        poisson_probability(
            goals,
            team_b_expected_goals,
        )
        for goals in range(
            MAX_GOALS + 1
        )
    ])

    score_matrix = np.outer(
        team_a_probabilities,
        team_b_probabilities,
    )

    return (
        score_matrix
        /
        score_matrix.sum()
    )


def calculate_matrix_outcome_probabilities(
    score_matrix: np.ndarray
) -> dict:
    team_a_win = float(
        np.tril(
            score_matrix,
            k=-1,
        ).sum()
    )

    draw = float(
        np.trace(
            score_matrix
        )
    )

    team_b_win = float(
        np.triu(
            score_matrix,
            k=1,
        ).sum()
    )

    return {
        "team_a_win_probability": team_a_win,
        "draw_probability": draw,
        "team_b_win_probability": team_b_win,
    }


def blend_outcome_probabilities(
    classifier_probabilities: dict,
    poisson_probabilities: dict,
) -> dict:
    team_a_win = (
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

    draw = (
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

    team_b_win = (
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

    total = (
        team_a_win
        +
        draw
        +
        team_b_win
    )

    return {
        "team_a_win_probability": team_a_win / total,
        "draw_probability": draw / total,
        "team_b_win_probability": team_b_win / total,
    }


def adjust_score_matrix(
    score_matrix: np.ndarray,
    target_probabilities: dict,
) -> np.ndarray:
    original_probabilities = (
        calculate_matrix_outcome_probabilities(
            score_matrix
        )
    )

    epsilon = 1e-12

    scale_a = (
        target_probabilities[
            "team_a_win_probability"
        ]
        /
        max(
            original_probabilities[
                "team_a_win_probability"
            ],
            epsilon,
        )
    )

    scale_draw = (
        target_probabilities[
            "draw_probability"
        ]
        /
        max(
            original_probabilities[
                "draw_probability"
            ],
            epsilon,
        )
    )

    scale_b = (
        target_probabilities[
            "team_b_win_probability"
        ]
        /
        max(
            original_probabilities[
                "team_b_win_probability"
            ],
            epsilon,
        )
    )

    adjusted = score_matrix.copy()

    for goals_a in range(
        adjusted.shape[0]
    ):
        for goals_b in range(
            adjusted.shape[1]
        ):
            if goals_a > goals_b:
                adjusted[
                    goals_a,
                    goals_b,
                ] *= scale_a

            elif goals_a < goals_b:
                adjusted[
                    goals_a,
                    goals_b,
                ] *= scale_b

            else:
                adjusted[
                    goals_a,
                    goals_b,
                ] *= scale_draw

    return (
        adjusted
        /
        adjusted.sum()
    )


def sample_scoreline(
    score_matrix: np.ndarray,
    random_generator,
) -> tuple[int, int]:
    flattened = score_matrix.flatten()

    selected_position = int(
        random_generator.choice(
            len(flattened),
            p=flattened,
        )
    )

    goals_a, goals_b = np.unravel_index(
        selected_position,
        score_matrix.shape,
    )

    return (
        int(goals_a),
        int(goals_b),
    )


def simulate_match(
    team_a: str,
    team_b: str,
    venue_country: str | None,
    random_seed: int,
) -> dict:
    venue_type = determine_venue_type(
        team_a=team_a,
        team_b=team_b,
        venue_country=venue_country,
    )

    classifier_probabilities = (
        get_venue_aware_classifier_probabilities(
            team_a=team_a,
            team_b=team_b,
            venue_type=venue_type,
        )
    )

    expected_goals = (
        get_venue_aware_expected_goals(
            team_a=team_a,
            team_b=team_b,
            venue_type=venue_type,
        )
    )

    poisson_matrix = create_poisson_score_matrix(
        team_a_expected_goals=expected_goals[
            "team_a_expected_goals"
        ],
        team_b_expected_goals=expected_goals[
            "team_b_expected_goals"
        ],
    )

    poisson_probabilities = (
        calculate_matrix_outcome_probabilities(
            poisson_matrix
        )
    )

    final_probabilities = blend_outcome_probabilities(
        classifier_probabilities=
            classifier_probabilities,
        poisson_probabilities=
            poisson_probabilities,
    )

    final_score_matrix = adjust_score_matrix(
        score_matrix=poisson_matrix,
        target_probabilities=
            final_probabilities,
    )

    random_generator = np.random.default_rng(
        random_seed
    )

    goals_a, goals_b = sample_scoreline(
        score_matrix=final_score_matrix,
        random_generator=random_generator,
    )

    return {
        "team_a": team_a,
        "team_b": team_b,
        "venue_country": normalize_country_name(
            venue_country
        ),
        "venue_type": venue_type,

        "team_a_expected_goals":
            expected_goals[
                "team_a_expected_goals"
            ],

        "team_b_expected_goals":
            expected_goals[
                "team_b_expected_goals"
            ],

        "team_a_win_probability":
            final_probabilities[
                "team_a_win_probability"
            ],

        "draw_probability":
            final_probabilities[
                "draw_probability"
            ],

        "team_b_win_probability":
            final_probabilities[
                "team_b_win_probability"
            ],

        "team_a_goals": goals_a,
        "team_b_goals": goals_b,
        "score_matrix": final_score_matrix,
    }


# =============================================================================
# 13. CONFIGURATION VALIDATION
# =============================================================================

def validate_tournament_configuration() -> None:
    if len(GROUPS) != 12:
        raise ValueError(
            f"Expected 12 groups, found {len(GROUPS)}."
        )

    all_group_teams = [
        team
        for teams in GROUPS.values()
        for team in teams
    ]

    if len(all_group_teams) != 48:
        raise ValueError(
            "Expected 48 group-team slots."
        )

    if len(set(all_group_teams)) != 48:
        raise ValueError(
            "Duplicate team found in the group list."
        )

    if len(GROUP_FIXTURES) != 72:
        raise ValueError(
            f"Expected 72 group fixtures, found "
            f"{len(GROUP_FIXTURES)}."
        )

    for group_name, group_teams in GROUPS.items():
        fixtures = [
            fixture
            for fixture in GROUP_FIXTURES
            if fixture[2] == group_name
        ]

        if len(fixtures) != 6:
            raise ValueError(
                f"Group {group_name} does not have six matches."
            )

        pairings = {
            tuple(
                sorted(
                    (
                        fixture[3],
                        fixture[4],
                    )
                )
            )
            for fixture in fixtures
        }

        if len(pairings) != 6:
            raise ValueError(
                f"Group {group_name} does not have six unique pairings."
            )

        for fixture in fixtures:
            if (
                fixture[3]
                not in group_teams
                or
                fixture[4]
                not in group_teams
            ):
                raise ValueError(
                    f"Invalid Group {group_name} fixture: {fixture}"
                )


# =============================================================================
# 14. GROUP-TABLE FUNCTIONS
# =============================================================================

def create_empty_group_table(
    group_name: str
) -> dict:
    table = {}

    for display_team in GROUPS[
        group_name
    ]:
        model_team = DISPLAY_TO_MODEL[
            display_team
        ]

        table[display_team] = {
            "Team": display_team,
            "ModelTeam": model_team,
            "P": 0,
            "W": 0,
            "D": 0,
            "L": 0,
            "GF": 0,
            "GA": 0,
            "GD": 0,
            "Pts": 0,
            "Elo": float(
                team_data.loc[
                    model_team,
                    "latest_elo",
                ]
            ),
        }

    return table


def update_group_table(
    table: dict,
    team_a: str,
    team_b: str,
    goals_a: int,
    goals_b: int,
) -> None:
    table[team_a]["P"] += 1
    table[team_b]["P"] += 1

    table[team_a]["GF"] += goals_a
    table[team_a]["GA"] += goals_b

    table[team_b]["GF"] += goals_b
    table[team_b]["GA"] += goals_a

    if goals_a > goals_b:
        table[team_a]["W"] += 1
        table[team_b]["L"] += 1
        table[team_a]["Pts"] += 3

    elif goals_b > goals_a:
        table[team_b]["W"] += 1
        table[team_a]["L"] += 1
        table[team_b]["Pts"] += 3

    else:
        table[team_a]["D"] += 1
        table[team_b]["D"] += 1
        table[team_a]["Pts"] += 1
        table[team_b]["Pts"] += 1

    for team in (
        team_a,
        team_b,
    ):
        table[team]["GD"] = (
            table[team]["GF"]
            -
            table[team]["GA"]
        )


def calculate_head_to_head_stats(
    tied_teams: list[str],
    group_matches: list[dict],
) -> dict:
    tied_set = set(
        tied_teams
    )

    mini_table = {
        team: {
            "Pts": 0,
            "GF": 0,
            "GA": 0,
            "GD": 0,
        }
        for team in tied_teams
    }

    for match in group_matches:
        team_a = match["team_a"]
        team_b = match["team_b"]

        if (
            team_a not in tied_set
            or
            team_b not in tied_set
        ):
            continue

        goals_a = int(
            match["team_a_goals"]
        )

        goals_b = int(
            match["team_b_goals"]
        )

        mini_table[team_a]["GF"] += goals_a
        mini_table[team_a]["GA"] += goals_b

        mini_table[team_b]["GF"] += goals_b
        mini_table[team_b]["GA"] += goals_a

        if goals_a > goals_b:
            mini_table[team_a]["Pts"] += 3

        elif goals_b > goals_a:
            mini_table[team_b]["Pts"] += 3

        else:
            mini_table[team_a]["Pts"] += 1
            mini_table[team_b]["Pts"] += 1

    for team in tied_teams:
        mini_table[team]["GD"] = (
            mini_table[team]["GF"]
            -
            mini_table[team]["GA"]
        )

    return mini_table


def rank_tied_teams(
    tied_teams: list[str],
    table: dict,
    group_matches: list[dict],
) -> list[str]:
    if len(tied_teams) <= 1:
        return tied_teams

    mini_table = calculate_head_to_head_stats(
        tied_teams=tied_teams,
        group_matches=group_matches,
    )

    head_to_head_key = {
        team: (
            mini_table[team]["Pts"],
            mini_table[team]["GD"],
            mini_table[team]["GF"],
        )
        for team in tied_teams
    }

    sorted_teams = sorted(
        tied_teams,
        key=lambda team:
            head_to_head_key[team],
        reverse=True,
    )

    partitions = []

    for team in sorted_teams:
        if not partitions:
            partitions.append(
                [team]
            )

        else:
            previous_team = partitions[
                -1
            ][0]

            if (
                head_to_head_key[team]
                ==
                head_to_head_key[
                    previous_team
                ]
            ):
                partitions[-1].append(
                    team
                )

            else:
                partitions.append(
                    [team]
                )

    final_order = []

    for partition in partitions:
        if len(partition) == 1:
            final_order.extend(
                partition
            )

        elif len(partition) < len(
            tied_teams
        ):
            final_order.extend(
                rank_tied_teams(
                    tied_teams=partition,
                    table=table,
                    group_matches=group_matches,
                )
            )

        else:
            final_order.extend(
                sorted(
                    partition,
                    key=lambda team: (
                        table[team]["GD"],
                        table[team]["GF"],
                        table[team]["Elo"],
                    ),
                    reverse=True,
                )
            )

    return final_order


def rank_group(
    table: dict,
    group_matches: list[dict],
) -> list[str]:
    point_levels = sorted(
        {
            row["Pts"]
            for row in table.values()
        },
        reverse=True,
    )

    ranked_teams = []

    for point_level in point_levels:
        tied_teams = [
            team
            for team, row
            in table.items()
            if row["Pts"] == point_level
        ]

        ranked_teams.extend(
            rank_tied_teams(
                tied_teams=tied_teams,
                table=table,
                group_matches=group_matches,
            )
        )

    return ranked_teams


# =============================================================================
# 15. IMAGE FUNCTIONS
# =============================================================================

def save_dataframe_image(
    dataframe: pd.DataFrame,
    title: str,
    output_path: Path,
    footer: str = "",
    figsize_width: float = 18,
) -> None:
    figure_height = max(
        3.2,
        1.8
        +
        0.58
        *
        len(dataframe),
    )

    fig, axis = plt.subplots(
        figsize=(
            figsize_width,
            figure_height,
        )
    )

    axis.axis(
        "off"
    )

    axis.set_title(
        title,
        fontsize=17,
        fontweight="bold",
        pad=18,
    )

    table = axis.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(
        False
    )

    table.set_fontsize(
        9
    )

    table.scale(
        1,
        1.5,
    )

    for (
        row_index,
        column_index,
    ), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_text_props(
                weight="bold"
            )

        elif (
            column_index
            <
            len(dataframe.columns)
            and
            dataframe.columns[
                column_index
            ]
            in {
                "Team",
                "Team A",
                "Team B",
                "Winner",
                "Champion",
            }
        ):
            cell.set_text_props(
                weight="bold"
            )

    if footer:
        fig.text(
            0.5,
            0.025,
            footer,
            ha="center",
            fontsize=8,
        )

    fig.tight_layout(
        rect=[
            0.01,
            0.06,
            0.99,
            0.94,
        ]
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_champion_image(
    champion: str,
    runner_up: str,
    third_place: str,
    fourth_place: str,
    final_description: str,
) -> None:
    fig, axis = plt.subplots(
        figsize=(12, 8)
    )

    axis.axis(
        "off"
    )

    axis.text(
        0.5,
        0.88,
        "FIFA World Cup 2026 Simulation",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
    )

    axis.text(
        0.5,
        0.70,
        "SIMULATED CHAMPION",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )

    axis.text(
        0.5,
        0.55,
        champion,
        ha="center",
        va="center",
        fontsize=32,
        fontweight="bold",
    )

    axis.text(
        0.5,
        0.40,
        final_description,
        ha="center",
        va="center",
        fontsize=15,
    )

    axis.text(
        0.5,
        0.24,
        (
            f"Runner-up: {runner_up}\n"
            f"Third place: {third_place}\n"
            f"Fourth place: {fourth_place}"
        ),
        ha="center",
        va="center",
        fontsize=14,
        linespacing=1.6,
    )

    axis.text(
        0.5,
        0.07,
        (
            "One seeded simulation — not a guaranteed forecast"
        ),
        ha="center",
        va="center",
        fontsize=10,
    )

    fig.tight_layout()

    fig.savefig(
        ROUND_IMAGE_FOLDER
        /
        "simulated_champion.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# 16. GROUP-STAGE SIMULATION
# =============================================================================

def run_group_stage() -> dict:
    random_generator = np.random.default_rng(
        GROUP_STAGE_SEED
    )

    group_tables = {
        group_name:
            create_empty_group_table(
                group_name
            )
        for group_name in GROUPS
    }

    group_matches = {
        group_name: []
        for group_name in GROUPS
    }

    all_match_rows = []

    print("\n" + "=" * 96)
    print("GROUP STAGE")
    print("=" * 96)

    sorted_fixtures = sorted(
        GROUP_FIXTURES,
        key=lambda fixture: (
            fixture[1],
            fixture[0],
        ),
    )

    for index, fixture in enumerate(
        sorted_fixtures,
        start=1,
    ):
        (
            match_number,
            match_date,
            group_name,
            team_a_display,
            team_b_display,
            stadium,
            venue_country,
        ) = fixture

        team_a_model = DISPLAY_TO_MODEL[
            team_a_display
        ]

        team_b_model = DISPLAY_TO_MODEL[
            team_b_display
        ]

        match_seed = int(
            random_generator.integers(
                0,
                np.iinfo(
                    np.int32
                ).max,
            )
        )

        simulation = simulate_match(
            team_a=team_a_model,
            team_b=team_b_model,
            venue_country=venue_country,
            random_seed=match_seed,
        )

        goals_a = int(
            simulation[
                "team_a_goals"
            ]
        )

        goals_b = int(
            simulation[
                "team_b_goals"
            ]
        )

        update_group_table(
            table=group_tables[
                group_name
            ],
            team_a=team_a_display,
            team_b=team_b_display,
            goals_a=goals_a,
            goals_b=goals_b,
        )

        row = {
            "stage": "Group stage",
            "match_number": match_number,
            "date": match_date,
            "group": group_name,
            "team_a": team_a_display,
            "team_b": team_b_display,
            "team_a_model_name": team_a_model,
            "team_b_model_name": team_b_model,
            "stadium": stadium,
            "venue_country": venue_country,
            "venue_type": simulation[
                "venue_type"
            ],
            "team_a_expected_goals": round(
                simulation[
                    "team_a_expected_goals"
                ],
                4,
            ),
            "team_b_expected_goals": round(
                simulation[
                    "team_b_expected_goals"
                ],
                4,
            ),
            "team_a_win_probability": round(
                simulation[
                    "team_a_win_probability"
                ],
                6,
            ),
            "draw_probability": round(
                simulation[
                    "draw_probability"
                ],
                6,
            ),
            "team_b_win_probability": round(
                simulation[
                    "team_b_win_probability"
                ],
                6,
            ),
            "team_a_goals": goals_a,
            "team_b_goals": goals_b,
            "score": (
                f"{goals_a}-{goals_b}"
            ),
            "winner": (
                team_a_display
                if goals_a > goals_b
                else
                team_b_display
                if goals_b > goals_a
                else
                ""
            ),
            "decision": (
                "90 minutes"
                if goals_a != goals_b
                else
                "Draw"
            ),
            "match_seed": match_seed,
        }

        all_match_rows.append(
            row
        )

        group_matches[
            group_name
        ].append(
            row
        )

        print(
            f"{index:>2}/72 | "
            f"Group {group_name} | "
            f"{team_a_display} "
            f"{goals_a}-{goals_b} "
            f"{team_b_display}"
        )

    ranked_groups = {
        group_name: rank_group(
            table=group_tables[
                group_name
            ],
            group_matches=group_matches[
                group_name
            ],
        )
        for group_name in GROUPS
    }

    third_place_rows = []

    for group_name, ranked_teams in (
        ranked_groups.items()
    ):
        third_team = ranked_teams[2]

        statistics = group_tables[
            group_name
        ][
            third_team
        ]

        third_place_rows.append({
            "Group": group_name,
            "Team": third_team,
            "P": statistics["P"],
            "W": statistics["W"],
            "D": statistics["D"],
            "L": statistics["L"],
            "GF": statistics["GF"],
            "GA": statistics["GA"],
            "GD": statistics["GD"],
            "Pts": statistics["Pts"],
            "Elo": statistics["Elo"],
        })

    third_place_rows = sorted(
        third_place_rows,
        key=lambda row: (
            row["Pts"],
            row["GD"],
            row["GF"],
            row["Elo"],
        ),
        reverse=True,
    )

    qualified_third_teams = {
        row["Team"]
        for row in third_place_rows[
            :8
        ]
    }

    standings_rows = []
    qualification_rows = []
    group_positions = {}
    group_dataframes = {}

    for group_name, ranked_teams in (
        ranked_groups.items()
    ):
        group_rows = []

        for rank, team in enumerate(
            ranked_teams,
            start=1,
        ):
            statistics = group_tables[
                group_name
            ][
                team
            ]

            group_positions[
                f"{rank}{group_name}"
            ] = team

            if rank <= 2:
                status = "Qualified"

                qualification_rows.append({
                    "Team": team,
                    "Group": group_name,
                    "GroupRank": rank,
                    "Route": (
                        f"Group {group_name} "
                        f"position {rank}"
                    ),
                })

            elif (
                rank == 3
                and
                team
                in qualified_third_teams
            ):
                status = (
                    "Qualified (best 3rd)"
                )

                qualification_rows.append({
                    "Team": team,
                    "Group": group_name,
                    "GroupRank": 3,
                    "Route": (
                        "Best third-placed team"
                    ),
                })

            else:
                status = "Eliminated"

            row = {
                "Rank": rank,
                "Team": team,
                "P": statistics["P"],
                "W": statistics["W"],
                "D": statistics["D"],
                "L": statistics["L"],
                "GF": statistics["GF"],
                "GA": statistics["GA"],
                "GD": statistics["GD"],
                "Pts": statistics["Pts"],
                "Status": status,
            }

            group_rows.append(
                row
            )

            standings_rows.append({
                "Group": group_name,
                **row,
            })

        group_dataframe = pd.DataFrame(
            group_rows
        )

        group_dataframes[
            group_name
        ] = group_dataframe

        save_dataframe_image(
            dataframe=group_dataframe,
            title=(
                "FIFA World Cup 2026 Simulation "
                f"— Group {group_name}"
            ),
            output_path=(
                GROUP_IMAGE_FOLDER
                /
                f"Group_{group_name}_standings.png"
            ),
            footer=(
                "P=Played | W=Wins | D=Draws | L=Losses | "
                "GF=Goals For | GA=Goals Against | "
                "GD=Goal Difference | Pts=Points"
            ),
            figsize_width=15,
        )

    third_output_rows = []

    for rank, row in enumerate(
        third_place_rows,
        start=1,
    ):
        third_output_rows.append({
            "ThirdRank": rank,
            "Group": row["Group"],
            "Team": row["Team"],
            "P": row["P"],
            "W": row["W"],
            "D": row["D"],
            "L": row["L"],
            "GF": row["GF"],
            "GA": row["GA"],
            "GD": row["GD"],
            "Pts": row["Pts"],
            "Status": (
                "Qualified"
                if rank <= 8
                else
                "Eliminated"
            ),
        })

    matches_df = pd.DataFrame(
        all_match_rows
    ).sort_values(
        "match_number"
    )

    standings_df = pd.DataFrame(
        standings_rows
    ).sort_values(
        [
            "Group",
            "Rank",
        ]
    )

    third_place_df = pd.DataFrame(
        third_output_rows
    )

    qualification_df = pd.DataFrame(
        qualification_rows
    )

    matches_df.to_csv(
        OUTPUT_FOLDER
        /
        "group_stage_matches.csv",
        index=False,
    )

    standings_df.to_csv(
        OUTPUT_FOLDER
        /
        "all_group_standings.csv",
        index=False,
    )

    third_place_df.to_csv(
        OUTPUT_FOLDER
        /
        "third_place_ranking.csv",
        index=False,
    )

    qualification_df.to_csv(
        OUTPUT_FOLDER
        /
        "group_stage_qualifiers.csv",
        index=False,
    )

    save_dataframe_image(
        dataframe=third_place_df,
        title=(
            "FIFA World Cup 2026 Simulation "
            "— Third-Place Ranking"
        ),
        output_path=(
            GROUP_IMAGE_FOLDER
            /
            "Best_third_placed_teams.png"
        ),
        footer=(
            "The first eight third-placed teams qualify. "
            "Fair-play cards are not simulated."
        ),
        figsize_width=17,
    )

    third_team_by_group = {
        row["Group"]: row["Team"]
        for row in third_place_rows[:8]
    }

    qualified_third_group_set = "".join(
        sorted(
            third_team_by_group.keys()
        )
    )

    if len(qualification_df) != 32:
        raise ValueError(
            "Group stage should produce exactly 32 qualifiers."
        )

    return {
        "matches": matches_df,
        "standings": standings_df,
        "third_place": third_place_df,
        "qualifiers": qualification_df,
        "group_positions": group_positions,
        "third_team_by_group":
            third_team_by_group,
        "qualified_third_group_set":
            qualified_third_group_set,
    }


# =============================================================================
# 17. ANNEX C THIRD-PLACE ALLOCATION
# =============================================================================

def flatten_column_name(
    column
) -> str:
    if isinstance(
        column,
        tuple,
    ):
        parts = [
            str(value).strip()
            for value in column
            if (
                str(value).strip()
                and
                str(value).lower()
                !=
                "nan"
                and
                not str(value)
                .lower()
                .startswith(
                    "unnamed"
                )
            )
        ]

        return " ".join(
            dict.fromkeys(
                parts
            )
        )

    return str(
        column
    ).strip()


def compact_column_name(
    value: str
) -> str:
    return re.sub(
        r"[^A-Z0-9]",
        "",
        value.upper(),
    )


def normalize_advancing_group_set(
    value
) -> str | None:
    letters = re.findall(
        r"[A-L]",
        str(value).upper(),
    )

    unique_letters = sorted(
        set(letters)
    )

    if len(unique_letters) != 8:
        return None

    return "".join(
        unique_letters
    )


def normalize_third_slot(
    value
) -> str | None:
    match = re.search(
        r"3\s*([A-L])",
        str(value).upper(),
    )

    if not match:
        return None

    return match.group(1)


def validate_annex_mapping(
    mapping_df: pd.DataFrame
) -> None:
    required_columns = (
        ["GroupSet"]
        +
        THIRD_PLACE_WINNER_COLUMNS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in mapping_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Annex C mapping is missing columns:\n"
            f"{missing_columns}"
        )

    for _, row in mapping_df.iterrows():
        advancing_groups = set(
            str(
                row["GroupSet"]
            )
        )

        assigned_groups = {
            str(
                row[winner_slot]
            )
            for winner_slot
            in THIRD_PLACE_WINNER_COLUMNS
        }

        if (
            assigned_groups
            !=
            advancing_groups
        ):
            raise ValueError(
                "Annex C row does not use each advancing "
                f"group exactly once: {row['GroupSet']}"
            )

        for winner_slot in (
            THIRD_PLACE_WINNER_COLUMNS
        ):
            assigned_group = str(
                row[winner_slot]
            )

            if (
                assigned_group
                not in
                ALLOWED_THIRD_GROUPS[
                    winner_slot
                ]
            ):
                raise ValueError(
                    f"Invalid assignment: "
                    f"{winner_slot} vs 3"
                    f"{assigned_group}"
                )


def download_annex_mapping() -> pd.DataFrame:
    print(
        "\nDownloading official Annex C "
        "third-place allocation table..."
    )

    tables = pd.read_html(
        ANNEX_C_TABLE_URL
    )

    selected_table = None
    flattened_columns = None

    for table in tables:
        column_names = [
            flatten_column_name(
                column
            )
            for column in table.columns
        ]

        compact_names = [
            compact_column_name(
                column
            )
            for column in column_names
        ]

        if (
            any(
                "1A"
                in column
                for column in compact_names
            )
            and
            any(
                "1L"
                in column
                for column in compact_names
            )
        ):
            selected_table = (
                table.copy()
            )

            flattened_columns = (
                column_names
            )

            break

    if selected_table is None:
        raise RuntimeError(
            "Annex C table could not be found online."
        )

    selected_table.columns = (
        flattened_columns
    )

    winner_column_lookup = {}

    for winner_slot in (
        THIRD_PLACE_WINNER_COLUMNS
    ):
        matches = [
            column
            for column
            in selected_table.columns
            if winner_slot
            in compact_column_name(
                column
            )
        ]

        if not matches:
            raise RuntimeError(
                f"Annex C column {winner_slot} "
                "was not found."
            )

        winner_column_lookup[
            winner_slot
        ] = matches[0]

    used_columns = set(
        winner_column_lookup.values()
    )

    best_advancing_column = None
    best_valid_count = -1

    for column in selected_table.columns:
        if column in used_columns:
            continue

        valid_count = sum(
            normalize_advancing_group_set(
                value
            )
            is not None
            for value
            in selected_table[column]
        )

        if valid_count > best_valid_count:
            best_valid_count = valid_count
            best_advancing_column = (
                column
            )

    mapping_rows = []

    for _, row in selected_table.iterrows():
        group_set = (
            normalize_advancing_group_set(
                row[
                    best_advancing_column
                ]
            )
        )

        if group_set is None:
            continue

        output_row = {
            "GroupSet": group_set
        }

        valid_row = True

        for winner_slot, source_column in (
            winner_column_lookup.items()
        ):
            assigned_group = (
                normalize_third_slot(
                    row[
                        source_column
                    ]
                )
            )

            if assigned_group is None:
                valid_row = False
                break

            output_row[
                winner_slot
            ] = assigned_group

        if valid_row:
            mapping_rows.append(
                output_row
            )

    mapping_df = (
        pd.DataFrame(
            mapping_rows
        )
        .drop_duplicates(
            subset=[
                "GroupSet"
            ]
        )
        .sort_values(
            "GroupSet"
        )
        .reset_index(
            drop=True
        )
    )

    if len(mapping_df) != 495:
        raise RuntimeError(
            "The downloaded Annex C table did not produce "
            f"495 combinations. Produced: {len(mapping_df)}"
        )

    validate_annex_mapping(
        mapping_df
    )

    mapping_df.to_csv(
        ANNEX_C_CACHE_PATH,
        index=False,
    )

    return mapping_df


def solve_fallback_third_assignment(
    advancing_groups: set[str]
) -> dict:
    """
    Produce a deterministic valid matching if the official
    Annex C table cannot be downloaded.

    This respects every allowed winner-v-third-group pairing,
    but may not reproduce FIFA's exact Annex C row.
    """

    winner_slots = sorted(
        THIRD_PLACE_WINNER_COLUMNS,
        key=lambda slot: len(
            ALLOWED_THIRD_GROUPS[
                slot
            ]
            &
            advancing_groups
        ),
    )

    solution = {}

    def backtrack(
        index: int,
        unused_groups: set[str],
    ) -> bool:
        if index == len(
            winner_slots
        ):
            return True

        winner_slot = winner_slots[
            index
        ]

        candidates = sorted(
            ALLOWED_THIRD_GROUPS[
                winner_slot
            ]
            &
            unused_groups
        )

        for group_letter in candidates:
            solution[
                winner_slot
            ] = group_letter

            if backtrack(
                index + 1,
                unused_groups
                -
                {
                    group_letter
                },
            ):
                return True

            solution.pop(
                winner_slot,
                None,
            )

        return False

    if not backtrack(
        0,
        set(
            advancing_groups
        ),
    ):
        raise RuntimeError(
            "A valid fallback third-place assignment "
            "could not be created."
        )

    return solution


def get_third_place_assignment(
    qualified_group_set: str
) -> tuple[dict, str]:
    try:
        if ANNEX_C_CACHE_PATH.exists():
            mapping_df = pd.read_csv(
                ANNEX_C_CACHE_PATH,
                dtype=str,
            )

            validate_annex_mapping(
                mapping_df
            )

        else:
            mapping_df = (
                download_annex_mapping()
            )

        matching_rows = mapping_df[
            mapping_df["GroupSet"]
            ==
            qualified_group_set
        ]

        if len(matching_rows) != 1:
            raise RuntimeError(
                "The exact Annex C row was not found."
            )

        row = matching_rows.iloc[0]

        assignment = {
            winner_slot: str(
                row[
                    winner_slot
                ]
            )
            for winner_slot
            in THIRD_PLACE_WINNER_COLUMNS
        }

        return (
            assignment,
            "Official Annex C mapping",
        )

    except Exception as error:
        print(
            "\nWARNING: Official Annex C table "
            "could not be loaded."
        )

        print(
            f"Reason: {error}"
        )

        print(
            "Using a valid deterministic fallback "
            "assignment instead."
        )

        fallback_assignment = (
            solve_fallback_third_assignment(
                set(
                    qualified_group_set
                )
            )
        )

        return (
            fallback_assignment,
            "Fallback valid assignment",
        )


# =============================================================================
# 18. KNOCKOUT RESOLUTION
# =============================================================================

def resolve_knockout_match(
    simulation: dict,
    team_a_display: str,
    team_b_display: str,
    resolution_seed: int,
) -> dict:
    goals_a_90 = int(
        simulation[
            "team_a_goals"
        ]
    )

    goals_b_90 = int(
        simulation[
            "team_b_goals"
        ]
    )

    if goals_a_90 > goals_b_90:
        return {
            "goals_a_90": goals_a_90,
            "goals_b_90": goals_b_90,
            "extra_time_goals_a": 0,
            "extra_time_goals_b": 0,
            "goals_a_final": goals_a_90,
            "goals_b_final": goals_b_90,
            "winner": team_a_display,
            "loser": team_b_display,
            "decision": "90 minutes",
            "penalty_winner": "",
        }

    if goals_b_90 > goals_a_90:
        return {
            "goals_a_90": goals_a_90,
            "goals_b_90": goals_b_90,
            "extra_time_goals_a": 0,
            "extra_time_goals_b": 0,
            "goals_a_final": goals_a_90,
            "goals_b_final": goals_b_90,
            "winner": team_b_display,
            "loser": team_a_display,
            "decision": "90 minutes",
            "penalty_winner": "",
        }

    random_generator = np.random.default_rng(
        resolution_seed
    )

    extra_lambda_a = (
        simulation[
            "team_a_expected_goals"
        ]
        *
        EXTRA_TIME_MINUTE_FACTOR
        *
        EXTRA_TIME_SCORING_FACTOR
    )

    extra_lambda_b = (
        simulation[
            "team_b_expected_goals"
        ]
        *
        EXTRA_TIME_MINUTE_FACTOR
        *
        EXTRA_TIME_SCORING_FACTOR
    )

    extra_goals_a = int(
        random_generator.poisson(
            extra_lambda_a
        )
    )

    extra_goals_b = int(
        random_generator.poisson(
            extra_lambda_b
        )
    )

    final_goals_a = (
        goals_a_90
        +
        extra_goals_a
    )

    final_goals_b = (
        goals_b_90
        +
        extra_goals_b
    )

    if final_goals_a > final_goals_b:
        return {
            "goals_a_90": goals_a_90,
            "goals_b_90": goals_b_90,
            "extra_time_goals_a": extra_goals_a,
            "extra_time_goals_b": extra_goals_b,
            "goals_a_final": final_goals_a,
            "goals_b_final": final_goals_b,
            "winner": team_a_display,
            "loser": team_b_display,
            "decision": "After extra time",
            "penalty_winner": "",
        }

    if final_goals_b > final_goals_a:
        return {
            "goals_a_90": goals_a_90,
            "goals_b_90": goals_b_90,
            "extra_time_goals_a": extra_goals_a,
            "extra_time_goals_b": extra_goals_b,
            "goals_a_final": final_goals_a,
            "goals_b_final": final_goals_b,
            "winner": team_b_display,
            "loser": team_a_display,
            "decision": "After extra time",
            "penalty_winner": "",
        }

    team_a_strength = float(
        simulation[
            "team_a_win_probability"
        ]
    )

    team_b_strength = float(
        simulation[
            "team_b_win_probability"
        ]
    )

    non_draw_total = (
        team_a_strength
        +
        team_b_strength
    )

    if non_draw_total <= 0:
        raw_team_a_probability = 0.5

    else:
        raw_team_a_probability = (
            team_a_strength
            /
            non_draw_total
        )

    # Shrink towards 50% because shootouts are highly uncertain.
    team_a_penalty_probability = (
        0.5
        +
        0.5
        *
        (
            raw_team_a_probability
            -
            0.5
        )
    )

    if (
        random_generator.random()
        <
        team_a_penalty_probability
    ):
        penalty_winner = (
            team_a_display
        )

        penalty_loser = (
            team_b_display
        )

    else:
        penalty_winner = (
            team_b_display
        )

        penalty_loser = (
            team_a_display
        )

    return {
        "goals_a_90": goals_a_90,
        "goals_b_90": goals_b_90,
        "extra_time_goals_a": extra_goals_a,
        "extra_time_goals_b": extra_goals_b,
        "goals_a_final": final_goals_a,
        "goals_b_final": final_goals_b,
        "winner": penalty_winner,
        "loser": penalty_loser,
        "decision": "Penalties",
        "penalty_winner": penalty_winner,
    }


def format_knockout_score(
    resolution: dict
) -> str:
    goals_a = int(
        resolution[
            "goals_a_final"
        ]
    )

    goals_b = int(
        resolution[
            "goals_b_final"
        ]
    )

    if (
        resolution["decision"]
        ==
        "Penalties"
    ):
        return (
            f"{goals_a}-{goals_b} "
            f"({resolution['penalty_winner']} won pens)"
        )

    if (
        resolution["decision"]
        ==
        "After extra time"
    ):
        return (
            f"{goals_a}-{goals_b} AET"
        )

    return (
        f"{goals_a}-{goals_b}"
    )


# =============================================================================
# 19. GENERIC KNOCKOUT ROUND
# =============================================================================

def simulate_knockout_round(
    stage_name: str,
    match_definitions: list[tuple],
    slot_teams: dict,
    round_seed: int,
) -> tuple[pd.DataFrame, dict]:
    random_generator = np.random.default_rng(
        round_seed
    )

    result_rows = []
    new_slots = {}

    print("\n" + "=" * 96)
    print(stage_name.upper())
    print("=" * 96)

    for match_definition in (
        match_definitions
    ):
        (
            match_number,
            match_date,
            source_a,
            source_b,
            stadium,
            venue_country,
        ) = match_definition

        if source_a not in slot_teams:
            raise KeyError(
                f"Missing bracket source: {source_a}"
            )

        if source_b not in slot_teams:
            raise KeyError(
                f"Missing bracket source: {source_b}"
            )

        team_a_display = slot_teams[
            source_a
        ]

        team_b_display = slot_teams[
            source_b
        ]

        team_a_model = DISPLAY_TO_MODEL[
            team_a_display
        ]

        team_b_model = DISPLAY_TO_MODEL[
            team_b_display
        ]

        match_seed = int(
            random_generator.integers(
                0,
                np.iinfo(
                    np.int32
                ).max,
            )
        )

        simulation = simulate_match(
            team_a=team_a_model,
            team_b=team_b_model,
            venue_country=venue_country,
            random_seed=match_seed,
        )

        resolution = resolve_knockout_match(
            simulation=simulation,
            team_a_display=team_a_display,
            team_b_display=team_b_display,
            resolution_seed=(
                match_seed + 1
            ),
        )

        score_text = format_knockout_score(
            resolution
        )

        winner_slot = (
            f"W{match_number}"
        )

        loser_slot = (
            f"L{match_number}"
        )

        new_slots[
            winner_slot
        ] = resolution[
            "winner"
        ]

        new_slots[
            loser_slot
        ] = resolution[
            "loser"
        ]

        row = {
            "stage": stage_name,
            "match_number": match_number,
            "date": match_date,
            "source_a": source_a,
            "source_b": source_b,
            "team_a": team_a_display,
            "team_b": team_b_display,
            "team_a_model_name": team_a_model,
            "team_b_model_name": team_b_model,
            "stadium": stadium,
            "venue_country": venue_country,
            "venue_type": simulation[
                "venue_type"
            ],
            "team_a_expected_goals": round(
                simulation[
                    "team_a_expected_goals"
                ],
                4,
            ),
            "team_b_expected_goals": round(
                simulation[
                    "team_b_expected_goals"
                ],
                4,
            ),
            "team_a_win_probability": round(
                simulation[
                    "team_a_win_probability"
                ],
                6,
            ),
            "draw_probability": round(
                simulation[
                    "draw_probability"
                ],
                6,
            ),
            "team_b_win_probability": round(
                simulation[
                    "team_b_win_probability"
                ],
                6,
            ),
            "goals_a_90": resolution[
                "goals_a_90"
            ],
            "goals_b_90": resolution[
                "goals_b_90"
            ],
            "extra_time_goals_a":
                resolution[
                    "extra_time_goals_a"
                ],
            "extra_time_goals_b":
                resolution[
                    "extra_time_goals_b"
                ],
            "goals_a_final":
                resolution[
                    "goals_a_final"
                ],
            "goals_b_final":
                resolution[
                    "goals_b_final"
                ],
            "score": score_text,
            "decision": resolution[
                "decision"
            ],
            "winner": resolution[
                "winner"
            ],
            "loser": resolution[
                "loser"
            ],
            "winner_slot": winner_slot,
            "loser_slot": loser_slot,
            "match_seed": match_seed,
        }

        result_rows.append(
            row
        )

        print(
            f"M{match_number}: "
            f"{team_a_display} "
            f"{score_text} "
            f"{team_b_display} "
            f"→ {resolution['winner']}"
        )

    results_df = pd.DataFrame(
        result_rows
    ).sort_values(
        "match_number"
    )

    display_df = pd.DataFrame({
        "Match": (
            "M"
            +
            results_df[
                "match_number"
            ].astype(str)
        ),
        "Date": results_df["date"],
        "Team A": results_df["team_a"],
        "Score": results_df["score"],
        "Team B": results_df["team_b"],
        "Decision": results_df[
            "decision"
        ],
        "Winner": results_df["winner"],
        "Venue": (
            results_df["stadium"]
            +
            ", "
            +
            results_df[
                "venue_country"
            ]
        ),
    })

    filename = (
        stage_name
        .lower()
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )

    results_df.to_csv(
        OUTPUT_FOLDER
        /
        f"{filename}.csv",
        index=False,
    )

    save_dataframe_image(
        dataframe=display_df,
        title=(
            "FIFA World Cup 2026 Simulation "
            f"— {stage_name}"
        ),
        output_path=(
            ROUND_IMAGE_FOLDER
            /
            f"{filename}.png"
        ),
        footer=(
            "Tied knockout matches proceed to extra time "
            "and penalties."
        ),
        figsize_width=21,
    )

    return (
        results_df,
        new_slots,
    )


# =============================================================================
# 20. BUILD ROUND-OF-32 SLOTS
# =============================================================================

def build_round_of_32_slots(
    group_results: dict
) -> tuple[dict, pd.DataFrame]:
    qualified_group_set = (
        group_results[
            "qualified_third_group_set"
        ]
    )

    assignment, source = (
        get_third_place_assignment(
            qualified_group_set
        )
    )

    slots = dict(
        group_results[
            "group_positions"
        ]
    )

    assignment_rows = []

    for winner_slot in (
        THIRD_PLACE_WINNER_COLUMNS
    ):
        third_group = assignment[
            winner_slot
        ]

        third_team = (
            group_results[
                "third_team_by_group"
            ][
                third_group
            ]
        )

        slots[
            f"THIRD_{winner_slot}"
        ] = third_team

        assignment_rows.append({
            "WinnerSlot": winner_slot,
            "ThirdPlaceGroup":
                third_group,
            "ThirdPlaceTeam":
                third_team,
            "AssignmentSource":
                source,
        })

    assignment_df = pd.DataFrame(
        assignment_rows
    )

    assignment_df.to_csv(
        OUTPUT_FOLDER
        /
        "third_place_round_of_32_assignment.csv",
        index=False,
    )

    return (
        slots,
        assignment_df,
    )


# =============================================================================
# 21. MAIN COMPLETE TOURNAMENT
# =============================================================================

def run_complete_tournament() -> dict:
    validate_tournament_configuration()

    print("=" * 100)
    print("FIFA WORLD CUP 2026 — COMPLETE STANDALONE SIMULATION")
    print("=" * 100)

    print(
        "\nHybrid weights:"
    )

    print(
        "Random Forest:",
        f"{CLASSIFIER_WEIGHT:.0%}"
    )

    print(
        "Poisson:",
        f"{POISSON_WEIGHT:.0%}"
    )

    mapping_df = pd.DataFrame([
        {
            "DisplayName": display_name,
            "ModelName": model_name,
        }
        for display_name, model_name
        in sorted(
            DISPLAY_TO_MODEL.items()
        )
    ])

    mapping_df.to_csv(
        OUTPUT_FOLDER
        /
        "team_name_mapping.csv",
        index=False,
    )

    group_results = run_group_stage()

    (
        slot_teams,
        third_assignment_df,
    ) = build_round_of_32_slots(
        group_results
    )

    (
        round_of_32_df,
        round_of_32_slots,
    ) = simulate_knockout_round(
        stage_name="Round of 32",
        match_definitions=
            ROUND_OF_32_MATCHES,
        slot_teams=slot_teams,
        round_seed=ROUND_OF_32_SEED,
    )

    slot_teams.update(
        round_of_32_slots
    )

    (
        round_of_16_df,
        round_of_16_slots,
    ) = simulate_knockout_round(
        stage_name="Round of 16",
        match_definitions=
            ROUND_OF_16_MATCHES,
        slot_teams=slot_teams,
        round_seed=ROUND_OF_16_SEED,
    )

    slot_teams.update(
        round_of_16_slots
    )

    (
        quarter_final_df,
        quarter_final_slots,
    ) = simulate_knockout_round(
        stage_name="Quarter-finals",
        match_definitions=
            QUARTER_FINAL_MATCHES,
        slot_teams=slot_teams,
        round_seed=QUARTER_FINAL_SEED,
    )

    slot_teams.update(
        quarter_final_slots
    )

    (
        semi_final_df,
        semi_final_slots,
    ) = simulate_knockout_round(
        stage_name="Semi-finals",
        match_definitions=
            SEMI_FINAL_MATCHES,
        slot_teams=slot_teams,
        round_seed=SEMI_FINAL_SEED,
    )

    slot_teams.update(
        semi_final_slots
    )

    (
        third_place_df,
        third_place_slots,
    ) = simulate_knockout_round(
        stage_name="Third-place play-off",
        match_definitions=
            THIRD_PLACE_MATCHES,
        slot_teams=slot_teams,
        round_seed=THIRD_PLACE_SEED,
    )

    slot_teams.update(
        third_place_slots
    )

    (
        final_df,
        final_slots,
    ) = simulate_knockout_round(
        stage_name="Final",
        match_definitions=
            FINAL_MATCHES,
        slot_teams=slot_teams,
        round_seed=FINAL_SEED,
    )

    slot_teams.update(
        final_slots
    )

    champion = final_df.iloc[0][
        "winner"
    ]

    runner_up = final_df.iloc[0][
        "loser"
    ]

    third_place = third_place_df.iloc[0][
        "winner"
    ]

    fourth_place = third_place_df.iloc[0][
        "loser"
    ]

    all_knockout_df = pd.concat(
        [
            round_of_32_df,
            round_of_16_df,
            quarter_final_df,
            semi_final_df,
            third_place_df,
            final_df,
        ],
        ignore_index=True,
        sort=False,
    ).sort_values(
        "match_number"
    )

    all_knockout_df.to_csv(
        OUTPUT_FOLDER
        /
        "all_knockout_matches.csv",
        index=False,
    )

    all_matches_df = pd.concat(
        [
            group_results[
                "matches"
            ],
            all_knockout_df,
        ],
        ignore_index=True,
        sort=False,
    ).sort_values(
        "match_number"
    )

    if len(all_matches_df) != 104:
        raise ValueError(
            "The complete tournament output must contain "
            f"104 matches, but contains {len(all_matches_df)}."
        )

    all_matches_df.to_csv(
        OUTPUT_FOLDER
        /
        "all_104_matches.csv",
        index=False,
    )

    podium_df = pd.DataFrame([
        {
            "Position": 1,
            "Team": champion,
            "Status": "Champion",
        },
        {
            "Position": 2,
            "Team": runner_up,
            "Status": "Runner-up",
        },
        {
            "Position": 3,
            "Team": third_place,
            "Status": "Third place",
        },
        {
            "Position": 4,
            "Team": fourth_place,
            "Status": "Fourth place",
        },
    ])

    podium_df.to_csv(
        OUTPUT_FOLDER
        /
        "tournament_podium.csv",
        index=False,
    )

    save_dataframe_image(
        dataframe=podium_df,
        title=(
            "FIFA World Cup 2026 Simulation "
            "— Final Tournament Positions"
        ),
        output_path=(
            ROUND_IMAGE_FOLDER
            /
            "tournament_podium.png"
        ),
        figsize_width=12,
    )

    final_row = final_df.iloc[0]

    final_description = (
        f"{final_row['team_a']} "
        f"{final_row['score']} "
        f"{final_row['team_b']}"
    )

    save_champion_image(
        champion=champion,
        runner_up=runner_up,
        third_place=third_place,
        fourth_place=fourth_place,
        final_description=
            final_description,
    )

    complete_knockout_display = pd.DataFrame({
        "Stage": all_knockout_df[
            "stage"
        ],
        "Match": (
            "M"
            +
            all_knockout_df[
                "match_number"
            ].astype(str)
        ),
        "Team A": all_knockout_df[
            "team_a"
        ],
        "Score": all_knockout_df[
            "score"
        ],
        "Team B": all_knockout_df[
            "team_b"
        ],
        "Winner": all_knockout_df[
            "winner"
        ],
    })

    save_dataframe_image(
        dataframe=complete_knockout_display,
        title=(
            "FIFA World Cup 2026 Simulation "
            "— Complete Knockout Results"
        ),
        output_path=(
            ROUND_IMAGE_FOLDER
            /
            "complete_knockout_results.png"
        ),
        figsize_width=18,
    )

    with open(
        OUTPUT_FOLDER
        /
        "tournament_summary.txt",
        "w",
        encoding="utf-8",
    ) as summary_file:
        summary_file.write(
            "FIFA WORLD CUP 2026 COMPLETE SIMULATION\n"
        )

        summary_file.write(
            "=" * 60 + "\n\n"
        )

        summary_file.write(
            f"Champion: {champion}\n"
        )

        summary_file.write(
            f"Runner-up: {runner_up}\n"
        )

        summary_file.write(
            f"Third place: {third_place}\n"
        )

        summary_file.write(
            f"Fourth place: {fourth_place}\n"
        )

        summary_file.write(
            f"Final: {final_description}\n\n"
        )

        summary_file.write(
            "Hybrid weights\n"
        )

        summary_file.write(
            f"Random Forest: "
            f"{CLASSIFIER_WEIGHT:.4f}\n"
        )

        summary_file.write(
            f"Poisson: "
            f"{POISSON_WEIGHT:.4f}\n\n"
        )

        summary_file.write(
            "Seeds\n"
        )

        summary_file.write(
            f"Group stage: {GROUP_STAGE_SEED}\n"
        )

        summary_file.write(
            f"Round of 32: {ROUND_OF_32_SEED}\n"
        )

        summary_file.write(
            f"Round of 16: {ROUND_OF_16_SEED}\n"
        )

        summary_file.write(
            f"Quarter-finals: {QUARTER_FINAL_SEED}\n"
        )

        summary_file.write(
            f"Semi-finals: {SEMI_FINAL_SEED}\n"
        )

        summary_file.write(
            f"Third place: {THIRD_PLACE_SEED}\n"
        )

        summary_file.write(
            f"Final: {FINAL_SEED}\n\n"
        )

        summary_file.write(
            "This is one stochastic model simulation, "
            "not a guaranteed forecast.\n"
        )

    print("\n" + "=" * 100)
    print("COMPLETE TOURNAMENT FINISHED")
    print("=" * 100)

    print(
        "\nSIMULATED CHAMPION:",
        champion
    )

    print(
        "Runner-up:",
        runner_up
    )

    print(
        "Third place:",
        third_place
    )

    print(
        "Fourth place:",
        fourth_place
    )

    print(
        "\nFinal:"
    )

    print(
        final_description
    )

    print(
        "\nAll outputs saved to:"
    )

    print(
        OUTPUT_FOLDER
    )

    return {
        "champion": champion,
        "runner_up": runner_up,
        "third_place": third_place,
        "fourth_place": fourth_place,
        "group_stage": group_results,
        "third_place_assignment":
            third_assignment_df,
        "round_of_32": round_of_32_df,
        "round_of_16": round_of_16_df,
        "quarter_finals":
            quarter_final_df,
        "semi_finals": semi_final_df,
        "third_place_match":
            third_place_df,
        "final": final_df,
        "all_104_matches":
            all_matches_df,
    }




# =============================================================================
# 22. MONTE CARLO FINAL-PRODUCT SETTINGS
# =============================================================================

# Recommended final run:
N_SIMULATIONS = 10_000

# Use 1,000 first only when testing the script.
MONTE_CARLO_SEED = 20260610

# Console progress frequency.
PROGRESS_INTERVAL = 100

MONTE_CARLO_OUTPUT_FOLDER = (
    DATA_FOLDER /
    "world_cup_2026_monte_carlo_results"
)

MONTE_CARLO_IMAGE_FOLDER = (
    MONTE_CARLO_OUTPUT_FOLDER /
    "probability_images"
)

MONTE_CARLO_GROUP_IMAGE_FOLDER = (
    MONTE_CARLO_IMAGE_FOLDER /
    "group_probability_tables"
)

MONTE_CARLO_OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

MONTE_CARLO_IMAGE_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

MONTE_CARLO_GROUP_IMAGE_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 23. MONTE CARLO MATCH-DISTRIBUTION CACHE
# =============================================================================

def build_cached_distribution(
    team_a_display: str,
    team_b_display: str,
    venue_type: str,
) -> dict:
    """
    Build one score distribution without sampling a result.

    Models are called only during precomputation. All tournament
    simulations later sample from the cached cumulative distribution.
    """

    team_a_model = DISPLAY_TO_MODEL[
        team_a_display
    ]

    team_b_model = DISPLAY_TO_MODEL[
        team_b_display
    ]

    expected_goals = (
        get_venue_aware_expected_goals(
            team_a=team_a_model,
            team_b=team_b_model,
            venue_type=venue_type,
        )
    )

    score_matrix = create_poisson_score_matrix(
        team_a_expected_goals=expected_goals[
            "team_a_expected_goals"
        ],
        team_b_expected_goals=expected_goals[
            "team_b_expected_goals"
        ],
    )

    poisson_probabilities = (
        calculate_matrix_outcome_probabilities(
            score_matrix
        )
    )

    if CLASSIFIER_WEIGHT > 0.0:
        classifier_probabilities = (
            get_venue_aware_classifier_probabilities(
                team_a=team_a_model,
                team_b=team_b_model,
                venue_type=venue_type,
            )
        )

        final_probabilities = (
            blend_outcome_probabilities(
                classifier_probabilities=
                    classifier_probabilities,
                poisson_probabilities=
                    poisson_probabilities,
            )
        )

        final_score_matrix = adjust_score_matrix(
            score_matrix=score_matrix,
            target_probabilities=
                final_probabilities,
        )

    else:
        # Your tuned model currently selects 100% Poisson.
        final_probabilities = (
            poisson_probabilities
        )

        final_score_matrix = (
            score_matrix
        )

    flattened_probabilities = (
        final_score_matrix
        .flatten()
    )

    cumulative_probabilities = (
        np.cumsum(
            flattened_probabilities
        )
    )

    cumulative_probabilities[
        -1
    ] = 1.0

    return {
        "team_a": team_a_display,
        "team_b": team_b_display,
        "venue_type": venue_type,

        "team_a_expected_goals":
            float(
                expected_goals[
                    "team_a_expected_goals"
                ]
            ),

        "team_b_expected_goals":
            float(
                expected_goals[
                    "team_b_expected_goals"
                ]
            ),

        "team_a_win_probability":
            float(
                final_probabilities[
                    "team_a_win_probability"
                ]
            ),

        "draw_probability":
            float(
                final_probabilities[
                    "draw_probability"
                ]
            ),

        "team_b_win_probability":
            float(
                final_probabilities[
                    "team_b_win_probability"
                ]
            ),

        "score_matrix":
            final_score_matrix,

        "cdf":
            cumulative_probabilities,
    }


def reverse_cached_distribution(
    distribution: dict,
) -> dict:
    """
    Reverse Team A and Team B without calling the models again.
    """

    reversed_matrix = (
        distribution[
            "score_matrix"
        ].T.copy()
    )

    reversed_cdf = np.cumsum(
        reversed_matrix.flatten()
    )

    reversed_cdf[-1] = 1.0

    return {
        "team_a":
            distribution["team_b"],

        "team_b":
            distribution["team_a"],

        "venue_type": (
            "team_b_home"
            if distribution["venue_type"]
            ==
            "team_a_home"
            else
            "team_a_home"
            if distribution["venue_type"]
            ==
            "team_b_home"
            else
            "neutral"
        ),

        "team_a_expected_goals":
            distribution[
                "team_b_expected_goals"
            ],

        "team_b_expected_goals":
            distribution[
                "team_a_expected_goals"
            ],

        "team_a_win_probability":
            distribution[
                "team_b_win_probability"
            ],

        "draw_probability":
            distribution[
                "draw_probability"
            ],

        "team_b_win_probability":
            distribution[
                "team_a_win_probability"
            ],

        "score_matrix":
            reversed_matrix,

        "cdf":
            reversed_cdf,
    }


def precompute_all_match_distributions() -> dict:
    """
    Precompute neutral distributions for every pair of teams and
    host-home distributions for Canada, Mexico and United States.
    """

    all_teams = sorted(
        DISPLAY_TO_MODEL.keys()
    )

    cache = {}

    total_neutral_pairs = (
        len(all_teams)
        *
        (
            len(all_teams)
            -
            1
        )
        //
        2
    )

    print("\n" + "=" * 100)
    print("PRECOMPUTING MATCH DISTRIBUTIONS")
    print("=" * 100)

    print(
        "\nNeutral pair distributions:",
        total_neutral_pairs,
    )

    neutral_counter = 0

    for team_index, team_a in enumerate(
        all_teams
    ):
        for team_b in all_teams[
            team_index + 1:
        ]:
            distribution = (
                build_cached_distribution(
                    team_a_display=team_a,
                    team_b_display=team_b,
                    venue_type="neutral",
                )
            )

            cache[
                (
                    team_a,
                    team_b,
                    "neutral",
                )
            ] = distribution

            cache[
                (
                    team_b,
                    team_a,
                    "neutral",
                )
            ] = reverse_cached_distribution(
                distribution
            )

            neutral_counter += 1

            if (
                neutral_counter
                %
                150
                ==
                0
                or
                neutral_counter
                ==
                total_neutral_pairs
            ):
                print(
                    f"Neutral distributions prepared: "
                    f"{neutral_counter}/"
                    f"{total_neutral_pairs}"
                )

    host_teams = [
        "Canada",
        "Mexico",
        "United States",
    ]

    total_host_pairs = (
        len(host_teams)
        *
        (
            len(all_teams)
            -
            1
        )
    )

    print(
        "\nHost-home distributions:",
        total_host_pairs,
    )

    host_counter = 0

    for host_team in host_teams:
        for opponent in all_teams:
            if opponent == host_team:
                continue

            distribution = (
                build_cached_distribution(
                    team_a_display=
                        host_team,
                    team_b_display=
                        opponent,
                    venue_type=
                        "team_a_home",
                )
            )

            cache[
                (
                    host_team,
                    opponent,
                    "team_a_home",
                )
            ] = distribution

            cache[
                (
                    opponent,
                    host_team,
                    "team_b_home",
                )
            ] = reverse_cached_distribution(
                distribution
            )

            host_counter += 1

            if (
                host_counter
                %
                30
                ==
                0
                or
                host_counter
                ==
                total_host_pairs
            ):
                print(
                    f"Host distributions prepared: "
                    f"{host_counter}/"
                    f"{total_host_pairs}"
                )

    print(
        "\nCached ordered distributions:",
        len(cache),
    )

    return cache


def get_cached_distribution(
    cache: dict,
    team_a_display: str,
    team_b_display: str,
    venue_country: str,
) -> dict:
    venue_type = determine_venue_type(
        team_a=team_a_display,
        team_b=team_b_display,
        venue_country=venue_country,
    )

    cache_key = (
        team_a_display,
        team_b_display,
        venue_type,
    )

    if cache_key not in cache:
        raise KeyError(
            "Distribution cache is missing:\n"
            f"{cache_key}"
        )

    return cache[
        cache_key
    ]


def sample_cached_distribution(
    distribution: dict,
    random_generator,
) -> dict:
    """
    Sample one scoreline from a cached cumulative distribution.
    """

    random_value = (
        random_generator.random()
    )

    selected_position = int(
        np.searchsorted(
            distribution["cdf"],
            random_value,
            side="right",
        )
    )

    score_dimension = (
        MAX_GOALS + 1
    )

    team_a_goals = (
        selected_position
        //
        score_dimension
    )

    team_b_goals = (
        selected_position
        %
        score_dimension
    )

    return {
        "team_a_expected_goals":
            distribution[
                "team_a_expected_goals"
            ],

        "team_b_expected_goals":
            distribution[
                "team_b_expected_goals"
            ],

        "team_a_win_probability":
            distribution[
                "team_a_win_probability"
            ],

        "draw_probability":
            distribution[
                "draw_probability"
            ],

        "team_b_win_probability":
            distribution[
                "team_b_win_probability"
            ],

        "team_a_goals":
            int(
                team_a_goals
            ),

        "team_b_goals":
            int(
                team_b_goals
            ),
    }


# =============================================================================
# 24. ANNEX C MAPPING WITH BROWSER USER AGENT
# =============================================================================

def parse_annex_tables(
    tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Parse a list of HTML tables into the 495-row Annex C mapping.
    """

    selected_table = None
    flattened_columns = None

    for table in tables:
        column_names = [
            flatten_column_name(
                column
            )
            for column in table.columns
        ]

        compact_names = [
            compact_column_name(
                column
            )
            for column in column_names
        ]

        if (
            any(
                "1A"
                in column
                for column in compact_names
            )
            and
            any(
                "1L"
                in column
                for column in compact_names
            )
        ):
            selected_table = (
                table.copy()
            )

            flattened_columns = (
                column_names
            )

            break

    if selected_table is None:
        raise RuntimeError(
            "The Annex C table was not found in the downloaded HTML."
        )

    selected_table.columns = (
        flattened_columns
    )

    winner_column_lookup = {}

    for winner_slot in (
        THIRD_PLACE_WINNER_COLUMNS
    ):
        matches = [
            column
            for column
            in selected_table.columns
            if winner_slot
            in compact_column_name(
                column
            )
        ]

        if not matches:
            raise RuntimeError(
                f"Annex C column "
                f"{winner_slot} was not found."
            )

        winner_column_lookup[
            winner_slot
        ] = matches[0]

    used_columns = set(
        winner_column_lookup.values()
    )

    best_advancing_column = None
    best_valid_count = -1

    for column in selected_table.columns:
        if column in used_columns:
            continue

        valid_count = sum(
            normalize_advancing_group_set(
                value
            )
            is not None
            for value
            in selected_table[column]
        )

        if valid_count > best_valid_count:
            best_valid_count = (
                valid_count
            )

            best_advancing_column = (
                column
            )

    mapping_rows = []

    for _, row in (
        selected_table.iterrows()
    ):
        group_set = (
            normalize_advancing_group_set(
                row[
                    best_advancing_column
                ]
            )
        )

        if group_set is None:
            continue

        output_row = {
            "GroupSet": group_set
        }

        valid_row = True

        for winner_slot, source_column in (
            winner_column_lookup.items()
        ):
            assigned_group = (
                normalize_third_slot(
                    row[
                        source_column
                    ]
                )
            )

            if assigned_group is None:
                valid_row = False
                break

            output_row[
                winner_slot
            ] = assigned_group

        if valid_row:
            mapping_rows.append(
                output_row
            )

    mapping_df = (
        pd.DataFrame(
            mapping_rows
        )
        .drop_duplicates(
            subset=[
                "GroupSet"
            ]
        )
        .sort_values(
            "GroupSet"
        )
        .reset_index(
            drop=True
        )
    )

    if len(mapping_df) != 495:
        raise RuntimeError(
            "Annex C parsing did not produce 495 combinations. "
            f"Produced: {len(mapping_df)}"
        )

    validate_annex_mapping(
        mapping_df
    )

    return mapping_df


def load_annex_mapping_for_monte_carlo() -> tuple[dict, str]:
    """
    Load the cached official table or retry the download using a
    browser User-Agent. If unavailable, use deterministic valid
    assignments during simulations.
    """

    if ANNEX_C_CACHE_PATH.exists():
        try:
            mapping_df = pd.read_csv(
                ANNEX_C_CACHE_PATH,
                dtype=str,
            )

            validate_annex_mapping(
                mapping_df
            )

            lookup = {
                str(
                    row["GroupSet"]
                ): {
                    winner_slot: str(
                        row[
                            winner_slot
                        ]
                    )
                    for winner_slot
                    in THIRD_PLACE_WINNER_COLUMNS
                }
                for _, row
                in mapping_df.iterrows()
            }

            print(
                "\nAnnex C source: local official cache"
            )

            return (
                lookup,
                "Official Annex C cache",
            )

        except Exception as error:
            print(
                "\nExisting Annex C cache was invalid:"
            )

            print(
                error
            )

    try:
        from io import StringIO
        from urllib.request import (
            Request,
            urlopen,
        )

        request = Request(
            ANNEX_C_TABLE_URL,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                ),
                "Accept-Language":
                    "en-US,en;q=0.9",
            },
        )

        with urlopen(
            request,
            timeout=30,
        ) as response:
            html_text = (
                response.read()
                .decode(
                    "utf-8",
                    errors="replace",
                )
            )

        mapping_df = parse_annex_tables(
            pd.read_html(
                StringIO(
                    html_text
                )
            )
        )

        mapping_df.to_csv(
            ANNEX_C_CACHE_PATH,
            index=False,
        )

        lookup = {
            str(
                row["GroupSet"]
            ): {
                winner_slot: str(
                    row[
                        winner_slot
                    ]
                )
                for winner_slot
                in THIRD_PLACE_WINNER_COLUMNS
            }
            for _, row
            in mapping_df.iterrows()
        }

        print(
            "\nAnnex C source: official table downloaded "
            "with browser User-Agent"
        )

        return (
            lookup,
            "Official Annex C downloaded table",
        )

    except Exception as error:
        print(
            "\nWARNING: Official Annex C mapping remains unavailable."
        )

        print(
            f"Reason: {error}"
        )

        print(
            "The Monte Carlo engine will use a valid deterministic "
            "fallback matching for each third-place group combination."
        )

        return (
            {},
            "Fallback valid assignment",
        )


def get_monte_carlo_third_assignment(
    qualified_group_set: str,
    official_lookup: dict,
    fallback_cache: dict,
) -> dict:
    if (
        qualified_group_set
        in official_lookup
    ):
        return official_lookup[
            qualified_group_set
        ]

    if (
        qualified_group_set
        not in fallback_cache
    ):
        fallback_cache[
            qualified_group_set
        ] = (
            solve_fallback_third_assignment(
                set(
                    qualified_group_set
                )
            )
        )

    return fallback_cache[
        qualified_group_set
    ]


# =============================================================================
# 25. FAST GROUP-STAGE SIMULATION
# =============================================================================

def simulate_group_stage_fast(
    distribution_cache: dict,
    random_generator,
) -> dict:
    group_tables = {
        group_name:
            create_empty_group_table(
                group_name
            )
        for group_name in GROUPS
    }

    group_matches = {
        group_name: []
        for group_name in GROUPS
    }

    for fixture in GROUP_FIXTURES:
        (
            match_number,
            match_date,
            group_name,
            team_a_display,
            team_b_display,
            stadium,
            venue_country,
        ) = fixture

        distribution = (
            get_cached_distribution(
                cache=
                    distribution_cache,
                team_a_display=
                    team_a_display,
                team_b_display=
                    team_b_display,
                venue_country=
                    venue_country,
            )
        )

        simulation = (
            sample_cached_distribution(
                distribution=
                    distribution,
                random_generator=
                    random_generator,
            )
        )

        goals_a = int(
            simulation[
                "team_a_goals"
            ]
        )

        goals_b = int(
            simulation[
                "team_b_goals"
            ]
        )

        update_group_table(
            table=group_tables[
                group_name
            ],
            team_a=team_a_display,
            team_b=team_b_display,
            goals_a=goals_a,
            goals_b=goals_b,
        )

        group_matches[
            group_name
        ].append({
            "team_a":
                team_a_display,
            "team_b":
                team_b_display,
            "team_a_goals":
                goals_a,
            "team_b_goals":
                goals_b,
        })

    ranked_groups = {
        group_name: rank_group(
            table=group_tables[
                group_name
            ],
            group_matches=group_matches[
                group_name
            ],
        )
        for group_name in GROUPS
    }

    third_rows = []

    for group_name, ranked_teams in (
        ranked_groups.items()
    ):
        third_team = ranked_teams[2]

        statistics = group_tables[
            group_name
        ][
            third_team
        ]

        third_rows.append({
            "Group": group_name,
            "Team": third_team,
            "Pts":
                statistics["Pts"],
            "GD":
                statistics["GD"],
            "GF":
                statistics["GF"],
            "Elo":
                statistics["Elo"],
        })

    third_rows = sorted(
        third_rows,
        key=lambda row: (
            row["Pts"],
            row["GD"],
            row["GF"],
            row["Elo"],
        ),
        reverse=True,
    )

    qualified_thirds = (
        third_rows[:8]
    )

    third_team_by_group = {
        row["Group"]:
            row["Team"]
        for row in qualified_thirds
    }

    qualified_third_group_set = "".join(
        sorted(
            third_team_by_group.keys()
        )
    )

    group_positions = {}
    qualifiers = []
    qualification_routes = {}

    for group_name, ranked_teams in (
        ranked_groups.items()
    ):
        for rank_position, team in enumerate(
            ranked_teams,
            start=1,
        ):
            group_positions[
                f"{rank_position}{group_name}"
            ] = team

            if rank_position <= 2:
                qualifiers.append(
                    team
                )

                qualification_routes[
                    team
                ] = (
                    "Group winner"
                    if rank_position == 1
                    else
                    "Group runner-up"
                )

            elif (
                rank_position == 3
                and
                group_name
                in third_team_by_group
            ):
                qualifiers.append(
                    team
                )

                qualification_routes[
                    team
                ] = (
                    "Best third"
                )

    if len(qualifiers) != 32:
        raise ValueError(
            "Fast group simulation did not produce "
            "32 qualified teams."
        )

    return {
        "group_tables":
            group_tables,

        "ranked_groups":
            ranked_groups,

        "group_positions":
            group_positions,

        "qualifiers":
            qualifiers,

        "qualification_routes":
            qualification_routes,

        "third_team_by_group":
            third_team_by_group,

        "qualified_third_group_set":
            qualified_third_group_set,
    }


# =============================================================================
# 26. FAST KNOCKOUT SIMULATION
# =============================================================================

def simulate_knockout_round_fast(
    match_definitions: list[tuple],
    slot_teams: dict,
    distribution_cache: dict,
    random_generator,
) -> tuple[list[dict], dict]:
    result_rows = []
    new_slots = {}

    for match_definition in (
        match_definitions
    ):
        (
            match_number,
            match_date,
            source_a,
            source_b,
            stadium,
            venue_country,
        ) = match_definition

        team_a_display = slot_teams[
            source_a
        ]

        team_b_display = slot_teams[
            source_b
        ]

        distribution = (
            get_cached_distribution(
                cache=
                    distribution_cache,
                team_a_display=
                    team_a_display,
                team_b_display=
                    team_b_display,
                venue_country=
                    venue_country,
            )
        )

        simulation = (
            sample_cached_distribution(
                distribution=
                    distribution,
                random_generator=
                    random_generator,
            )
        )

        resolution_seed = int(
            random_generator.integers(
                0,
                np.iinfo(
                    np.int32
                ).max,
            )
        )

        resolution = resolve_knockout_match(
            simulation=simulation,
            team_a_display=
                team_a_display,
            team_b_display=
                team_b_display,
            resolution_seed=
                resolution_seed,
        )

        winner = resolution[
            "winner"
        ]

        loser = resolution[
            "loser"
        ]

        new_slots[
            f"W{match_number}"
        ] = winner

        new_slots[
            f"L{match_number}"
        ] = loser

        result_rows.append({
            "match_number":
                match_number,

            "team_a":
                team_a_display,

            "team_b":
                team_b_display,

            "winner":
                winner,

            "loser":
                loser,

            "decision":
                resolution[
                    "decision"
                ],
        })

    return (
        result_rows,
        new_slots,
    )


def simulate_one_complete_tournament_fast(
    distribution_cache: dict,
    official_annex_lookup: dict,
    fallback_annex_cache: dict,
    simulation_seed: int,
) -> dict:
    random_generator = np.random.default_rng(
        simulation_seed
    )

    group_results = (
        simulate_group_stage_fast(
            distribution_cache=
                distribution_cache,
            random_generator=
                random_generator,
        )
    )

    third_assignment = (
        get_monte_carlo_third_assignment(
            qualified_group_set=
                group_results[
                    "qualified_third_group_set"
                ],
            official_lookup=
                official_annex_lookup,
            fallback_cache=
                fallback_annex_cache,
        )
    )

    slot_teams = dict(
        group_results[
            "group_positions"
        ]
    )

    for winner_slot in (
        THIRD_PLACE_WINNER_COLUMNS
    ):
        assigned_group = (
            third_assignment[
                winner_slot
            ]
        )

        slot_teams[
            f"THIRD_{winner_slot}"
        ] = (
            group_results[
                "third_team_by_group"
            ][
                assigned_group
            ]
        )

    (
        round_of_32_results,
        round_of_32_slots,
    ) = simulate_knockout_round_fast(
        match_definitions=
            ROUND_OF_32_MATCHES,
        slot_teams=slot_teams,
        distribution_cache=
            distribution_cache,
        random_generator=
            random_generator,
    )

    slot_teams.update(
        round_of_32_slots
    )

    (
        round_of_16_results,
        round_of_16_slots,
    ) = simulate_knockout_round_fast(
        match_definitions=
            ROUND_OF_16_MATCHES,
        slot_teams=slot_teams,
        distribution_cache=
            distribution_cache,
        random_generator=
            random_generator,
    )

    slot_teams.update(
        round_of_16_slots
    )

    (
        quarter_final_results,
        quarter_final_slots,
    ) = simulate_knockout_round_fast(
        match_definitions=
            QUARTER_FINAL_MATCHES,
        slot_teams=slot_teams,
        distribution_cache=
            distribution_cache,
        random_generator=
            random_generator,
    )

    slot_teams.update(
        quarter_final_slots
    )

    (
        semi_final_results,
        semi_final_slots,
    ) = simulate_knockout_round_fast(
        match_definitions=
            SEMI_FINAL_MATCHES,
        slot_teams=slot_teams,
        distribution_cache=
            distribution_cache,
        random_generator=
            random_generator,
    )

    slot_teams.update(
        semi_final_slots
    )

    (
        third_place_results,
        third_place_slots,
    ) = simulate_knockout_round_fast(
        match_definitions=
            THIRD_PLACE_MATCHES,
        slot_teams=slot_teams,
        distribution_cache=
            distribution_cache,
        random_generator=
            random_generator,
    )

    slot_teams.update(
        third_place_slots
    )

    (
        final_results,
        final_slots,
    ) = simulate_knockout_round_fast(
        match_definitions=
            FINAL_MATCHES,
        slot_teams=slot_teams,
        distribution_cache=
            distribution_cache,
        random_generator=
            random_generator,
    )

    slot_teams.update(
        final_slots
    )

    round_of_32_winners = [
        row["winner"]
        for row in round_of_32_results
    ]

    round_of_16_winners = [
        row["winner"]
        for row in round_of_16_results
    ]

    quarter_final_winners = [
        row["winner"]
        for row in quarter_final_results
    ]

    semi_final_winners = [
        row["winner"]
        for row in semi_final_results
    ]

    final_row = final_results[0]
    third_place_row = (
        third_place_results[0]
    )

    return {
        "group_results":
            group_results,

        "round_of_32_teams":
            list(
                group_results[
                    "qualifiers"
                ]
            ),

        # Teams reaching the Round of 16:
        "round_of_16_teams":
            round_of_32_winners,

        # Teams reaching the quarter-finals:
        "quarter_final_teams":
            round_of_16_winners,

        # Teams reaching the semi-finals:
        "semi_final_teams":
            quarter_final_winners,

        # Teams reaching the Final:
        "final_teams":
            semi_final_winners,

        "champion":
            final_row["winner"],

        "runner_up":
            final_row["loser"],

        "third_place":
            third_place_row[
                "winner"
            ],

        "fourth_place":
            third_place_row[
                "loser"
            ],

        "final_matchup": tuple(
            sorted(
                [
                    final_row["team_a"],
                    final_row["team_b"],
                ]
            )
        ),

        "final_ordered_pair": (
            final_row["winner"],
            final_row["loser"],
        ),
    }


# =============================================================================
# 27. MONTE CARLO AGGREGATION
# =============================================================================

def create_empty_team_counter() -> dict:
    return {
        team: 0
        for team in DISPLAY_TO_MODEL
    }


def create_empty_team_float_counter() -> dict:
    return {
        team: 0.0
        for team in DISPLAY_TO_MODEL
    }


def wilson_normal_interval(
    count: int,
    total: int,
) -> tuple[float, float]:
    """
    Simple 95% normal approximation, returned as percentages.
    """

    if total <= 0:
        return (
            0.0,
            0.0,
        )

    probability = (
        count
        /
        total
    )

    standard_error = math.sqrt(
        probability
        *
        (
            1.0
            -
            probability
        )
        /
        total
    )

    lower = max(
        0.0,
        probability
        -
        1.96
        *
        standard_error,
    )

    upper = min(
        1.0,
        probability
        +
        1.96
        *
        standard_error,
    )

    return (
        100.0
        *
        lower,
        100.0
        *
        upper,
    )


def run_monte_carlo_final_product(
    number_of_simulations: int =
        N_SIMULATIONS,
    monte_carlo_seed: int =
        MONTE_CARLO_SEED,
) -> dict:
    if number_of_simulations < 1:
        raise ValueError(
            "N_SIMULATIONS must be at least 1."
        )

    validate_tournament_configuration()

    (
        official_annex_lookup,
        annex_source,
    ) = load_annex_mapping_for_monte_carlo()

    fallback_annex_cache = {}

    distribution_cache = (
        precompute_all_match_distributions()
    )

    all_teams = sorted(
        DISPLAY_TO_MODEL.keys()
    )

    team_group = {
        team: group_name
        for group_name, teams
        in GROUPS.items()
        for team in teams
    }

    stage_counts = {
        "RoundOf32":
            create_empty_team_counter(),

        "RoundOf16":
            create_empty_team_counter(),

        "QuarterFinal":
            create_empty_team_counter(),

        "SemiFinal":
            create_empty_team_counter(),

        "Final":
            create_empty_team_counter(),

        "Champion":
            create_empty_team_counter(),

        "RunnerUp":
            create_empty_team_counter(),

        "ThirdPlace":
            create_empty_team_counter(),

        "FourthPlace":
            create_empty_team_counter(),
    }

    group_winner_counts = (
        create_empty_team_counter()
    )

    group_runner_up_counts = (
        create_empty_team_counter()
    )

    best_third_counts = (
        create_empty_team_counter()
    )

    group_rank_sums = (
        create_empty_team_float_counter()
    )

    group_points_sums = (
        create_empty_team_float_counter()
    )

    group_gf_sums = (
        create_empty_team_float_counter()
    )

    group_ga_sums = (
        create_empty_team_float_counter()
    )

    group_gd_sums = (
        create_empty_team_float_counter()
    )

    final_matchup_counts = (
        defaultdict(int)
    )

    ordered_final_counts = (
        defaultdict(int)
    )

    champion_first_half = (
        create_empty_team_counter()
    )

    champion_second_half = (
        create_empty_team_counter()
    )

    first_half_total = (
        number_of_simulations
        //
        2
    )

    second_half_total = (
        number_of_simulations
        -
        first_half_total
    )

    master_seed_sequence = (
        np.random.SeedSequence(
            monte_carlo_seed
        )
    )

    child_seed_sequences = (
        master_seed_sequence.spawn(
            number_of_simulations
        )
    )

    print("\n" + "=" * 100)
    print("WORLD CUP 2026 MONTE CARLO FINAL PRODUCT")
    print("=" * 100)

    print(
        "\nSimulations:",
        f"{number_of_simulations:,}",
    )

    print(
        "Random Forest weight:",
        f"{CLASSIFIER_WEIGHT:.0%}",
    )

    print(
        "Poisson weight:",
        f"{POISSON_WEIGHT:.0%}",
    )

    print(
        "Third-place allocation source:",
        annex_source,
    )

    for simulation_index, seed_sequence in enumerate(
        child_seed_sequences,
        start=1,
    ):
        simulation_seed = int(
            seed_sequence.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )

        tournament = (
            simulate_one_complete_tournament_fast(
                distribution_cache=
                    distribution_cache,
                official_annex_lookup=
                    official_annex_lookup,
                fallback_annex_cache=
                    fallback_annex_cache,
                simulation_seed=
                    simulation_seed,
            )
        )

        group_results = (
            tournament[
                "group_results"
            ]
        )

        for group_name, ranked_teams in (
            group_results[
                "ranked_groups"
            ].items()
        ):
            for rank_position, team in enumerate(
                ranked_teams,
                start=1,
            ):
                group_rank_sums[
                    team
                ] += rank_position

                statistics = (
                    group_results[
                        "group_tables"
                    ][
                        group_name
                    ][
                        team
                    ]
                )

                group_points_sums[
                    team
                ] += statistics[
                    "Pts"
                ]

                group_gf_sums[
                    team
                ] += statistics[
                    "GF"
                ]

                group_ga_sums[
                    team
                ] += statistics[
                    "GA"
                ]

                group_gd_sums[
                    team
                ] += statistics[
                    "GD"
                ]

                if rank_position == 1:
                    group_winner_counts[
                        team
                    ] += 1

                elif rank_position == 2:
                    group_runner_up_counts[
                        team
                    ] += 1

        for team, route in (
            group_results[
                "qualification_routes"
            ].items()
        ):
            if route == "Best third":
                best_third_counts[
                    team
                ] += 1

        for team in tournament[
            "round_of_32_teams"
        ]:
            stage_counts[
                "RoundOf32"
            ][
                team
            ] += 1

        for team in tournament[
            "round_of_16_teams"
        ]:
            stage_counts[
                "RoundOf16"
            ][
                team
            ] += 1

        for team in tournament[
            "quarter_final_teams"
        ]:
            stage_counts[
                "QuarterFinal"
            ][
                team
            ] += 1

        for team in tournament[
            "semi_final_teams"
        ]:
            stage_counts[
                "SemiFinal"
            ][
                team
            ] += 1

        for team in tournament[
            "final_teams"
        ]:
            stage_counts[
                "Final"
            ][
                team
            ] += 1

        champion = tournament[
            "champion"
        ]

        runner_up = tournament[
            "runner_up"
        ]

        third_place = tournament[
            "third_place"
        ]

        fourth_place = tournament[
            "fourth_place"
        ]

        stage_counts[
            "Champion"
        ][
            champion
        ] += 1

        stage_counts[
            "RunnerUp"
        ][
            runner_up
        ] += 1

        stage_counts[
            "ThirdPlace"
        ][
            third_place
        ] += 1

        stage_counts[
            "FourthPlace"
        ][
            fourth_place
        ] += 1

        final_matchup_counts[
            tournament[
                "final_matchup"
            ]
        ] += 1

        ordered_final_counts[
            tournament[
                "final_ordered_pair"
            ]
        ] += 1

        if (
            simulation_index
            <=
            first_half_total
        ):
            champion_first_half[
                champion
            ] += 1

        else:
            champion_second_half[
                champion
            ] += 1

        if (
            simulation_index
            %
            PROGRESS_INTERVAL
            ==
            0
            or
            simulation_index
            ==
            number_of_simulations
        ):
            current_leaders = sorted(
                stage_counts[
                    "Champion"
                ].items(),
                key=lambda item:
                    item[1],
                reverse=True,
            )[:5]

            leader_text = ", ".join(
                (
                    f"{team} "
                    f"{100.0 * count / simulation_index:.2f}%"
                )
                for team, count
                in current_leaders
            )

            print(
                f"Completed "
                f"{simulation_index:,}/"
                f"{number_of_simulations:,} "
                f"| Current leaders: "
                f"{leader_text}"
            )

    probability_rows = []

    for team in all_teams:
        champion_count = (
            stage_counts[
                "Champion"
            ][
                team
            ]
        )

        (
            champion_ci_lower,
            champion_ci_upper,
        ) = wilson_normal_interval(
            count=champion_count,
            total=number_of_simulations,
        )

        probability_rows.append({
            "Team": team,
            "Group":
                team_group[
                    team
                ],

            "GroupWinnerProbability":
                100.0
                *
                group_winner_counts[
                    team
                ]
                /
                number_of_simulations,

            "GroupRunnerUpProbability":
                100.0
                *
                group_runner_up_counts[
                    team
                ]
                /
                number_of_simulations,

            "BestThirdQualificationProbability":
                100.0
                *
                best_third_counts[
                    team
                ]
                /
                number_of_simulations,

            "RoundOf32Probability":
                100.0
                *
                stage_counts[
                    "RoundOf32"
                ][
                    team
                ]
                /
                number_of_simulations,

            "RoundOf16Probability":
                100.0
                *
                stage_counts[
                    "RoundOf16"
                ][
                    team
                ]
                /
                number_of_simulations,

            "QuarterFinalProbability":
                100.0
                *
                stage_counts[
                    "QuarterFinal"
                ][
                    team
                ]
                /
                number_of_simulations,

            "SemiFinalProbability":
                100.0
                *
                stage_counts[
                    "SemiFinal"
                ][
                    team
                ]
                /
                number_of_simulations,

            "FinalProbability":
                100.0
                *
                stage_counts[
                    "Final"
                ][
                    team
                ]
                /
                number_of_simulations,

            "ChampionProbability":
                100.0
                *
                champion_count
                /
                number_of_simulations,

            "Champion95CILower":
                champion_ci_lower,

            "Champion95CIUpper":
                champion_ci_upper,

            "RunnerUpProbability":
                100.0
                *
                stage_counts[
                    "RunnerUp"
                ][
                    team
                ]
                /
                number_of_simulations,

            "ThirdPlaceProbability":
                100.0
                *
                stage_counts[
                    "ThirdPlace"
                ][
                    team
                ]
                /
                number_of_simulations,

            "FourthPlaceProbability":
                100.0
                *
                stage_counts[
                    "FourthPlace"
                ][
                    team
                ]
                /
                number_of_simulations,

            "AverageGroupRank":
                group_rank_sums[
                    team
                ]
                /
                number_of_simulations,

            "AverageGroupPoints":
                group_points_sums[
                    team
                ]
                /
                number_of_simulations,

            "AverageGroupGoalsFor":
                group_gf_sums[
                    team
                ]
                /
                number_of_simulations,

            "AverageGroupGoalsAgainst":
                group_ga_sums[
                    team
                ]
                /
                number_of_simulations,

            "AverageGroupGoalDifference":
                group_gd_sums[
                    team
                ]
                /
                number_of_simulations,
        })

    probabilities_df = (
        pd.DataFrame(
            probability_rows
        )
        .sort_values(
            [
                "ChampionProbability",
                "FinalProbability",
                "SemiFinalProbability",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    probabilities_df.insert(
        0,
        "ChampionRank",
        np.arange(
            1,
            len(
                probabilities_df
            )
            +
            1,
        ),
    )

    final_matchup_rows = []

    for (
        team_a,
        team_b,
    ), count in sorted(
        final_matchup_counts.items(),
        key=lambda item:
            item[1],
        reverse=True,
    ):
        final_matchup_rows.append({
            "TeamA": team_a,
            "TeamB": team_b,
            "FinalCount": count,
            "FinalMatchupProbability":
                100.0
                *
                count
                /
                number_of_simulations,
        })

    final_matchups_df = pd.DataFrame(
        final_matchup_rows
    )

    ordered_final_rows = []

    for (
        champion,
        runner_up,
    ), count in sorted(
        ordered_final_counts.items(),
        key=lambda item:
            item[1],
        reverse=True,
    ):
        ordered_final_rows.append({
            "Champion": champion,
            "RunnerUp": runner_up,
            "Count": count,
            "Probability":
                100.0
                *
                count
                /
                number_of_simulations,
        })

    ordered_finals_df = pd.DataFrame(
        ordered_final_rows
    )

    convergence_rows = []

    for team in all_teams:
        first_probability = (
            100.0
            *
            champion_first_half[
                team
            ]
            /
            max(
                first_half_total,
                1,
            )
        )

        second_probability = (
            100.0
            *
            champion_second_half[
                team
            ]
            /
            max(
                second_half_total,
                1,
            )
        )

        convergence_rows.append({
            "Team": team,
            "FirstHalfChampionProbability":
                first_probability,
            "SecondHalfChampionProbability":
                second_probability,
            "AbsoluteDifference":
                abs(
                    first_probability
                    -
                    second_probability
                ),
        })

    convergence_df = (
        pd.DataFrame(
            convergence_rows
        )
        .sort_values(
            "AbsoluteDifference",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return {
        "probabilities":
            probabilities_df,

        "final_matchups":
            final_matchups_df,

        "ordered_finals":
            ordered_finals_df,

        "convergence":
            convergence_df,

        "number_of_simulations":
            number_of_simulations,

        "annex_source":
            annex_source,

        "fallback_combination_count":
            len(
                fallback_annex_cache
            ),
    }


# =============================================================================
# 28. FINAL OUTPUT FILES AND VISUALS
# =============================================================================

def save_champion_probability_chart(
    probabilities_df: pd.DataFrame,
) -> None:
    chart_df = (
        probabilities_df
        .head(20)
        .sort_values(
            "ChampionProbability",
            ascending=True,
        )
    )

    fig, axis = plt.subplots(
        figsize=(12, 9)
    )

    axis.barh(
        chart_df["Team"],
        chart_df[
            "ChampionProbability"
        ],
    )

    axis.set_xlabel(
        "Championship probability (%)"
    )

    axis.set_ylabel(
        "Team"
    )

    axis.set_title(
        (
            "FIFA World Cup 2026 "
            "— Monte Carlo Championship Probabilities"
        )
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        MONTE_CARLO_IMAGE_FOLDER
        /
        "championship_probabilities_top20.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_stage_probability_chart(
    probabilities_df: pd.DataFrame,
) -> None:
    chart_df = (
        probabilities_df
        .head(16)
        .copy()
    )

    stage_columns = [
        "RoundOf32Probability",
        "QuarterFinalProbability",
        "SemiFinalProbability",
        "FinalProbability",
        "ChampionProbability",
    ]

    x_positions = np.arange(
        len(
            chart_df
        )
    )

    bottom_values = np.zeros(
        len(
            chart_df
        )
    )

    # Plot cumulative-looking stage layers as separate bars.
    # The underlying CSV remains the precise source of truth.
    fig, axis = plt.subplots(
        figsize=(16, 8)
    )

    bar_width = 0.15

    for column_index, column in enumerate(
        stage_columns
    ):
        axis.bar(
            x_positions
            +
            (
                column_index
                -
                2
            )
            *
            bar_width,
            chart_df[
                column
            ],
            width=bar_width,
            label=column.replace(
                "Probability",
                "",
            ),
        )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        chart_df[
            "Team"
        ],
        rotation=45,
        ha="right",
    )

    axis.set_ylabel(
        "Probability (%)"
    )

    axis.set_title(
        (
            "FIFA World Cup 2026 "
            "— Stage-Reach Probabilities, Top 16"
        )
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        MONTE_CARLO_IMAGE_FOLDER
        /
        "stage_reach_probabilities_top16.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_group_probability_images(
    probabilities_df: pd.DataFrame,
) -> None:
    selected_columns = [
        "Team",
        "GroupWinnerProbability",
        "GroupRunnerUpProbability",
        "BestThirdQualificationProbability",
        "RoundOf32Probability",
        "AverageGroupRank",
        "AverageGroupPoints",
        "AverageGroupGoalDifference",
    ]

    renamed_columns = {
        "GroupWinnerProbability":
            "Winner %",
        "GroupRunnerUpProbability":
            "Runner-up %",
        "BestThirdQualificationProbability":
            "Best 3rd %",
        "RoundOf32Probability":
            "Qualify %",
        "AverageGroupRank":
            "Avg Rank",
        "AverageGroupPoints":
            "Avg Pts",
        "AverageGroupGoalDifference":
            "Avg GD",
    }

    for group_name in GROUPS:
        group_df = (
            probabilities_df[
                probabilities_df[
                    "Group"
                ]
                ==
                group_name
            ][
                selected_columns
            ]
            .copy()
            .sort_values(
                [
                    "RoundOf32Probability",
                    "GroupWinnerProbability",
                ],
                ascending=False,
            )
        )

        group_df = group_df.rename(
            columns=renamed_columns
        )

        numeric_columns = [
            column
            for column in group_df.columns
            if column != "Team"
        ]

        group_df[
            numeric_columns
        ] = group_df[
            numeric_columns
        ].round(2)

        save_dataframe_image(
            dataframe=group_df,
            title=(
                "FIFA World Cup 2026 Monte Carlo "
                f"— Group {group_name}"
            ),
            output_path=(
                MONTE_CARLO_GROUP_IMAGE_FOLDER
                /
                f"Group_{group_name}_probabilities.png"
            ),
            footer=(
                f"Based on "
                f"{N_SIMULATIONS:,} "
                "complete tournament simulations."
            ),
            figsize_width=15,
        )


def save_monte_carlo_final_outputs(
    monte_carlo_results: dict,
) -> None:
    probabilities_df = (
        monte_carlo_results[
            "probabilities"
        ]
    )

    final_matchups_df = (
        monte_carlo_results[
            "final_matchups"
        ]
    )

    ordered_finals_df = (
        monte_carlo_results[
            "ordered_finals"
        ]
    )

    convergence_df = (
        monte_carlo_results[
            "convergence"
        ]
    )

    probabilities_df.to_csv(
        MONTE_CARLO_OUTPUT_FOLDER
        /
        "team_stage_and_champion_probabilities.csv",
        index=False,
    )

    final_matchups_df.to_csv(
        MONTE_CARLO_OUTPUT_FOLDER
        /
        "most_common_final_matchups.csv",
        index=False,
    )

    ordered_finals_df.to_csv(
        MONTE_CARLO_OUTPUT_FOLDER
        /
        "most_common_champion_runner_up_pairs.csv",
        index=False,
    )

    convergence_df.to_csv(
        MONTE_CARLO_OUTPUT_FOLDER
        /
        "champion_probability_convergence.csv",
        index=False,
    )

    display_probability_df = (
        probabilities_df[
            [
                "ChampionRank",
                "Team",
                "Group",
                "RoundOf32Probability",
                "QuarterFinalProbability",
                "SemiFinalProbability",
                "FinalProbability",
                "ChampionProbability",
                "Champion95CILower",
                "Champion95CIUpper",
            ]
        ]
        .copy()
    )

    display_probability_df = (
        display_probability_df.rename(
            columns={
                "RoundOf32Probability":
                    "R32 %",
                "QuarterFinalProbability":
                    "QF %",
                "SemiFinalProbability":
                    "SF %",
                "FinalProbability":
                    "Final %",
                "ChampionProbability":
                    "Champion %",
                "Champion95CILower":
                    "95% CI Low",
                "Champion95CIUpper":
                    "95% CI High",
            }
        )
    )

    numeric_columns = [
        column
        for column
        in display_probability_df.columns
        if column not in {
            "ChampionRank",
            "Team",
            "Group",
        }
    ]

    display_probability_df[
        numeric_columns
    ] = display_probability_df[
        numeric_columns
    ].round(2)

    save_dataframe_image(
        dataframe=
            display_probability_df,
        title=(
            "FIFA World Cup 2026 "
            "— Final Monte Carlo Probability Ranking"
        ),
        output_path=(
            MONTE_CARLO_IMAGE_FOLDER
            /
            "all_teams_probability_table.png"
        ),
        footer=(
            f"Based on "
            f"{monte_carlo_results['number_of_simulations']:,} "
            "complete tournament simulations. "
            "Champion confidence intervals show approximate "
            "95% Monte Carlo sampling uncertainty."
        ),
        figsize_width=20,
    )

    top_final_matchups = (
        final_matchups_df
        .head(20)
        .copy()
    )

    if not top_final_matchups.empty:
        top_final_matchups[
            "FinalMatchup"
        ] = (
            top_final_matchups[
                "TeamA"
            ]
            +
            " vs "
            +
            top_final_matchups[
                "TeamB"
            ]
        )

        final_display = (
            top_final_matchups[
                [
                    "FinalMatchup",
                    "FinalCount",
                    "FinalMatchupProbability",
                ]
            ]
            .rename(
                columns={
                    "FinalCount":
                        "Count",
                    "FinalMatchupProbability":
                        "Probability %",
                }
            )
        )

        final_display[
            "Probability %"
        ] = final_display[
            "Probability %"
        ].round(2)

        save_dataframe_image(
            dataframe=
                final_display,
            title=(
                "FIFA World Cup 2026 "
                "— Most Common Final Matchups"
            ),
            output_path=(
                MONTE_CARLO_IMAGE_FOLDER
                /
                "most_common_final_matchups.png"
            ),
            footer=(
                f"Top final pairings across "
                f"{monte_carlo_results['number_of_simulations']:,} "
                "simulations."
            ),
            figsize_width=14,
        )

    save_champion_probability_chart(
        probabilities_df
    )

    save_stage_probability_chart(
        probabilities_df
    )

    save_group_probability_images(
        probabilities_df
    )

    top_team = (
        probabilities_df.iloc[0]
    )

    maximum_convergence_difference = float(
        convergence_df[
            "AbsoluteDifference"
        ].max()
    )

    with open(
        MONTE_CARLO_OUTPUT_FOLDER
        /
        "final_prediction_summary.txt",
        "w",
        encoding="utf-8",
    ) as summary_file:
        summary_file.write(
            "FIFA WORLD CUP 2026 MONTE CARLO FINAL PRODUCT\n"
        )

        summary_file.write(
            "=" * 70 + "\n\n"
        )

        summary_file.write(
            f"Complete tournament simulations: "
            f"{monte_carlo_results['number_of_simulations']:,}\n"
        )

        summary_file.write(
            f"Monte Carlo seed: "
            f"{MONTE_CARLO_SEED}\n"
        )

        summary_file.write(
            f"Random Forest weight: "
            f"{CLASSIFIER_WEIGHT:.4f}\n"
        )

        summary_file.write(
            f"Poisson weight: "
            f"{POISSON_WEIGHT:.4f}\n"
        )

        summary_file.write(
            f"Third-place assignment source: "
            f"{monte_carlo_results['annex_source']}\n"
        )

        summary_file.write(
            f"Fallback group-set mappings used: "
            f"{monte_carlo_results['fallback_combination_count']}\n\n"
        )

        summary_file.write(
            "FINAL MODEL FAVOURITE\n"
        )

        summary_file.write(
            f"Team: {top_team['Team']}\n"
        )

        summary_file.write(
            f"Championship probability: "
            f"{top_team['ChampionProbability']:.2f}%\n"
        )

        summary_file.write(
            f"Approximate 95% Monte Carlo CI: "
            f"{top_team['Champion95CILower']:.2f}% "
            f"to "
            f"{top_team['Champion95CIUpper']:.2f}%\n"
        )

        summary_file.write(
            f"Final probability: "
            f"{top_team['FinalProbability']:.2f}%\n"
        )

        summary_file.write(
            f"Semifinal probability: "
            f"{top_team['SemiFinalProbability']:.2f}%\n"
        )

        summary_file.write(
            f"Quarterfinal probability: "
            f"{top_team['QuarterFinalProbability']:.2f}%\n\n"
        )

        summary_file.write(
            "TOP 15 CHAMPIONSHIP PROBABILITIES\n"
        )

        for _, row in (
            probabilities_df
            .head(15)
            .iterrows()
        ):
            summary_file.write(
                f"{int(row['ChampionRank']):>2}. "
                f"{row['Team']}: "
                f"{row['ChampionProbability']:.2f}% "
                f"(Final {row['FinalProbability']:.2f}%)\n"
            )

        summary_file.write(
            "\nCONVERGENCE CHECK\n"
        )

        summary_file.write(
            "Maximum absolute difference between first-half "
            "and second-half championship probabilities: "
            f"{maximum_convergence_difference:.3f} percentage points\n"
        )

        summary_file.write(
            "\nInterpretation: these are model-estimated "
            "probabilities, not certainties.\n"
        )


# =============================================================================
# 29. RUN THE FINAL MONTE CARLO PRODUCT
# =============================================================================

if __name__ == "__main__":

    final_monte_carlo_results = (
        run_monte_carlo_final_product(
            number_of_simulations=
                N_SIMULATIONS,
            monte_carlo_seed=
                MONTE_CARLO_SEED,
        )
    )

    save_monte_carlo_final_outputs(
        final_monte_carlo_results
    )

    final_probabilities = (
        final_monte_carlo_results[
            "probabilities"
        ]
    )

    print("\n" + "=" * 100)
    print("FINAL MONTE CARLO PRODUCT COMPLETED")
    print("=" * 100)

    print(
        "\nTop 15 championship probabilities:"
    )

    print(
        final_probabilities[
            [
                "ChampionRank",
                "Team",
                "ChampionProbability",
                "FinalProbability",
                "SemiFinalProbability",
                "QuarterFinalProbability",
                "RoundOf32Probability",
            ]
        ]
        .head(15)
        .round(2)
        .to_string(
            index=False
        )
    )

    print(
        "\nAll final outputs saved to:"
    )

    print(
        MONTE_CARLO_OUTPUT_FOLDER
    )

    print(
        "\nMain probability CSV:"
    )

    print(
        MONTE_CARLO_OUTPUT_FOLDER
        /
        "team_stage_and_champion_probabilities.csv"
    )

    print(
        "\nMain probability image:"
    )

    print(
        MONTE_CARLO_IMAGE_FOLDER
        /
        "all_teams_probability_table.png"
    )
