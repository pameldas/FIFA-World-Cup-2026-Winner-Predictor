import json
import math
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, f1_score, log_loss


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

DATA_FOLDER = Path(r"D:\wc data")

INPUT_PATH = DATA_FOLDER / "ml_features.csv"

RESULTS_OUTPUT_PATH = (
    DATA_FOLDER / "hybrid_weight_comparison.csv"
)

BEST_WEIGHT_OUTPUT_PATH = (
    DATA_FOLDER / "best_hybrid_weights.json"
)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_STATE = 42
POISSON_ALPHA = 0.1
MAX_GOALS = 10
CLASS_LABELS = [0, 1, 2]

WEIGHT_COMBINATIONS = [
    {"poisson_weight": 1.00, "classifier_weight": 0.00},
    {"poisson_weight": 0.80, "classifier_weight": 0.20},
    {"poisson_weight": 0.60, "classifier_weight": 0.40},
    {"poisson_weight": 0.50, "classifier_weight": 0.50},
    {"poisson_weight": 0.40, "classifier_weight": 0.60},
    {"poisson_weight": 0.20, "classifier_weight": 0.80},
    {"poisson_weight": 0.00, "classifier_weight": 1.00},
]


# ============================================================
# 3. FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "neutral",
    "home_advantage",
    "home_elo",
    "away_elo",
    "elo_difference",
    "adjusted_elo_difference",
    "expected_home_elo_result",
    "expected_away_elo_result",
    "home_win_rate_5",
    "away_win_rate_5",
    "win_rate_difference_5",
    "home_points_per_game_5",
    "away_points_per_game_5",
    "points_per_game_difference_5",
    "home_goals_for_5",
    "away_goals_for_5",
    "goals_for_difference_5",
    "home_goals_against_5",
    "away_goals_against_5",
    "goals_against_difference_5",
    "home_goal_difference_5",
    "away_goal_difference_5",
    "recent_goal_difference_gap_5",
]

TARGET_COLUMN = "target"
HOME_GOAL_TARGET = "home_score"
AWAY_GOAL_TARGET = "away_score"


# ============================================================
# 4. LOAD DATA
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Feature dataset was not found:\n{INPUT_PATH}"
    )

df = pd.read_csv(
    INPUT_PATH,
    parse_dates=["date"]
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

print("=" * 80)
print("HYBRID RANDOM FOREST-POISSON WEIGHT TUNING")
print("=" * 80)

print("\nDataset shape:", df.shape)
print("Starting date:", df["date"].min())
print("Ending date:", df["date"].max())

required_columns = (
    [
        "date",
        TARGET_COLUMN,
        HOME_GOAL_TARGET,
        AWAY_GOAL_TARGET,
    ]
    + FEATURE_COLUMNS
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing:\n{missing_columns}"
    )

df["neutral"] = (
    df["neutral"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
    })
)

for column in FEATURE_COLUMNS:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce"
)

df[HOME_GOAL_TARGET] = pd.to_numeric(
    df[HOME_GOAL_TARGET],
    errors="coerce"
)

df[AWAY_GOAL_TARGET] = pd.to_numeric(
    df[AWAY_GOAL_TARGET],
    errors="coerce"
)

df = df.dropna(
    subset=required_columns
).copy()

df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
df[HOME_GOAL_TARGET] = df[HOME_GOAL_TARGET].astype(int)
df[AWAY_GOAL_TARGET] = df[AWAY_GOAL_TARGET].astype(int)
df["neutral"] = df["neutral"].astype(int)
df["home_advantage"] = df["home_advantage"].astype(int)


# ============================================================
# 5. MIRROR NEUTRAL MATCHES
# ============================================================

def make_mirrored_neutral_rows(input_df):
    neutral_rows = input_df[
        input_df["neutral"] == 1
    ].copy()

    if neutral_rows.empty:
        return neutral_rows

    mirrored = neutral_rows.copy()

    swap_pairs = [
        ("home_team", "away_team"),
        ("home_score", "away_score"),
        ("home_elo", "away_elo"),
        (
            "expected_home_elo_result",
            "expected_away_elo_result",
        ),
        (
            "home_previous_matches_5",
            "away_previous_matches_5",
        ),
        (
            "home_win_rate_5",
            "away_win_rate_5",
        ),
        (
            "home_draw_rate_5",
            "away_draw_rate_5",
        ),
        (
            "home_loss_rate_5",
            "away_loss_rate_5",
        ),
        (
            "home_points_per_game_5",
            "away_points_per_game_5",
        ),
        (
            "home_goals_for_5",
            "away_goals_for_5",
        ),
        (
            "home_goals_against_5",
            "away_goals_against_5",
        ),
        (
            "home_goal_difference_5",
            "away_goal_difference_5",
        ),
    ]

    for left_column, right_column in swap_pairs:
        if (
            left_column in mirrored.columns
            and right_column in mirrored.columns
        ):
            temporary_values = mirrored[left_column].copy()
            mirrored[left_column] = mirrored[right_column]
            mirrored[right_column] = temporary_values

    signed_difference_columns = [
        "elo_difference",
        "adjusted_elo_difference",
        "win_rate_difference_5",
        "draw_rate_difference_5",
        "loss_rate_difference_5",
        "points_per_game_difference_5",
        "goals_for_difference_5",
        "goals_against_difference_5",
        "recent_goal_difference_gap_5",
    ]

    for column in signed_difference_columns:
        if column in mirrored.columns:
            mirrored[column] = -mirrored[column]

    mirrored["neutral"] = 1
    mirrored["home_advantage"] = 0

    if TARGET_COLUMN in mirrored.columns:
        mirrored[TARGET_COLUMN] = (
            mirrored[TARGET_COLUMN]
            .map({
                0: 2,
                1: 1,
                2: 0,
            })
            .astype(int)
        )

    if "result" in mirrored.columns:
        mirrored["result"] = (
            mirrored["result"]
            .map({
                "Away Win": "Home Win",
                "Draw": "Draw",
                "Home Win": "Away Win",
            })
        )

    mirrored["is_mirrored"] = 1

    return mirrored.reset_index(drop=True)


def augment_training_data(input_df):
    original = input_df.copy()
    original["is_mirrored"] = 0

    mirrored = make_mirrored_neutral_rows(
        input_df
    )

    augmented = pd.concat(
        [original, mirrored],
        ignore_index=True
    )

    return (
        augmented
        .sort_values("date")
        .reset_index(drop=True)
    )


# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================

TRAIN_END = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2024-01-01")

train_df = df[
    df["date"] < TRAIN_END
].copy()

validation_df = df[
    (df["date"] >= TRAIN_END)
    &
    (df["date"] < VALIDATION_END)
].copy()

neutral_validation_df = validation_df[
    validation_df["neutral"] == 1
].copy().reset_index(drop=True)

if train_df.empty:
    raise ValueError("Training dataset is empty.")

if neutral_validation_df.empty:
    raise ValueError(
        "Neutral validation dataset is empty."
    )

augmented_train_df = augment_training_data(
    train_df
)

print("\n" + "=" * 80)
print("DATA SPLIT")
print("=" * 80)

print("\nOriginal training matches:", len(train_df))
print(
    "Mirrored training matches:",
    int(
        augmented_train_df[
            "is_mirrored"
        ].sum()
    )
)
print(
    "Augmented training rows:",
    len(augmented_train_df)
)
print(
    "\nNeutral validation matches:",
    len(neutral_validation_df)
)
print(
    "Validation period:",
    neutral_validation_df["date"].min(),
    "to",
    neutral_validation_df["date"].max()
)


# ============================================================
# 7. RANDOM FOREST MODEL
# ============================================================

classifier_model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=600,
            max_depth=14,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    ),
])

classifier_model.fit(
    augmented_train_df[FEATURE_COLUMNS],
    augmented_train_df[TARGET_COLUMN]
)


# ============================================================
# 8. POISSON MODELS
# ============================================================

def create_poisson_model():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            PoissonRegressor(
                alpha=POISSON_ALPHA,
                max_iter=3000,
                tol=1e-8,
            )
        ),
    ])


home_goal_model = create_poisson_model()
away_goal_model = create_poisson_model()

home_goal_model.fit(
    augmented_train_df[FEATURE_COLUMNS],
    augmented_train_df[HOME_GOAL_TARGET]
)

away_goal_model.fit(
    augmented_train_df[FEATURE_COLUMNS],
    augmented_train_df[AWAY_GOAL_TARGET]
)

print("\nRandom Forest and Poisson models trained.")


# ============================================================
# 9. ORIGINAL AND MIRRORED VALIDATION DATA
# ============================================================

mirrored_validation_df = (
    make_mirrored_neutral_rows(
        neutral_validation_df
    )
)

X_validation_original = (
    neutral_validation_df[
        FEATURE_COLUMNS
    ]
)

X_validation_mirrored = (
    mirrored_validation_df[
        FEATURE_COLUMNS
    ]
)

y_validation = (
    neutral_validation_df[
        TARGET_COLUMN
    ].to_numpy()
)


# ============================================================
# 10. SYMMETRICAL CLASSIFIER PROBABILITIES
# ============================================================

original_classifier_probabilities = (
    classifier_model.predict_proba(
        X_validation_original
    )
)

mirrored_classifier_probabilities = (
    classifier_model.predict_proba(
        X_validation_mirrored
    )
)

model_classes = (
    classifier_model
    .named_steps["model"]
    .classes_
)

class_positions = {
    int(class_value): position
    for position, class_value
    in enumerate(model_classes)
}

away_position = class_positions[0]
draw_position = class_positions[1]
home_position = class_positions[2]

classifier_home_win = (
    original_classifier_probabilities[
        :,
        home_position
    ]
    +
    mirrored_classifier_probabilities[
        :,
        away_position
    ]
) / 2.0

classifier_draw = (
    original_classifier_probabilities[
        :,
        draw_position
    ]
    +
    mirrored_classifier_probabilities[
        :,
        draw_position
    ]
) / 2.0

classifier_away_win = (
    original_classifier_probabilities[
        :,
        away_position
    ]
    +
    mirrored_classifier_probabilities[
        :,
        home_position
    ]
) / 2.0

classifier_probabilities = np.column_stack([
    classifier_away_win,
    classifier_draw,
    classifier_home_win,
])

classifier_probabilities = (
    classifier_probabilities
    /
    classifier_probabilities.sum(
        axis=1,
        keepdims=True
    )
)


# ============================================================
# 11. SYMMETRICAL EXPECTED GOALS
# ============================================================

original_home_expected_goals = (
    home_goal_model.predict(
        X_validation_original
    )
)

original_away_expected_goals = (
    away_goal_model.predict(
        X_validation_original
    )
)

mirrored_home_expected_goals = (
    home_goal_model.predict(
        X_validation_mirrored
    )
)

mirrored_away_expected_goals = (
    away_goal_model.predict(
        X_validation_mirrored
    )
)

symmetrical_home_expected_goals = (
    original_home_expected_goals
    +
    mirrored_away_expected_goals
) / 2.0

symmetrical_away_expected_goals = (
    original_away_expected_goals
    +
    mirrored_home_expected_goals
) / 2.0

symmetrical_home_expected_goals = np.clip(
    symmetrical_home_expected_goals,
    0.05,
    6.00
)

symmetrical_away_expected_goals = np.clip(
    symmetrical_away_expected_goals,
    0.05,
    6.00
)


# ============================================================
# 12. POISSON OUTCOME PROBABILITIES
# ============================================================

def poisson_probability(
    goals,
    expected_goals
):
    log_probability = (
        -expected_goals
        +
        goals * math.log(expected_goals)
        -
        math.lgamma(goals + 1)
    )

    return math.exp(
        log_probability
    )


def calculate_poisson_outcome_probabilities(
    home_expected_goals,
    away_expected_goals
):
    home_goal_probabilities = np.array([
        poisson_probability(
            goals,
            home_expected_goals
        )
        for goals in range(MAX_GOALS + 1)
    ])

    away_goal_probabilities = np.array([
        poisson_probability(
            goals,
            away_expected_goals
        )
        for goals in range(MAX_GOALS + 1)
    ])

    score_matrix = np.outer(
        home_goal_probabilities,
        away_goal_probabilities
    )

    score_matrix = (
        score_matrix /
        score_matrix.sum()
    )

    home_win_probability = float(
        np.tril(
            score_matrix,
            k=-1
        ).sum()
    )

    draw_probability = float(
        np.trace(
            score_matrix
        )
    )

    away_win_probability = float(
        np.triu(
            score_matrix,
            k=1
        ).sum()
    )

    return [
        away_win_probability,
        draw_probability,
        home_win_probability,
    ]


poisson_probability_rows = []

for home_expected, away_expected in zip(
    symmetrical_home_expected_goals,
    symmetrical_away_expected_goals
):
    poisson_probability_rows.append(
        calculate_poisson_outcome_probabilities(
            home_expected_goals=home_expected,
            away_expected_goals=away_expected
        )
    )

poisson_probabilities = np.array(
    poisson_probability_rows
)


# ============================================================
# 13. EVALUATE WEIGHT COMBINATIONS
# ============================================================

one_hot_targets = np.eye(3)[
    y_validation
]

comparison_rows = []

print("\n" + "=" * 80)
print("HYBRID WEIGHT COMPARISON")
print("=" * 80)

for weights in WEIGHT_COMBINATIONS:
    poisson_weight = weights[
        "poisson_weight"
    ]

    classifier_weight = weights[
        "classifier_weight"
    ]

    blended_probabilities = (
        poisson_weight
        *
        poisson_probabilities
        +
        classifier_weight
        *
        classifier_probabilities
    )

    blended_probabilities = (
        blended_probabilities
        /
        blended_probabilities.sum(
            axis=1,
            keepdims=True
        )
    )

    predicted_classes = np.argmax(
        blended_probabilities,
        axis=1
    )

    validation_log_loss = log_loss(
        y_validation,
        blended_probabilities,
        labels=CLASS_LABELS
    )

    validation_accuracy = accuracy_score(
        y_validation,
        predicted_classes
    )

    validation_macro_f1 = f1_score(
        y_validation,
        predicted_classes,
        average="macro"
    )

    validation_weighted_f1 = f1_score(
        y_validation,
        predicted_classes,
        average="weighted"
    )

    multiclass_brier_score = np.mean(
        np.sum(
            (
                blended_probabilities
                -
                one_hot_targets
            ) ** 2,
            axis=1
        )
    )

    comparison_rows.append({
        "poisson_weight":
            poisson_weight,
        "classifier_weight":
            classifier_weight,
        "validation_log_loss":
            validation_log_loss,
        "validation_accuracy":
            validation_accuracy,
        "validation_macro_f1":
            validation_macro_f1,
        "validation_weighted_f1":
            validation_weighted_f1,
        "multiclass_brier_score":
            multiclass_brier_score,
    })

    print(
        f"\nPoisson {poisson_weight:.0%} | "
        f"Classifier {classifier_weight:.0%}"
    )

    print(
        "Log loss:",
        round(validation_log_loss, 4)
    )

    print(
        "Accuracy:",
        round(validation_accuracy, 4)
    )

    print(
        "Macro F1:",
        round(validation_macro_f1, 4)
    )

    print(
        "Brier score:",
        round(multiclass_brier_score, 4)
    )


# ============================================================
# 14. SELECT AND SAVE BEST WEIGHTS
# ============================================================

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df = (
    comparison_df
    .sort_values(
        by="validation_log_loss",
        ascending=True
    )
    .reset_index(drop=True)
)

best_row = comparison_df.iloc[0]

best_poisson_weight = float(
    best_row["poisson_weight"]
)

best_classifier_weight = float(
    best_row["classifier_weight"]
)

comparison_df[
    "selected_weight"
] = False

comparison_df.loc[
    0,
    "selected_weight"
] = True

print("\n" + "=" * 80)
print("FINAL HYBRID-WEIGHT RANKING")
print("=" * 80)

print(
    comparison_df
    .round(4)
    .to_string(index=False)
)

print("\nBest weight combination:")

print(
    "Poisson weight:",
    f"{best_poisson_weight:.0%}"
)

print(
    "Classifier weight:",
    f"{best_classifier_weight:.0%}"
)

print(
    "Validation log loss:",
    round(
        float(
            best_row[
                "validation_log_loss"
            ]
        ),
        4
    )
)

comparison_df.to_csv(
    RESULTS_OUTPUT_PATH,
    index=False
)

best_weight_information = {
    "poisson_weight":
        best_poisson_weight,
    "classifier_weight":
        best_classifier_weight,
    "selection_metric":
        "neutral_validation_log_loss",
    "validation_log_loss":
        float(
            best_row[
                "validation_log_loss"
            ]
        ),
    "validation_accuracy":
        float(
            best_row[
                "validation_accuracy"
            ]
        ),
    "validation_macro_f1":
        float(
            best_row[
                "validation_macro_f1"
            ]
        ),
    "multiclass_brier_score":
        float(
            best_row[
                "multiclass_brier_score"
            ]
        ),
    "neutral_validation_matches":
        int(
            len(
                neutral_validation_df
            )
        ),
    "validation_start_date":
        str(
            neutral_validation_df[
                "date"
            ].min().date()
        ),
    "validation_end_date":
        str(
            neutral_validation_df[
                "date"
            ].max().date()
        ),
}

with open(
    BEST_WEIGHT_OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as output_file:
    json.dump(
        best_weight_information,
        output_file,
        indent=4
    )

print("\n" + "=" * 80)
print("HYBRID-WEIGHT TUNING COMPLETED")
print("=" * 80)

print("\nComparison table saved to:")
print(RESULTS_OUTPUT_PATH)

print("\nBest weights saved to:")
print(BEST_WEIGHT_OUTPUT_PATH)

print("\nUse these values automatically in simulate_match_score.py:")

print(
    f"CLASSIFIER_WEIGHT = "
    f"{best_classifier_weight:.2f}"
)

print(
    f"POISSON_WEIGHT = "
    f"{best_poisson_weight:.2f}"
)
