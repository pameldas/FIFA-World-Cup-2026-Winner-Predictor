import json
import joblib
import math
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

DATA_FOLDER = Path(r"D:\wc data")

INPUT_PATH = DATA_FOLDER / "ml_features.csv"
WEIGHT_PATH = DATA_FOLDER / "best_hybrid_weights.json"
GOAL_MODEL_METADATA_PATH = (
    DATA_FOLDER / "best_goal_prediction_models.joblib"
)

METRICS_OUTPUT_PATH = (
    DATA_FOLDER / "final_hybrid_test_metrics.csv"
)

PREDICTIONS_OUTPUT_PATH = (
    DATA_FOLDER / "final_hybrid_test_predictions.csv"
)

REPORT_OUTPUT_PATH = (
    DATA_FOLDER / "final_hybrid_test_report.txt"
)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_STATE = 42
MAX_GOALS = 10

CLASS_LABELS = [0, 1, 2]

CLASS_NAMES = {
    0: "Away Win",
    1: "Draw",
    2: "Home Win",
}

TARGET_COLUMN = "target"
HOME_GOAL_TARGET = "home_score"
AWAY_GOAL_TARGET = "away_score"


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


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

for required_path in [
    INPUT_PATH,
    WEIGHT_PATH,
    GOAL_MODEL_METADATA_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n{required_path}"
        )


# ============================================================
# 5. LOAD SELECTED WEIGHTS AND POISSON ALPHA
# ============================================================

with open(
    WEIGHT_PATH,
    "r",
    encoding="utf-8",
) as input_file:
    weight_information = json.load(input_file)

POISSON_WEIGHT = float(
    weight_information["poisson_weight"]
)

CLASSIFIER_WEIGHT = float(
    weight_information["classifier_weight"]
)

total_weight = (
    POISSON_WEIGHT
    +
    CLASSIFIER_WEIGHT
)

if total_weight <= 0:
    raise ValueError(
        "The selected hybrid weights have an invalid total."
    )

POISSON_WEIGHT /= total_weight
CLASSIFIER_WEIGHT /= total_weight


goal_model_metadata = joblib.load(
    GOAL_MODEL_METADATA_PATH
)

POISSON_ALPHA = float(
    goal_model_metadata.get(
        "best_alpha",
        0.1,
    )
)


# ============================================================
# 6. LOAD AND CLEAN DATA
# ============================================================

df = pd.read_csv(
    INPUT_PATH,
    parse_dates=["date"],
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

print("=" * 84)
print("FINAL UNTOUCHED TEST EVALUATION")
print("RANDOM FOREST + POISSON WORLD CUP MATCH MODEL")
print("=" * 84)

print("\nDataset shape:", df.shape)
print("Dataset start:", df["date"].min())
print("Dataset end:", df["date"].max())

print("\nSelected weights:")
print("Random Forest:", f"{CLASSIFIER_WEIGHT:.0%}")
print("Poisson:", f"{POISSON_WEIGHT:.0%}")
print("Poisson alpha:", POISSON_ALPHA)


required_columns = (
    [
        "date",
        "home_team",
        "away_team",
        "result",
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
        errors="coerce",
    )

df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce",
)

df[HOME_GOAL_TARGET] = pd.to_numeric(
    df[HOME_GOAL_TARGET],
    errors="coerce",
)

df[AWAY_GOAL_TARGET] = pd.to_numeric(
    df[AWAY_GOAL_TARGET],
    errors="coerce",
)

df = df.dropna(
    subset=required_columns
).copy()

df["neutral"] = df["neutral"].astype(int)
df["home_advantage"] = df["home_advantage"].astype(int)
df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
df[HOME_GOAL_TARGET] = df[HOME_GOAL_TARGET].astype(int)
df[AWAY_GOAL_TARGET] = df[AWAY_GOAL_TARGET].astype(int)


# ============================================================
# 7. MIRROR NEUTRAL MATCHES
# ============================================================

def make_mirrored_neutral_rows(input_df):
    """
    Reverse neutral matches and swap all home/away features.
    """

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
    """
    Add reversed copies of neutral matches to training only.
    """

    original = input_df.copy()
    original["is_mirrored"] = 0

    mirrored = make_mirrored_neutral_rows(
        input_df
    )

    augmented = pd.concat(
        [original, mirrored],
        ignore_index=True,
    )

    return (
        augmented
        .sort_values("date")
        .reset_index(drop=True)
    )


# ============================================================
# 8. FINAL TRAIN / TEST SPLIT
# ============================================================

# Train on all available data through 31 December 2023.
# Test only on untouched matches from 1 January 2024 onward.

TEST_START = pd.Timestamp("2024-01-01")

train_validation_df = df[
    df["date"] < TEST_START
].copy()

test_df = df[
    df["date"] >= TEST_START
].copy()

if train_validation_df.empty:
    raise ValueError(
        "The train + validation dataset is empty."
    )

if test_df.empty:
    raise ValueError(
        "The untouched test dataset is empty."
    )

augmented_train_validation_df = augment_training_data(
    train_validation_df
)

print("\n" + "=" * 84)
print("FINAL DATA SPLIT")
print("=" * 84)

print(
    "\nOriginal training + validation rows:",
    len(train_validation_df),
)

print(
    "Mirrored neutral rows added:",
    int(
        augmented_train_validation_df[
            "is_mirrored"
        ].sum()
    ),
)

print(
    "Augmented training rows:",
    len(augmented_train_validation_df),
)

print(
    "Training period:",
    train_validation_df["date"].min(),
    "to",
    train_validation_df["date"].max(),
)

print("\nUntouched test rows:", len(test_df))

print(
    "Test period:",
    test_df["date"].min(),
    "to",
    test_df["date"].max(),
)

print(
    "Neutral test matches:",
    int(
        (test_df["neutral"] == 1).sum()
    ),
)

print(
    "Non-neutral test matches:",
    int(
        (test_df["neutral"] == 0).sum()
    ),
)


# ============================================================
# 9. TRAIN FINAL RANDOM FOREST
# ============================================================

classifier_model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median"),
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
        ),
    ),
])

classifier_model.fit(
    augmented_train_validation_df[
        FEATURE_COLUMNS
    ],
    augmented_train_validation_df[
        TARGET_COLUMN
    ],
)


# ============================================================
# 10. TRAIN FINAL POISSON MODELS
# ============================================================

def create_poisson_model():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            PoissonRegressor(
                alpha=POISSON_ALPHA,
                max_iter=3000,
                tol=1e-8,
            ),
        ),
    ])


home_goal_model = create_poisson_model()
away_goal_model = create_poisson_model()

home_goal_model.fit(
    augmented_train_validation_df[
        FEATURE_COLUMNS
    ],
    augmented_train_validation_df[
        HOME_GOAL_TARGET
    ],
)

away_goal_model.fit(
    augmented_train_validation_df[
        FEATURE_COLUMNS
    ],
    augmented_train_validation_df[
        AWAY_GOAL_TARGET
    ],
)

print("\nFinal Random Forest and Poisson models trained.")


# ============================================================
# 11. CLASSIFIER PROBABILITIES
# ============================================================

def get_classifier_probabilities(
    trained_model,
    evaluation_df,
):
    """
    Use symmetrical probabilities for neutral matches and
    direct probabilities for genuine home matches.
    """

    output_probabilities = np.zeros(
        (len(evaluation_df), 3),
        dtype=float,
    )

    model_classes = (
        trained_model
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

    neutral_mask = (
        evaluation_df["neutral"].to_numpy()
        == 1
    )

    non_neutral_mask = ~neutral_mask

    # Direct probabilities for non-neutral matches.
    if non_neutral_mask.any():
        non_neutral_df = evaluation_df.loc[
            non_neutral_mask
        ]

        direct_probabilities = (
            trained_model.predict_proba(
                non_neutral_df[
                    FEATURE_COLUMNS
                ]
            )
        )

        output_probabilities[
            non_neutral_mask,
            0
        ] = direct_probabilities[
            :,
            away_position
        ]

        output_probabilities[
            non_neutral_mask,
            1
        ] = direct_probabilities[
            :,
            draw_position
        ]

        output_probabilities[
            non_neutral_mask,
            2
        ] = direct_probabilities[
            :,
            home_position
        ]

    # Averaged probabilities for neutral matches.
    if neutral_mask.any():
        neutral_original = (
            evaluation_df.loc[
                neutral_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        neutral_mirrored = (
            make_mirrored_neutral_rows(
                neutral_original
            )
        )

        original_probabilities = (
            trained_model.predict_proba(
                neutral_original[
                    FEATURE_COLUMNS
                ]
            )
        )

        mirrored_probabilities = (
            trained_model.predict_proba(
                neutral_mirrored[
                    FEATURE_COLUMNS
                ]
            )
        )

        neutral_away_probability = (
            original_probabilities[
                :,
                away_position
            ]
            +
            mirrored_probabilities[
                :,
                home_position
            ]
        ) / 2.0

        neutral_draw_probability = (
            original_probabilities[
                :,
                draw_position
            ]
            +
            mirrored_probabilities[
                :,
                draw_position
            ]
        ) / 2.0

        neutral_home_probability = (
            original_probabilities[
                :,
                home_position
            ]
            +
            mirrored_probabilities[
                :,
                away_position
            ]
        ) / 2.0

        neutral_probabilities = np.column_stack([
            neutral_away_probability,
            neutral_draw_probability,
            neutral_home_probability,
        ])

        neutral_probabilities = (
            neutral_probabilities
            /
            neutral_probabilities.sum(
                axis=1,
                keepdims=True,
            )
        )

        output_probabilities[
            neutral_mask
        ] = neutral_probabilities

    return output_probabilities


classifier_probabilities = (
    get_classifier_probabilities(
        classifier_model,
        test_df,
    )
)


# ============================================================
# 12. EXPECTED GOALS
# ============================================================

def get_expected_goals(
    home_model,
    away_model,
    evaluation_df,
):
    """
    Average both team orientations for neutral matches.
    Use direct home/away estimates for non-neutral matches.
    """

    predicted_home_goals = np.zeros(
        len(evaluation_df),
        dtype=float,
    )

    predicted_away_goals = np.zeros(
        len(evaluation_df),
        dtype=float,
    )

    neutral_mask = (
        evaluation_df["neutral"].to_numpy()
        == 1
    )

    non_neutral_mask = ~neutral_mask

    if non_neutral_mask.any():
        non_neutral_df = evaluation_df.loc[
            non_neutral_mask
        ]

        predicted_home_goals[
            non_neutral_mask
        ] = home_model.predict(
            non_neutral_df[
                FEATURE_COLUMNS
            ]
        )

        predicted_away_goals[
            non_neutral_mask
        ] = away_model.predict(
            non_neutral_df[
                FEATURE_COLUMNS
            ]
        )

    if neutral_mask.any():
        neutral_original = (
            evaluation_df.loc[
                neutral_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        neutral_mirrored = (
            make_mirrored_neutral_rows(
                neutral_original
            )
        )

        original_home_expected = (
            home_model.predict(
                neutral_original[
                    FEATURE_COLUMNS
                ]
            )
        )

        original_away_expected = (
            away_model.predict(
                neutral_original[
                    FEATURE_COLUMNS
                ]
            )
        )

        mirrored_home_expected = (
            home_model.predict(
                neutral_mirrored[
                    FEATURE_COLUMNS
                ]
            )
        )

        mirrored_away_expected = (
            away_model.predict(
                neutral_mirrored[
                    FEATURE_COLUMNS
                ]
            )
        )

        predicted_home_goals[
            neutral_mask
        ] = (
            original_home_expected
            +
            mirrored_away_expected
        ) / 2.0

        predicted_away_goals[
            neutral_mask
        ] = (
            original_away_expected
            +
            mirrored_home_expected
        ) / 2.0

    predicted_home_goals = np.clip(
        predicted_home_goals,
        0.05,
        6.00,
    )

    predicted_away_goals = np.clip(
        predicted_away_goals,
        0.05,
        6.00,
    )

    return (
        predicted_home_goals,
        predicted_away_goals,
    )


(
    predicted_home_expected_goals,
    predicted_away_expected_goals,
) = get_expected_goals(
    home_goal_model,
    away_goal_model,
    test_df,
)


# ============================================================
# 13. POISSON OUTCOME PROBABILITIES
# ============================================================

def poisson_probability(
    goals,
    expected_goals,
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
    away_expected_goals,
):
    home_goal_probabilities = np.array([
        poisson_probability(
            goals,
            home_expected_goals,
        )
        for goals in range(MAX_GOALS + 1)
    ])

    away_goal_probabilities = np.array([
        poisson_probability(
            goals,
            away_expected_goals,
        )
        for goals in range(MAX_GOALS + 1)
    ])

    score_matrix = np.outer(
        home_goal_probabilities,
        away_goal_probabilities,
    )

    score_matrix = (
        score_matrix
        /
        score_matrix.sum()
    )

    home_win_probability = float(
        np.tril(
            score_matrix,
            k=-1,
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
            k=1,
        ).sum()
    )

    return [
        away_win_probability,
        draw_probability,
        home_win_probability,
    ]


poisson_probability_rows = []

for home_expected, away_expected in zip(
    predicted_home_expected_goals,
    predicted_away_expected_goals,
):
    poisson_probability_rows.append(
        calculate_poisson_outcome_probabilities(
            home_expected,
            away_expected,
        )
    )

poisson_probabilities = np.array(
    poisson_probability_rows
)


# ============================================================
# 14. FINAL BLENDED PROBABILITIES
# ============================================================

hybrid_probabilities = (
    CLASSIFIER_WEIGHT
    *
    classifier_probabilities
    +
    POISSON_WEIGHT
    *
    poisson_probabilities
)

hybrid_probabilities = (
    hybrid_probabilities
    /
    hybrid_probabilities.sum(
        axis=1,
        keepdims=True,
    )
)

predicted_classes = np.argmax(
    hybrid_probabilities,
    axis=1,
)

actual_classes = (
    test_df[TARGET_COLUMN]
    .to_numpy()
)


# ============================================================
# 15. METRIC FUNCTIONS
# ============================================================

def calculate_subset_metrics(
    subset_name,
    row_mask,
):
    """
    Calculate probability and classification metrics for one
    part of the final test set.
    """

    row_mask = np.asarray(
        row_mask,
        dtype=bool,
    )

    subset_actual = actual_classes[
        row_mask
    ]

    subset_probabilities = hybrid_probabilities[
        row_mask
    ]

    subset_predictions = predicted_classes[
        row_mask
    ]

    one_hot_actual = np.eye(3)[
        subset_actual
    ]

    brier_score = np.mean(
        np.sum(
            (
                subset_probabilities
                -
                one_hot_actual
            ) ** 2,
            axis=1,
        )
    )

    return {
        "subset": subset_name,
        "matches": int(row_mask.sum()),
        "accuracy": accuracy_score(
            subset_actual,
            subset_predictions,
        ),
        "macro_f1": f1_score(
            subset_actual,
            subset_predictions,
            average="macro",
        ),
        "weighted_f1": f1_score(
            subset_actual,
            subset_predictions,
            average="weighted",
        ),
        "log_loss": log_loss(
            subset_actual,
            subset_probabilities,
            labels=CLASS_LABELS,
        ),
        "multiclass_brier_score": brier_score,
    }


all_mask = np.ones(
    len(test_df),
    dtype=bool,
)

neutral_mask = (
    test_df["neutral"].to_numpy()
    == 1
)

non_neutral_mask = (
    test_df["neutral"].to_numpy()
    == 0
)

metrics_rows = [
    calculate_subset_metrics(
        "All test matches",
        all_mask,
    ),
    calculate_subset_metrics(
        "Neutral test matches",
        neutral_mask,
    ),
    calculate_subset_metrics(
        "Non-neutral test matches",
        non_neutral_mask,
    ),
]

metrics_df = pd.DataFrame(
    metrics_rows
)


# ============================================================
# 16. GOAL-PREDICTION METRICS
# ============================================================

actual_home_goals = (
    test_df[HOME_GOAL_TARGET]
    .to_numpy()
)

actual_away_goals = (
    test_df[AWAY_GOAL_TARGET]
    .to_numpy()
)

home_goal_mae = mean_absolute_error(
    actual_home_goals,
    predicted_home_expected_goals,
)

away_goal_mae = mean_absolute_error(
    actual_away_goals,
    predicted_away_expected_goals,
)

combined_goal_mae = (
    home_goal_mae
    +
    away_goal_mae
) / 2.0

home_goal_rmse = np.sqrt(
    mean_squared_error(
        actual_home_goals,
        predicted_home_expected_goals,
    )
)

away_goal_rmse = np.sqrt(
    mean_squared_error(
        actual_away_goals,
        predicted_away_expected_goals,
    )
)

combined_goal_rmse = (
    home_goal_rmse
    +
    away_goal_rmse
) / 2.0

rounded_home_scores = np.rint(
    predicted_home_expected_goals
).astype(int)

rounded_away_scores = np.rint(
    predicted_away_expected_goals
).astype(int)

exact_score_accuracy = np.mean(
    (
        rounded_home_scores
        ==
        actual_home_goals
    )
    &
    (
        rounded_away_scores
        ==
        actual_away_goals
    )
)


# ============================================================
# 17. CONFUSION MATRIX AND REPORT
# ============================================================

final_confusion_matrix = confusion_matrix(
    actual_classes,
    predicted_classes,
    labels=CLASS_LABELS,
)

confusion_df = pd.DataFrame(
    final_confusion_matrix,
    index=[
        "Actual Away Win",
        "Actual Draw",
        "Actual Home Win",
    ],
    columns=[
        "Predicted Away Win",
        "Predicted Draw",
        "Predicted Home Win",
    ],
)

classification_text = classification_report(
    actual_classes,
    predicted_classes,
    labels=CLASS_LABELS,
    target_names=[
        "Away Win",
        "Draw",
        "Home Win",
    ],
    digits=4,
    zero_division=0,
)


# ============================================================
# 18. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 84)
print("FINAL TEST METRICS")
print("=" * 84)

print(
    metrics_df
    .round(4)
    .to_string(index=False)
)

print("\nGoal-prediction performance:")

print(
    "Home-goal MAE:",
    round(home_goal_mae, 4),
)

print(
    "Away-goal MAE:",
    round(away_goal_mae, 4),
)

print(
    "Combined goal MAE:",
    round(combined_goal_mae, 4),
)

print(
    "Combined goal RMSE:",
    round(combined_goal_rmse, 4),
)

print(
    "Rounded exact-score accuracy:",
    f"{exact_score_accuracy:.2%}",
)

print("\nConfusion matrix:")
print(confusion_df.to_string())

print("\nClassification report:")
print(classification_text)


# ============================================================
# 19. SAVE TEST PREDICTIONS
# ============================================================

test_output = test_df[
    [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "neutral",
        "result",
        "target",
    ]
].copy()

test_output[
    "predicted_home_expected_goals"
] = predicted_home_expected_goals

test_output[
    "predicted_away_expected_goals"
] = predicted_away_expected_goals

test_output[
    "probability_away_win"
] = hybrid_probabilities[:, 0]

test_output[
    "probability_draw"
] = hybrid_probabilities[:, 1]

test_output[
    "probability_home_win"
] = hybrid_probabilities[:, 2]

test_output[
    "predicted_target"
] = predicted_classes

test_output[
    "predicted_result"
] = test_output[
    "predicted_target"
].map(CLASS_NAMES)

test_output[
    "rounded_home_score"
] = rounded_home_scores

test_output[
    "rounded_away_score"
] = rounded_away_scores

test_output.to_csv(
    PREDICTIONS_OUTPUT_PATH,
    index=False,
)


# ============================================================
# 20. SAVE METRICS AND REPORT
# ============================================================

metrics_df[
    "classifier_weight"
] = CLASSIFIER_WEIGHT

metrics_df[
    "poisson_weight"
] = POISSON_WEIGHT

metrics_df[
    "poisson_alpha"
] = POISSON_ALPHA

metrics_df.to_csv(
    METRICS_OUTPUT_PATH,
    index=False,
)


with open(
    REPORT_OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as report_file:

    report_file.write(
        "FINAL UNTOUCHED TEST EVALUATION\n"
    )

    report_file.write(
        "Random Forest + Poisson World Cup Model\n"
    )

    report_file.write("=" * 78 + "\n\n")

    report_file.write(
        f"Random Forest weight: "
        f"{CLASSIFIER_WEIGHT:.4f}\n"
    )

    report_file.write(
        f"Poisson weight: "
        f"{POISSON_WEIGHT:.4f}\n"
    )

    report_file.write(
        f"Poisson alpha: "
        f"{POISSON_ALPHA:.4f}\n\n"
    )

    report_file.write(
        "Final test metrics:\n"
    )

    report_file.write(
        metrics_df
        .round(4)
        .to_string(index=False)
    )

    report_file.write("\n\n")

    report_file.write(
        f"Home-goal MAE: "
        f"{home_goal_mae:.4f}\n"
    )

    report_file.write(
        f"Away-goal MAE: "
        f"{away_goal_mae:.4f}\n"
    )

    report_file.write(
        f"Combined goal MAE: "
        f"{combined_goal_mae:.4f}\n"
    )

    report_file.write(
        f"Combined goal RMSE: "
        f"{combined_goal_rmse:.4f}\n"
    )

    report_file.write(
        f"Rounded exact-score accuracy: "
        f"{exact_score_accuracy:.4%}\n\n"
    )

    report_file.write(
        "Confusion matrix:\n"
    )

    report_file.write(
        confusion_df.to_string()
    )

    report_file.write("\n\n")

    report_file.write(
        "Classification report:\n"
    )

    report_file.write(
        classification_text
    )


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 84)
print("FINAL TEST EVALUATION COMPLETED")
print("=" * 84)

print("\nMetrics saved to:")
print(METRICS_OUTPUT_PATH)

print("\nPredictions saved to:")
print(PREDICTIONS_OUTPUT_PATH)

print("\nReport saved to:")
print(REPORT_OUTPUT_PATH)

print(
    "\nImportant: these test results are for reporting only. "
    "Do not retune the weights using this test set."
)
