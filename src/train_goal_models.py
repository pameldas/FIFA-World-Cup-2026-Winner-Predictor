import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_poisson_deviance
)


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

DATA_FOLDER = Path(r"D:\wc data")

INPUT_PATH = DATA_FOLDER / "ml_features.csv"

MODEL_OUTPUT_PATH = (
    DATA_FOLDER /
    "best_goal_prediction_models.joblib"
)

COMPARISON_OUTPUT_PATH = (
    DATA_FOLDER /
    "goal_model_comparison.csv"
)

TEST_PREDICTIONS_OUTPUT_PATH = (
    DATA_FOLDER /
    "goal_test_predictions.csv"
)

REPORT_OUTPUT_PATH = (
    DATA_FOLDER /
    "goal_model_report.txt"
)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_STATE = 42

# These values control model regularization.
ALPHA_VALUES = [
    0.001,
    0.01,
    0.1,
    0.5,
    1.0
]

# Limit expected goals to a reasonable range when evaluating.
MIN_EXPECTED_GOALS = 0.05
MAX_EXPECTED_GOALS = 6.00


# ============================================================
# 3. MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [

    # Match setting
    "neutral",
    "home_advantage",

    # Elo
    "home_elo",
    "away_elo",
    "elo_difference",
    "adjusted_elo_difference",
    "expected_home_elo_result",
    "expected_away_elo_result",

    # Recent win form
    "home_win_rate_5",
    "away_win_rate_5",
    "win_rate_difference_5",

    # Recent points
    "home_points_per_game_5",
    "away_points_per_game_5",
    "points_per_game_difference_5",

    # Recent goals scored
    "home_goals_for_5",
    "away_goals_for_5",
    "goals_for_difference_5",

    # Recent goals conceded
    "home_goals_against_5",
    "away_goals_against_5",
    "goals_against_difference_5",

    # Recent goal difference
    "home_goal_difference_5",
    "away_goal_difference_5",
    "recent_goal_difference_gap_5"
]

HOME_TARGET = "home_score"
AWAY_TARGET = "away_score"


# ============================================================
# 4. CHECK INPUT FILE
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"ML feature dataset was not found:\n{INPUT_PATH}"
    )


# ============================================================
# 5. LOAD DATA
# ============================================================

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
print("WORLD CUP POISSON GOAL-MODEL TRAINING")
print("=" * 80)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)
print("Starting date:", df["date"].min())
print("Ending date:", df["date"].max())


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    ["date", HOME_TARGET, AWAY_TARGET] +
    FEATURE_COLUMNS
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Required columns are missing:\n"
        f"{missing_columns}"
    )


# ============================================================
# 7. CLEAN MODEL COLUMNS
# ============================================================

df["neutral"] = (
    df["neutral"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0
    })
)

for column in FEATURE_COLUMNS:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df[HOME_TARGET] = pd.to_numeric(
    df[HOME_TARGET],
    errors="coerce"
)

df[AWAY_TARGET] = pd.to_numeric(
    df[AWAY_TARGET],
    errors="coerce"
)

df = df.dropna(
    subset=required_columns
).copy()

df["neutral"] = df["neutral"].astype(int)
df["home_advantage"] = df["home_advantage"].astype(int)

df[HOME_TARGET] = df[HOME_TARGET].astype(int)
df[AWAY_TARGET] = df[AWAY_TARGET].astype(int)

# Goal counts must be non-negative.
df = df[
    (df[HOME_TARGET] >= 0) &
    (df[AWAY_TARGET] >= 0)
].copy()

print("\nCleaned dataset shape:", df.shape)

print(
    "Average home goals:",
    round(df[HOME_TARGET].mean(), 4)
)

print(
    "Average away goals:",
    round(df[AWAY_TARGET].mean(), 4)
)

print(
    "Neutral matches:",
    int(df["neutral"].sum())
)


# ============================================================
# 8. MIRROR NEUTRAL MATCHES
# ============================================================

def make_mirrored_neutral_rows(input_df):
    """
    Reverse neutral matches so that the goal models learn that
    team-listing order should not strongly affect predictions.
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
            "expected_away_elo_result"
        ),

        (
            "home_previous_matches_5",
            "away_previous_matches_5"
        ),

        (
            "home_win_rate_5",
            "away_win_rate_5"
        ),

        (
            "home_draw_rate_5",
            "away_draw_rate_5"
        ),

        (
            "home_loss_rate_5",
            "away_loss_rate_5"
        ),

        (
            "home_points_per_game_5",
            "away_points_per_game_5"
        ),

        (
            "home_goals_for_5",
            "away_goals_for_5"
        ),

        (
            "home_goals_against_5",
            "away_goals_against_5"
        ),

        (
            "home_goal_difference_5",
            "away_goal_difference_5"
        )
    ]

    for left_column, right_column in swap_pairs:

        if (
            left_column in mirrored.columns and
            right_column in mirrored.columns
        ):
            temporary_values = (
                mirrored[left_column]
                .copy()
            )

            mirrored[left_column] = (
                mirrored[right_column]
            )

            mirrored[right_column] = (
                temporary_values
            )

    signed_difference_columns = [
        "elo_difference",
        "adjusted_elo_difference",
        "win_rate_difference_5",
        "draw_rate_difference_5",
        "loss_rate_difference_5",
        "points_per_game_difference_5",
        "goals_for_difference_5",
        "goals_against_difference_5",
        "recent_goal_difference_gap_5"
    ]

    for column in signed_difference_columns:

        if column in mirrored.columns:
            mirrored[column] = (
                -mirrored[column]
            )

    mirrored["neutral"] = 1
    mirrored["home_advantage"] = 0
    mirrored["is_mirrored"] = 1

    return mirrored.reset_index(drop=True)


def augment_training_data(input_df):
    """
    Add mirrored neutral matches to the training dataset.
    """

    original = input_df.copy()
    original["is_mirrored"] = 0

    mirrored = make_mirrored_neutral_rows(
        input_df
    )

    augmented = pd.concat(
        [original, mirrored],
        ignore_index=True
    )

    augmented = (
        augmented
        .sort_values("date")
        .reset_index(drop=True)
    )

    return augmented


# ============================================================
# 9. CHRONOLOGICAL SPLIT
# ============================================================

# Training: 2000–2021
# Validation: 2022–2023
# Testing: 2024–June 2026

TRAIN_END = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2024-01-01")

train_df = df[
    df["date"] < TRAIN_END
].copy()

validation_df = df[
    (df["date"] >= TRAIN_END) &
    (df["date"] < VALIDATION_END)
].copy()

test_df = df[
    df["date"] >= VALIDATION_END
].copy()

if (
    train_df.empty or
    validation_df.empty or
    test_df.empty
):
    raise ValueError(
        "At least one chronological split is empty."
    )

augmented_train_df = augment_training_data(
    train_df
)

print("\n" + "=" * 80)
print("CHRONOLOGICAL DATA SPLIT")
print("=" * 80)

print("\nOriginal training matches:", len(train_df))

print(
    "Mirrored neutral training rows:",
    int(augmented_train_df["is_mirrored"].sum())
)

print(
    "Augmented training rows:",
    len(augmented_train_df)
)

print(
    "Training period:",
    train_df["date"].min(),
    "to",
    train_df["date"].max()
)

print("\nValidation matches:", len(validation_df))

print(
    "Validation period:",
    validation_df["date"].min(),
    "to",
    validation_df["date"].max()
)

print("\nTesting matches:", len(test_df))

print(
    "Testing period:",
    test_df["date"].min(),
    "to",
    test_df["date"].max()
)


# ============================================================
# 10. PREPARE TRAINING DATA
# ============================================================

X_train = augmented_train_df[
    FEATURE_COLUMNS
]

y_home_train = augmented_train_df[
    HOME_TARGET
]

y_away_train = augmented_train_df[
    AWAY_TARGET
]

X_validation = validation_df[
    FEATURE_COLUMNS
]

y_home_validation = validation_df[
    HOME_TARGET
]

y_away_validation = validation_df[
    AWAY_TARGET
]

X_test = test_df[
    FEATURE_COLUMNS
]

y_home_test = test_df[
    HOME_TARGET
]

y_away_test = test_df[
    AWAY_TARGET
]


# ============================================================
# 11. MODEL CREATION
# ============================================================

def create_poisson_pipeline(alpha):
    """
    Create a Poisson regression pipeline.
    """

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
                alpha=alpha,
                max_iter=3000,
                tol=1e-8
            )
        )
    ])


# ============================================================
# 12. PREDICTION CLIPPING
# ============================================================

def clip_expected_goals(predictions):
    """
    Prevent impossible or extreme expected-goal values.
    """

    return np.clip(
        predictions,
        MIN_EXPECTED_GOALS,
        MAX_EXPECTED_GOALS
    )


# ============================================================
# 13. EVALUATION FUNCTION
# ============================================================

def evaluate_goal_predictions(
    actual_goals,
    predicted_goals
):
    """
    Calculate goal-prediction metrics.
    """

    predicted_goals = clip_expected_goals(
        predicted_goals
    )

    mae = mean_absolute_error(
        actual_goals,
        predicted_goals
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_goals,
            predicted_goals
        )
    )

    poisson_deviance = mean_poisson_deviance(
        actual_goals,
        predicted_goals
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "poisson_deviance": poisson_deviance
    }


# ============================================================
# 14. TEST ALPHA VALUES
# ============================================================

comparison_rows = []

print("\n" + "=" * 80)
print("POISSON MODEL VALIDATION")
print("=" * 80)

for alpha in ALPHA_VALUES:

    print(f"\nTraining models with alpha = {alpha}")

    home_model = create_poisson_pipeline(
        alpha
    )

    away_model = create_poisson_pipeline(
        alpha
    )

    home_model.fit(
        X_train,
        y_home_train
    )

    away_model.fit(
        X_train,
        y_away_train
    )

    home_validation_predictions = (
        home_model.predict(
            X_validation
        )
    )

    away_validation_predictions = (
        away_model.predict(
            X_validation
        )
    )

    home_metrics = evaluate_goal_predictions(
        y_home_validation,
        home_validation_predictions
    )

    away_metrics = evaluate_goal_predictions(
        y_away_validation,
        away_validation_predictions
    )

    combined_mae = (
        home_metrics["mae"] +
        away_metrics["mae"]
    ) / 2.0

    combined_rmse = (
        home_metrics["rmse"] +
        away_metrics["rmse"]
    ) / 2.0

    combined_deviance = (
        home_metrics["poisson_deviance"] +
        away_metrics["poisson_deviance"]
    ) / 2.0

    comparison_rows.append({
        "alpha": alpha,

        "home_validation_mae":
            home_metrics["mae"],

        "away_validation_mae":
            away_metrics["mae"],

        "combined_validation_mae":
            combined_mae,

        "home_validation_rmse":
            home_metrics["rmse"],

        "away_validation_rmse":
            away_metrics["rmse"],

        "combined_validation_rmse":
            combined_rmse,

        "home_validation_poisson_deviance":
            home_metrics["poisson_deviance"],

        "away_validation_poisson_deviance":
            away_metrics["poisson_deviance"],

        "combined_validation_poisson_deviance":
            combined_deviance
    })

    print(
        "Combined validation MAE:",
        round(combined_mae, 4)
    )

    print(
        "Combined validation RMSE:",
        round(combined_rmse, 4)
    )

    print(
        "Combined Poisson deviance:",
        round(combined_deviance, 4)
    )


# ============================================================
# 15. SELECT BEST ALPHA
# ============================================================

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df = (
    comparison_df
    .sort_values(
        by=(
            "combined_validation_"
            "poisson_deviance"
        ),
        ascending=True
    )
    .reset_index(drop=True)
)

best_alpha = float(
    comparison_df.iloc[0]["alpha"]
)

print("\n" + "=" * 80)
print("GOAL-MODEL COMPARISON")
print("=" * 80)

print(
    comparison_df
    .round(4)
    .to_string(index=False)
)

print("\nBest alpha:")
print(best_alpha)


# ============================================================
# 16. RETRAIN ON TRAINING + VALIDATION
# ============================================================

train_validation_df = pd.concat(
    [train_df, validation_df],
    ignore_index=True
)

train_validation_df = (
    train_validation_df
    .sort_values("date")
    .reset_index(drop=True)
)

train_validation_augmented = augment_training_data(
    train_validation_df
)

X_train_validation = (
    train_validation_augmented[
        FEATURE_COLUMNS
    ]
)

y_home_train_validation = (
    train_validation_augmented[
        HOME_TARGET
    ]
)

y_away_train_validation = (
    train_validation_augmented[
        AWAY_TARGET
    ]
)

home_model_for_test = create_poisson_pipeline(
    best_alpha
)

away_model_for_test = create_poisson_pipeline(
    best_alpha
)

home_model_for_test.fit(
    X_train_validation,
    y_home_train_validation
)

away_model_for_test.fit(
    X_train_validation,
    y_away_train_validation
)


# ============================================================
# 17. FINAL TEST PREDICTIONS
# ============================================================

home_test_predictions = clip_expected_goals(
    home_model_for_test.predict(
        X_test
    )
)

away_test_predictions = clip_expected_goals(
    away_model_for_test.predict(
        X_test
    )
)

home_test_metrics = evaluate_goal_predictions(
    y_home_test,
    home_test_predictions
)

away_test_metrics = evaluate_goal_predictions(
    y_away_test,
    away_test_predictions
)

combined_test_mae = (
    home_test_metrics["mae"] +
    away_test_metrics["mae"]
) / 2.0

combined_test_rmse = (
    home_test_metrics["rmse"] +
    away_test_metrics["rmse"]
) / 2.0

combined_test_deviance = (
    home_test_metrics["poisson_deviance"] +
    away_test_metrics["poisson_deviance"]
) / 2.0

print("\n" + "=" * 80)
print("FINAL GOAL-MODEL TEST RESULTS")
print("=" * 80)

print("\nBest alpha:", best_alpha)

print("\nHome-goal model:")

print(
    "MAE:",
    round(home_test_metrics["mae"], 4)
)

print(
    "RMSE:",
    round(home_test_metrics["rmse"], 4)
)

print(
    "Poisson deviance:",
    round(
        home_test_metrics[
            "poisson_deviance"
        ],
        4
    )
)

print("\nAway-goal model:")

print(
    "MAE:",
    round(away_test_metrics["mae"], 4)
)

print(
    "RMSE:",
    round(away_test_metrics["rmse"], 4)
)

print(
    "Poisson deviance:",
    round(
        away_test_metrics[
            "poisson_deviance"
        ],
        4
    )
)

print("\nCombined performance:")

print(
    "Combined MAE:",
    round(combined_test_mae, 4)
)

print(
    "Combined RMSE:",
    round(combined_test_rmse, 4)
)

print(
    "Combined Poisson deviance:",
    round(combined_test_deviance, 4)
)


# ============================================================
# 18. SIMPLE MEAN-GOAL BASELINE
# ============================================================

mean_home_goals = (
    train_validation_df[
        HOME_TARGET
    ].mean()
)

mean_away_goals = (
    train_validation_df[
        AWAY_TARGET
    ].mean()
)

baseline_home_predictions = np.full(
    len(test_df),
    mean_home_goals
)

baseline_away_predictions = np.full(
    len(test_df),
    mean_away_goals
)

baseline_home_mae = mean_absolute_error(
    y_home_test,
    baseline_home_predictions
)

baseline_away_mae = mean_absolute_error(
    y_away_test,
    baseline_away_predictions
)

baseline_combined_mae = (
    baseline_home_mae +
    baseline_away_mae
) / 2.0

print("\nMean-goal baseline:")

print(
    "Baseline combined MAE:",
    round(baseline_combined_mae, 4)
)


# ============================================================
# 19. SCORELINE ACCURACY
# ============================================================

rounded_home_goals = np.rint(
    home_test_predictions
).astype(int)

rounded_away_goals = np.rint(
    away_test_predictions
).astype(int)

exact_score_accuracy = np.mean(
    (
        rounded_home_goals ==
        y_home_test.to_numpy()
    )
    &
    (
        rounded_away_goals ==
        y_away_test.to_numpy()
    )
)

actual_goal_differences = (
    y_home_test.to_numpy() -
    y_away_test.to_numpy()
)

predicted_goal_differences = (
    home_test_predictions -
    away_test_predictions
)

actual_outcomes = np.sign(
    actual_goal_differences
)

predicted_outcomes = np.sign(
    predicted_goal_differences
)

expected_outcome_accuracy = np.mean(
    actual_outcomes ==
    predicted_outcomes
)

print(
    "\nRounded exact-score accuracy:",
    f"{exact_score_accuracy:.2%}"
)

print(
    "Expected-goal outcome accuracy:",
    f"{expected_outcome_accuracy:.2%}"
)


# ============================================================
# 20. NEUTRAL-MATCH SYMMETRY DIAGNOSTIC
# ============================================================

def calculate_goal_symmetry_error(
    home_goal_model,
    away_goal_model,
    evaluation_df
):
    """
    Compare expected goals before and after reversing neutral
    team orientation.
    """

    neutral_original = evaluation_df[
        evaluation_df["neutral"] == 1
    ].copy()

    neutral_original = (
        neutral_original
        .reset_index(drop=True)
    )

    if neutral_original.empty:
        return {
            "matches": 0,
            "mean_error": np.nan,
            "maximum_error": np.nan
        }

    neutral_mirrored = make_mirrored_neutral_rows(
        neutral_original
    )

    original_home_expected = clip_expected_goals(
        home_goal_model.predict(
            neutral_original[
                FEATURE_COLUMNS
            ]
        )
    )

    original_away_expected = clip_expected_goals(
        away_goal_model.predict(
            neutral_original[
                FEATURE_COLUMNS
            ]
        )
    )

    mirrored_home_expected = clip_expected_goals(
        home_goal_model.predict(
            neutral_mirrored[
                FEATURE_COLUMNS
            ]
        )
    )

    mirrored_away_expected = clip_expected_goals(
        away_goal_model.predict(
            neutral_mirrored[
                FEATURE_COLUMNS
            ]
        )
    )

    home_to_mirrored_away_error = np.abs(
        original_home_expected -
        mirrored_away_expected
    )

    away_to_mirrored_home_error = np.abs(
        original_away_expected -
        mirrored_home_expected
    )

    combined_errors = np.concatenate([
        home_to_mirrored_away_error,
        away_to_mirrored_home_error
    ])

    return {
        "matches":
            len(neutral_original),

        "mean_error":
            float(np.mean(combined_errors)),

        "maximum_error":
            float(np.max(combined_errors))
    }


symmetry_results = calculate_goal_symmetry_error(
    home_goal_model=home_model_for_test,
    away_goal_model=away_model_for_test,
    evaluation_df=test_df
)

print("\n" + "=" * 80)
print("NEUTRAL GOAL-MODEL SYMMETRY")
print("=" * 80)

print(
    "\nNeutral test matches:",
    symmetry_results["matches"]
)

print(
    "Mean expected-goal mismatch:",
    round(
        symmetry_results["mean_error"],
        4
    )
)

print(
    "Maximum expected-goal mismatch:",
    round(
        symmetry_results["maximum_error"],
        4
    )
)


# ============================================================
# 21. SAVE TEST PREDICTIONS
# ============================================================

test_output = test_df[
    [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "neutral"
    ]
].copy()

test_output[
    "predicted_home_expected_goals"
] = home_test_predictions

test_output[
    "predicted_away_expected_goals"
] = away_test_predictions

test_output[
    "rounded_home_score"
] = rounded_home_goals

test_output[
    "rounded_away_score"
] = rounded_away_goals

test_output.to_csv(
    TEST_PREDICTIONS_OUTPUT_PATH,
    index=False
)


# ============================================================
# 22. TRAIN PRODUCTION MODELS ON ALL DATA
# ============================================================

all_augmented_df = augment_training_data(
    df
)

X_all = all_augmented_df[
    FEATURE_COLUMNS
]

y_home_all = all_augmented_df[
    HOME_TARGET
]

y_away_all = all_augmented_df[
    AWAY_TARGET
]

production_home_model = create_poisson_pipeline(
    best_alpha
)

production_away_model = create_poisson_pipeline(
    best_alpha
)

production_home_model.fit(
    X_all,
    y_home_all
)

production_away_model.fit(
    X_all,
    y_away_all
)


# ============================================================
# 23. SAVE MODEL BUNDLE
# ============================================================

goal_model_bundle = {

    "home_goal_model":
        production_home_model,

    "away_goal_model":
        production_away_model,

    "feature_columns":
        FEATURE_COLUMNS,

    "best_alpha":
        best_alpha,

    "minimum_expected_goals":
        MIN_EXPECTED_GOALS,

    "maximum_expected_goals":
        MAX_EXPECTED_GOALS,

    "training_start_date":
        str(df["date"].min().date()),

    "training_end_date":
        str(df["date"].max().date()),

    "original_training_rows":
        len(df),

    "augmented_training_rows":
        len(all_augmented_df),

    "neutral_mirror_rows":
        int(
            all_augmented_df[
                "is_mirrored"
            ].sum()
        )
}

joblib.dump(
    goal_model_bundle,
    MODEL_OUTPUT_PATH
)


# ============================================================
# 24. SAVE COMPARISON TABLE
# ============================================================

comparison_df[
    "selected_alpha"
] = (
    comparison_df["alpha"] ==
    best_alpha
)

comparison_df.to_csv(
    COMPARISON_OUTPUT_PATH,
    index=False
)


# ============================================================
# 25. SAVE REPORT
# ============================================================

with open(
    REPORT_OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as report_file:

    report_file.write(
        "WORLD CUP POISSON GOAL-MODEL REPORT\n"
    )

    report_file.write("=" * 70 + "\n\n")

    report_file.write(
        f"Best alpha: {best_alpha}\n\n"
    )

    report_file.write(
        f"Home-goal test MAE: "
        f"{home_test_metrics['mae']:.4f}\n"
    )

    report_file.write(
        f"Away-goal test MAE: "
        f"{away_test_metrics['mae']:.4f}\n"
    )

    report_file.write(
        f"Combined test MAE: "
        f"{combined_test_mae:.4f}\n"
    )

    report_file.write(
        f"Combined test RMSE: "
        f"{combined_test_rmse:.4f}\n"
    )

    report_file.write(
        f"Combined Poisson deviance: "
        f"{combined_test_deviance:.4f}\n"
    )

    report_file.write(
        f"Mean-goal baseline MAE: "
        f"{baseline_combined_mae:.4f}\n"
    )

    report_file.write(
        f"Rounded exact-score accuracy: "
        f"{exact_score_accuracy:.4%}\n"
    )

    report_file.write(
        f"Expected-goal outcome accuracy: "
        f"{expected_outcome_accuracy:.4%}\n\n"
    )

    report_file.write(
        "Neutral-match symmetry:\n"
    )

    report_file.write(
        f"Matches: "
        f"{symmetry_results['matches']}\n"
    )

    report_file.write(
        f"Mean expected-goal mismatch: "
        f"{symmetry_results['mean_error']:.4f}\n"
    )

    report_file.write(
        f"Maximum expected-goal mismatch: "
        f"{symmetry_results['maximum_error']:.4f}\n"
    )


# ============================================================
# 26. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("GOAL-MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 80)

print("\nBest alpha:")
print(best_alpha)

print("\nSaved goal models:")
print(MODEL_OUTPUT_PATH)

print("\nSaved comparison table:")
print(COMPARISON_OUTPUT_PATH)

print("\nSaved test predictions:")
print(TEST_PREDICTIONS_OUTPUT_PATH)

print("\nSaved report:")
print(REPORT_OUTPUT_PATH)
