# FIFA World Cup 2026 Winner Predictor

A Python-based sports analytics project for predicting the FIFA World Cup 2026 winner using Elo ratings, Poisson goal modeling, Random Forest evaluation, and Monte Carlo tournament simulation.

## Project Overview

This project simulates the FIFA World Cup 2026 tournament and estimates possible match outcomes, knockout progression, and champion probabilities. The framework combines historical international football match data with machine learning and statistical modeling techniques.

The main goal of this project is to build a complete prediction pipeline, from dataset cleaning and feature engineering to model evaluation and full tournament simulation.

## Methodology

The prediction framework uses the following steps:

1. **Data Cleaning**

   * Historical international football match results were cleaned and filtered.
   * Duplicate matches and missing values were handled.
   * Match data before the prediction cutoff date was used to prevent future data leakage.

2. **Feature Engineering**

   * Elo ratings were calculated for each team.
   * Recent form features were created using the previous five matches.
   * Features included win rate, points per game, goals scored, goals conceded, and goal difference.

3. **Match Prediction**

   * A Random Forest classifier was trained to predict match outcomes:

     * Home Win
     * Draw
     * Away Win

4. **Poisson Goal Modeling**

   * Poisson regression models were used to estimate expected goals for both teams.
   * These expected goals were used to generate match-score probabilities.

5. **Hybrid Model Selection**

   * Different Random Forest and Poisson weight combinations were tested.
   * The final selected configuration used:

     * Poisson Model: 100%
     * Random Forest Model: 0%

6. **Tournament Simulation**

   * The complete FIFA World Cup 2026 structure was simulated.
   * Group stage, knockout rounds, extra time, penalty shootouts, and final winner selection were included.
   * Monte Carlo simulation was used to estimate champion probabilities.

## Dataset

The project uses historical international football match data containing:

* Match date
* Home team
* Away team
* Home score
* Away score
* Tournament name
* Venue/country information
* Neutral venue indicator

Additional engineered datasets include:

* Cleaned match results
* Machine learning feature table
* Latest team Elo ratings
* Model predictions
* Final hybrid evaluation results

## Key Results

Final test evaluation:

| Metric                       |  Value |
| ---------------------------- | -----: |
| Test Accuracy                | 59.65% |
| Macro F1 Score               | 0.4388 |
| Weighted F1 Score            | 0.5142 |
| Log Loss                     | 0.8736 |
| Combined Goal MAE            | 0.9508 |
| Combined Goal RMSE           | 1.2491 |
| Rounded Exact Score Accuracy | 10.44% |

Final selected model:

| Component                | Weight |
| ------------------------ | -----: |
| Poisson Goal Model       |   100% |
| Random Forest Classifier |     0% |

## Project Structure

```text
FIFA-World-Cup-2026-Winner-Predictor/
│
├── data/          # Datasets and processed files
├── models/        # Trained model files
├── reports/       # Evaluation reports and metrics
├── src/           # Python source code
├── README.md
├── LICENSE
└── requirements.txt
```

## How to Run

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

Then run the main simulation engine:

```bash
python src/world_cup_2026_final_monte_carlo_engine.py
```

You can also run individual pipeline files:

```bash
python src/build_features.py
python src/train_models.py
python src/train_goal_models.py
python src/tune_hybrid_weights.py
python src/evaluate_final_hybrid_test.py
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Poisson Regression
* Random Forest
* Monte Carlo Simulation

## Limitations

This project is a simulation-based prediction framework, not a guaranteed forecast. Football outcomes depend on many unpredictable factors such as injuries, tactical changes, player availability, form fluctuations, red cards, and real-time match conditions.

The model should be interpreted as a sports analytics experiment rather than a final deterministic prediction.

## Author

BPS
