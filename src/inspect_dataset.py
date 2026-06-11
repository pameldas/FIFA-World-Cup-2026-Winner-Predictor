import pandas as pd
from pathlib import Path

# Dataset location
file_path = Path(r"D:\wc data\results.csv")

# Output location
output_path = Path(r"D:\wc data\cleaned_results.csv")

# Check that the dataset exists
if not file_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at:\n{file_path}"
    )

# Load dataset
df = pd.read_csv(file_path)

print("=" * 70)
print("DATASET SUCCESSFULLY LOADED")
print("=" * 70)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# Convert date column
df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

print("\n" + "=" * 70)
print("DATE INFORMATION")
print("=" * 70)

print("Earliest match:", df["date"].min())
print("Latest match:", df["date"].max())
print("Invalid dates:", df["date"].isna().sum())


# Check teams
all_teams = pd.concat(
    [df["home_team"], df["away_team"]],
    ignore_index=True
)

print("\n" + "=" * 70)
print("TEAM INFORMATION")
print("=" * 70)

print("Number of unique teams:", all_teams.nunique())

main_teams = [
    "Argentina",
    "Brazil",
    "France",
    "Spain",
    "England",
    "Germany"
]

available_teams = set(all_teams.dropna().unique())

print("\nSelected teams:")

for team in main_teams:
    if team in available_teams:
        print(f"{team}: Found")
    else:
        print(f"{team}: Not found")


# Create match-result column
def determine_result(row):
    if row["home_score"] > row["away_score"]:
        return "Home Win"
    elif row["home_score"] < row["away_score"]:
        return "Away Win"
    else:
        return "Draw"


df["result"] = df.apply(determine_result, axis=1)

print("\n" + "=" * 70)
print("MATCH RESULT DISTRIBUTION")
print("=" * 70)

print(df["result"].value_counts())

print("\nResult percentages:")
print(
    df["result"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# Select recent matches
model_df = df[
    (df["date"] >= "2010-01-01") &
    (df["date"] <= "2026-06-10")
].copy()

# Remove rows missing important information
required_columns = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral"
]

model_df = model_df.dropna(subset=required_columns)

# Remove duplicates and sort chronologically
model_df = (
    model_df
    .drop_duplicates()
    .sort_values("date")
    .reset_index(drop=True)
)

print("\n" + "=" * 70)
print("CLEANED DATASET")
print("=" * 70)

print("Number of selected matches:", len(model_df))
print("Starting date:", model_df["date"].min())
print("Ending date:", model_df["date"].max())

print("\nFirst cleaned row:")
print(model_df.head(1))

print("\nLast cleaned row:")
print(model_df.tail(1))


# Save cleaned dataset
model_df.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("Cleaned dataset saved to:")
print(output_path)
