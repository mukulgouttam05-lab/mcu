import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. READ CSV FILE
# ============================================================

file_path = "mcu.csv"

df = pd.read_csv(file_path)

print("CSV loaded successfully!")
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. PREPARE DATA
# ============================================================

# Convert release_date into a date
df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

# Create release year
df["release_year"] = df["release_date"].dt.year


# ============================================================
# 3. DEFINE TARGET
# ============================================================

# This is what our model will predict
target = "worldwide_box_office"

# Remove rows where the target is missing
df = df.dropna(subset=[target])


# ============================================================
# 4. FEATURES
# ============================================================

features = [
    "mcu_phase",
    "release_year",
    "tomato_meter",
    "audience_score",
    "movie_duration",
    "production_budget",
    "opening_weekend",
    "domestic_box_office"
]

X = df[features]
y = df[target]


# ============================================================
# 5. CATEGORICAL + NUMERICAL FEATURES
# ============================================================

categorical_features = [
    "mcu_phase"
]

numeric_features = [
    "release_year",
    "tomato_meter",
    "audience_score",
    "movie_duration",
    "production_budget",
    "opening_weekend",
    "domestic_box_office"
]


# ============================================================
# 6. DATA PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "missing_values",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
    (
        "missing_values",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoding",
        OneHotEncoder(handle_unknown="ignore")
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ============================================================
# 7. CREATE MACHINE LEARNING MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=2
)


# ============================================================
# 8. CREATE COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),
    (
        "model",
        model
    )
])


# ============================================================
# 9. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model trained successfully!")


# ============================================================
# 11. TEST MODEL
# ============================================================

predictions = pipeline.predict(X_test)


mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("MAE :", f"${mae:,.0f}")
print("RMSE:", f"${rmse:,.0f}")
print("R²  :", round(r2, 3))


# ============================================================
# 12. TRAIN FINAL MODEL ON ALL DATA
# ============================================================

pipeline.fit(
    X,
    y
)


# ============================================================
# 13. PREDICTION FUNCTION
# ============================================================

def predict_movie():

    print("\n==============================")
    print("ENTER NEW MCU MOVIE DATA")
    print("==============================")

    mcu_phase = input(
        "MCU Phase (example: Phase 3): "
    )

    release_year = int(
        input("Release year: ")
    )

    tomato_meter = float(
        input("Rotten Tomatoes score: ")
    )

    audience_score = float(
        input("Audience score: ")
    )

    movie_duration = float(
        input("Movie duration (minutes): ")
    )

    production_budget = float(
        input("Production budget ($): ")
    )

    opening_weekend = float(
        input("Opening weekend ($): ")
    )

    domestic_box_office = float(
        input("Domestic box office ($): ")
    )


    # Create dataframe for new movie

    new_movie = pd.DataFrame({
        "mcu_phase": [mcu_phase],
        "release_year": [release_year],
        "tomato_meter": [tomato_meter],
        "audience_score": [audience_score],
        "movie_duration": [movie_duration],
        "production_budget": [production_budget],
        "opening_weekend": [opening_weekend],
        "domestic_box_office": [domestic_box_office]
    })


    # Make prediction

    prediction = pipeline.predict(
        new_movie
    )[0]


    print("\n==============================")
    print("PREDICTION")
    print("==============================")

    print(
        "Predicted worldwide box office:",
        f"${prediction:,.0f}"
    )


# ============================================================
# 14. RUN PREDICTION
# ============================================================

while True:

    predict_movie()

    again = input(
        "\nPredict another movie? (yes/no): "
    )

    if again.lower() != "yes":
        print("\nProgram finished.")
        break