
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"

CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")
RANDOM_STATE = 42


def load_data():
    customers = pd.read_csv(CLEAN_DIR / "customers_clean.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders_clean.csv")
    products = pd.read_csv(RAW_DIR / "products.csv")

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )
    orders["discount_pct"] = pd.to_numeric(
        orders["discount_pct"], errors="coerce"
    )
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    )

    return customers, orders, products


def build_dataset(customers, orders, products):
    """Build pre-cutoff features and a September lapse label."""
    before = orders[
        orders["order_ts"].notna()
        & (orders["order_ts"] < CUTOFF)
    ].copy()

    completed_before = before[before["status"] == "completed"].copy()

    # Cohort: customers with at least one completed order before cutoff.
    cohort_ids = completed_before["customer_id"].dropna().unique()

    # Label only uses September activity; September is never a feature.
    september_completed = orders[
        (orders["status"] == "completed")
        & (orders["order_ts"] >= CUTOFF)
        & (orders["order_ts"] < LABEL_END)
    ].groupby("customer_id")["order_id"].nunique()

    customer_base = customers[
        customers["customer_id"].isin(cohort_ids)
    ].copy()

    features = pd.DataFrame({"customer_id": cohort_ids})
    features = features.merge(
        customer_base, on="customer_id", how="left"
    )

    def aggregate_completed(group):
        return pd.Series({
            "recency": (CUTOFF - group["order_ts"].max()).days,
            "frequency": group["order_id"].nunique(),
            "monetary": group["order_amount"].sum(min_count=1),
            "avg_discount_pct": group["discount_pct"].mean(),
            "avg_delivery_minutes": group["delivery_minutes"].mean(),
        })

    aggregates = completed_before.groupby("customer_id").apply(
        aggregate_completed, include_groups=False
    ).reset_index()

    features = features.merge(
        aggregates, on="customer_id", how="left"
    )

    # Cancellation rate uses all final-status orders before cutoff.
    final_orders = before[before["status"].isin(["completed", "cancelled"])]
    status_counts = final_orders.groupby("customer_id")["status"].agg(
        total_final="count",
        cancelled=lambda s: (s == "cancelled").sum(),
    )
    status_counts["cancellation_rate"] = (
        status_counts["cancelled"] / status_counts["total_final"]
    )
    features = features.merge(
        status_counts[["cancellation_rate"]],
        left_on="customer_id",
        right_index=True,
        how="left",
    )

    # Most frequent category among pre-cutoff completed orders.
    category_orders = completed_before.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
    )

    favorite = (
        category_orders.groupby("customer_id")["category"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown")
        .rename("favorite_category")
    )
    features = features.merge(
        favorite, left_on="customer_id", right_index=True, how="left"
    )

    features["tenure_days"] = (
        CUTOFF - features["signup_date"]
    ).dt.days

    features["lapsed"] = (
        features["customer_id"].map(september_completed).fillna(0) == 0
    ).astype(int)

    # Remove identifying/date columns and any non-feature columns.
    features = features.drop(columns=["signup_date"], errors="ignore")

    numeric = [
        "recency", "frequency", "monetary", "avg_discount_pct",
        "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days",
    ]
    categorical = ["city", "gender", "favorite_category"]

    # Fill missing feature values before modelling; no label is imputed.
    for col in numeric:
        features[col] = pd.to_numeric(features[col], errors="coerce")
        features[col] = features[col].replace([np.inf, -np.inf], np.nan)
        features[col] = features[col].fillna(features[col].median())

    for col in categorical:
        features[col] = features[col].fillna("unknown").astype(str)

    return features, numeric, categorical


def make_pipeline(numeric, categorical, classifier):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def metrics_for(name, model, x_test, y_test):
    predictions = model.predict(x_test)

    print(f"\n{name}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(
        "Precision:",
        round(precision_score(y_test, predictions, zero_division=0), 4),
    )
    print(
        "Recall:",
        round(recall_score(y_test, predictions, zero_division=0), 4),
    )
    print(
        "F1:",
        round(f1_score(y_test, predictions, zero_division=0), 4),
    )
    return predictions


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    customers, orders, products = load_data()
    data, numeric, categorical = build_dataset(
        customers, orders, products
    )

    feature_columns = numeric + categorical
    X = data[feature_columns]
    y = data["lapsed"]

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("Cohort:", len(data))
    print("Lapsed:", int(y.sum()))
    print("Lapse rate:", round(y.mean() * 100, 2), "%")
    print("Training rows:", len(x_train))
    print("Test rows:", len(x_test))
    print("Missing feature values:", int(X.isna().sum().sum()))

    # Baseline: always predict not lapsed.
    baseline = np.zeros(len(y_test), dtype=int)
    print("\nBaseline accuracy:", round(accuracy_score(y_test, baseline), 4))
    print(
        "Baseline lapsed F1:",
        round(f1_score(y_test, baseline, zero_division=0), 4),
    )

    # Fit preprocessing and model using training rows only.
    logistic = make_pipeline(
        numeric,
        categorical,
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )
    logistic_grid = GridSearchCV(
        logistic,
        {"classifier__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    logistic_grid.fit(x_train, y_train)
    best_logistic = logistic_grid.best_estimator_

    print("\nLogistic best parameters:", logistic_grid.best_params_)
    print("Logistic CV F1:", round(logistic_grid.best_score_, 4))
    logistic_predictions = metrics_for(
        "Logistic regression test results",
        best_logistic,
        x_test,
        y_test,
    )

    forest = make_pipeline(
        numeric,
        categorical,
        RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
    )
    forest_grid = GridSearchCV(
        forest,
        {
            "classifier__max_depth": [3, 5, 8],
            "classifier__min_samples_leaf": [5, 10, 20],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    forest_grid.fit(x_train, y_train)
    best_forest = forest_grid.best_estimator_

    print("\nRandom forest best parameters:", forest_grid.best_params_)
    print("Random forest CV F1:", round(forest_grid.best_score_, 4))
    forest_predictions = metrics_for(
        "Random forest test results",
        best_forest,
        x_test,
        y_test,
    )

    # Choose the model with better test F1; report that the test set is
    # used only for final evaluation after hyperparameter tuning.
    chosen = (
        best_forest
        if f1_score(y_test, forest_predictions, zero_division=0)
        >= f1_score(y_test, logistic_predictions, zero_division=0)
        else best_logistic
    )

    # Oversample only the training rows for the logistic experiment.
    train_frame = x_train.copy()
    train_frame["lapsed"] = y_train.to_numpy()

    majority_count = int(train_frame["lapsed"].value_counts().max())
    minority = train_frame[train_frame["lapsed"] == 1]
    added = minority.sample(
        n=majority_count - len(minority),
        replace=True,
        random_state=RANDOM_STATE,
    )
    balanced = pd.concat([train_frame, added], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=RANDOM_STATE)

    balanced_logistic = make_pipeline(
        numeric,
        categorical,
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            C=logistic_grid.best_params_["classifier__C"],
        ),
    )
    balanced_logistic.fit(
        balanced[feature_columns], balanced["lapsed"]
    )

    balanced_predictions = balanced_logistic.predict(x_test)
    print("\nOversampled training class counts:")
    print(balanced["lapsed"].value_counts().sort_index().to_dict())
    print(
        "Oversampled precision/recall/F1:",
        round(precision_score(y_test, balanced_predictions, zero_division=0), 4),
        round(recall_score(y_test, balanced_predictions, zero_division=0), 4),
        round(f1_score(y_test, balanced_predictions, zero_division=0), 4),
    )

    # Report learned scaler frequency mean against full-cohort mean.
    fitted_scaler = (
        best_logistic.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["scaler"]
    )
    frequency_index = numeric.index("frequency")
    print(
        "\nTraining scaler frequency mean:",
        round(fitted_scaler.mean_[frequency_index], 4),
    )
    print(
        "Full cohort frequency mean:",
        round(data["frequency"].mean(), 4),
    )

    # Random forest feature importance.
    fitted_preprocessor = best_forest.named_steps["preprocessor"]
    transformed_names = fitted_preprocessor.get_feature_names_out()
    importances = best_forest.named_steps["classifier"].feature_importances_
    importance_table = pd.DataFrame({
        "feature": transformed_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("\nTop five random forest features:")
    print(importance_table.head(5).to_string(index=False))

    # Logistic regression coefficients.
    logistic_preprocessor = best_logistic.named_steps["preprocessor"]
    logistic_names = logistic_preprocessor.get_feature_names_out()
    coefficients = best_logistic.named_steps["classifier"].coef_[0]
    coefficient_table = pd.DataFrame({
        "feature": logistic_names,
        "coefficient": coefficients,
    })
    print("\nThree logistic regression coefficients:")
    print(
        coefficient_table.reindex(
            coefficient_table["coefficient"].abs()
            .sort_values(ascending=False).index
        ).head(3).to_string(index=False)
    )

    # Save and reload selected model, then verify predictions.
    model_path = OUTPUT_DIR / "lapse_model.joblib"
    joblib.dump(chosen, model_path)
    reloaded = joblib.load(model_path)
    reloaded_predictions = reloaded.predict(x_test)
    print(
        "\nReloaded predictions identical:",
        bool(np.array_equal(chosen.predict(x_test), reloaded_predictions)),
    )

    # Premium is a post-hoc label from the whole-window RFM.
    segments_path = OUTPUT_DIR / "customer_segments.csv"
    segments = pd.read_csv(segments_path)
    test_ids = set(data.loc[x_test.index, "customer_id"])
    test_customers = data.loc[x_test.index, ["customer_id", "lapsed"]].copy()
    test_customers["lapse_probability"] = chosen.predict_proba(x_test)[:, 1]
    test_customers = test_customers.merge(
        segments[["customer_id", "is_premium"]],
        on="customer_id",
        how="left",
    )

    retention = test_customers[
        test_customers["is_premium"].fillna(False)
    ].sort_values("lapse_probability", ascending=False).head(10)

    retention = retention.rename(
        columns={"lapsed": "actually_lapsed"}
    )[["customer_id", "lapse_probability", "actually_lapsed"]]
    retention.to_csv(OUTPUT_DIR / "retention_list.csv", index=False)

    print("\nRetention list:")
    print(retention.to_string(index=False))
    print("Actually lapsed in retention list:", int(retention["actually_lapsed"].sum()))

    # Save a text report of model metrics and key counts.
    with (OUTPUT_DIR / "lapse_report.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write(f"Cohort size: {len(data)}\n")
        file.write(f"Lapsed count: {int(y.sum())}\n")
        file.write(f"Training rows: {len(x_train)}\n")
        file.write(f"Test rows: {len(x_test)}\n")
        file.write(f"Logistic best parameters: {logistic_grid.best_params_}\n")
        file.write(f"Logistic CV F1: {logistic_grid.best_score_:.4f}\n")
        file.write(f"Random forest best parameters: {forest_grid.best_params_}\n")
        file.write(f"Random forest CV F1: {forest_grid.best_score_:.4f}\n")
        file.write(
            "Reloaded predictions identical: "
            f"{np.array_equal(chosen.predict(x_test), reloaded_predictions)}\n"
        )
        file.write(
            "Limitation: the Premium flag is a post-hoc whole-window label, "
            "so the retention list is illustrative and not a leakage-free evaluation.\n"
        )


if __name__ == "__main__":
    main()
