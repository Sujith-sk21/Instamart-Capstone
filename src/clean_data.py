
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"


def clean_customers():
    """Clean customer records while preserving the raw source file."""
    df = pd.read_csv(RAW_DIR / "customers.csv")

    # Remove exact duplicate records.
    df = df.drop_duplicates().copy()

    # Standardize column names and text.
    df.columns = [col.strip().lower() for col in df.columns]

    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()

    # Standardize city spellings.
    if "city" in df.columns:
        df["city"] = df["city"].str.lower().replace({
            "bangalore": "bengaluru",
            "bengaluru": "bengaluru",
            "bombay": "mumbai",
            "mumbai": "mumbai",
            "madras": "chennai",
            "chennai": "chennai",
            "calcutta": "kolkata",
            "kolkata": "kolkata",
            "new delhi": "delhi",
            "delhi": "delhi",
        }).str.title()

    # Standardize gender values.
    if "gender" in df.columns:
        df["gender"] = df["gender"].str.lower().replace({
            "m": "male",
            "man": "male",
            "f": "female",
            "woman": "female",
            "": pd.NA,
        })
        df["gender"] = df["gender"].fillna("unknown")

    # Convert age to numeric and handle implausible values.
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df.loc[~df["age"].between(10, 100), "age"] = pd.NA
        median_age = df["age"].median()
        df["age"] = df["age"].fillna(median_age).round().astype(int)

    # Fill missing values in other text columns.
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].replace("", pd.NA)
        if col != "gender":
            df[col] = df[col].fillna("unknown")

    return df


def clean_orders():
    """Clean orders without dropping rows for invalid field values."""
    df = pd.read_csv(RAW_DIR / "orders.csv")

    # Remove exact duplicate records.
    df = df.drop_duplicates().copy()

    # Standardize column names and text.
    df.columns = [col.strip().lower() for col in df.columns]

    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()

    # Normalize order status.
    if "status" in df.columns:
        df["status"] = df["status"].str.lower()

    # Convert numeric fields.
    for col in ["order_amount", "discount", "delivery_minutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Exclude pending orders.
    if "status" in df.columns:
        df = df[df["status"] != "pending"].copy()

    # Remove records without an order ID.
    if "order_id" in df.columns:
        df = df[
            df["order_id"].notna()
            & (df["order_id"].astype(str).str.strip() != "")
        ].copy()

    # Replace negative delivery times with missing values.
    if "delivery_minutes" in df.columns:
        df.loc[df["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA

    # Replace negative order amounts with missing values.
    if "order_amount" in df.columns:
        df.loc[df["order_amount"] < 0, "order_amount"] = pd.NA

    # Keep rows with missing amounts or delivery times for validation.
    # Cancelled orders may legitimately have missing delivery times.

    return df


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    customers = clean_customers()
    orders = clean_orders()

    customers.to_csv(CLEAN_DIR / "customers_clean.csv", index=False)
    orders.to_csv(CLEAN_DIR / "orders_clean.csv", index=False)

    print(f"Clean customers: {len(customers)}")
    print(f"Clean orders: {len(orders)}")
    print(f"Saved cleaned files to: {CLEAN_DIR}")


if __name__ == "__main__":
    main()
