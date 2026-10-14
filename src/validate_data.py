
from pathlib import Path
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"


def main():
    customers = pd.read_csv(CLEAN_DIR / "customers_clean.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders_clean.csv")
    raw_orders = pd.read_csv(RAW_DIR / "orders.csv")

    # Check expected dataset sizes.
    assert len(customers) == 1200, "Unexpected customer count"
    assert len(orders) == 12309, "Unexpected cleaned order count"
    assert raw_orders.duplicated().sum() == 119, "Unexpected duplicate count"

    # Check customer identifiers.
    assert customers["customer_id"].notna().all(), "Missing customer IDs"
    assert customers["customer_id"].is_unique, "Duplicate customer IDs"

    # Check order identifiers and statuses.
    assert orders["order_id"].notna().all(), "Missing order IDs"
    assert orders["order_id"].is_unique, "Duplicate order IDs"
    assert set(orders["status"].unique()) <= {
        "completed", "cancelled"
    }, "Unexpected order status"

    # Check cleaned numeric values.
    assert pd.to_numeric(
        orders["order_amount"], errors="coerce"
    ).notna().sum() == 12197, "Unexpected valid amount count"

    assert not (
        pd.to_numeric(
            orders["delivery_minutes"], errors="coerce"
        ).dropna() < 0
    ).any(), "Negative delivery times remain"

    # Check the SQLite database.
    assert DB_PATH.exists(), "SQLite database not found"

    with sqlite3.connect(DB_PATH) as conn:
        customer_count = conn.execute(
            "SELECT COUNT(*) FROM customers"
        ).fetchone()[0]
        product_count = conn.execute(
            "SELECT COUNT(*) FROM products"
        ).fetchone()[0]
        order_count = conn.execute(
            "SELECT COUNT(*) FROM orders"
        ).fetchone()[0]

    assert customer_count == 1200, "Database customer count mismatch"
    assert product_count == 40, "Database product count mismatch"
    assert order_count == 12309, "Database order count mismatch"

    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
