
from pathlib import Path
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"


def main():
    if not (CLEAN_DIR / "customers_clean.csv").exists():
        raise FileNotFoundError("Run src/clean_data.py first.")

    if not (CLEAN_DIR / "orders_clean.csv").exists():
        raise FileNotFoundError("Cleaned orders CSV not found.")

    customers = pd.read_csv(CLEAN_DIR / "customers_clean.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders_clean.csv")
    products = pd.read_csv(ROOT / "data" / "raw" / "products.csv")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        customers.to_sql("customers", conn, if_exists="replace", index=False)
        products.to_sql("products", conn, if_exists="replace", index=False)
        orders.to_sql("orders", conn, if_exists="replace", index=False)

        # Create indexes to speed up common joins and filters.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_orders_customer "
            "ON orders(customer_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_orders_product "
            "ON orders(product_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_orders_status "
            "ON orders(status)"
        )

        for table in ["customers", "products", "orders"]:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            print(f"{table}: {count} rows")

    print(f"Database created: {DB_PATH}")


if __name__ == "__main__":
    main()
