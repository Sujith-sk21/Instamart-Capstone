
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    orders = pd.read_csv(CLEAN_DIR / "orders_clean.csv")

    completed = orders.loc[
        orders["status"] == "completed"
    ].copy()

    amounts = pd.to_numeric(
        completed["order_amount"], errors="coerce"
    ).dropna()

    discounts = pd.to_numeric(
        completed["discount_pct"], errors="coerce"
    )

    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 35)
    print(f"Completed orders: {len(completed)}")
    print(f"Valid completed-order amounts: {len(amounts)}")
    print(f"Mean order amount: INR {amounts.mean():.2f}")
    print(f"Median order amount: INR {amounts.median():.2f}")
    print(f"Standard deviation: INR {amounts.std():.2f}")
    print(f"Minimum order amount: INR {amounts.min():.2f}")
    print(f"Maximum order amount: INR {amounts.max():.2f}")
    print(f"Skewness: {amounts.skew():.4f}")

    q1 = amounts.quantile(0.25)
    q3 = amounts.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outliers = amounts[
        (amounts < lower_fence) | (amounts > upper_fence)
    ]

    print(f"First quartile (Q1): INR {q1:.2f}")
    print(f"Third quartile (Q3): INR {q3:.2f}")
    print(f"Lower outlier fence: INR {lower_fence:.2f}")
    print(f"Upper outlier fence: INR {upper_fence:.2f}")
    print(f"Outlier count: {len(outliers)}")
    print(f"Lower fence outliers: {(amounts < lower_fence).sum()}")
    print(f"Upper fence outliers: {(amounts > upper_fence).sum()}")

    correlation = pd.DataFrame({
        "order_amount": pd.to_numeric(
            completed["order_amount"], errors="coerce"
        ),
        "discount_pct": discounts,
    }).corr()

    amount_discount_corr = correlation.loc[
        "order_amount", "discount_pct"
    ]

    print(
        "Order amount / discount correlation: "
        f"{amount_discount_corr:.4f}"
    )

    with open(
        OUTPUT_DIR / "eda_report.txt", "w", encoding="utf-8"
    ) as file:
        file.write("EXPLORATORY DATA ANALYSIS\n")
        file.write("=" * 35 + "\n")
        file.write(f"Completed orders: {len(completed)}\n")
        file.write(f"Valid order amounts: {len(amounts)}\n")
        file.write(f"Mean: INR {amounts.mean():.2f}\n")
        file.write(f"Median: INR {amounts.median():.2f}\n")
        file.write(f"Standard deviation: INR {amounts.std():.2f}\n")
        file.write(f"Skewness: {amounts.skew():.4f}\n")
        file.write(f"Q1: INR {q1:.2f}\n")
        file.write(f"Q3: INR {q3:.2f}\n")
        file.write(f"Lower fence: INR {lower_fence:.2f}\n")
        file.write(f"Upper fence: INR {upper_fence:.2f}\n")
        file.write(f"Outlier count: {len(outliers)}\n")
        file.write(
            f"Amount/discount correlation: {amount_discount_corr:.4f}\n"
        )

    # Plot 1: Order amount distribution.
    plt.figure(figsize=(9, 5))
    plt.hist(amounts, bins=40, edgecolor="black")
    plt.title("Distribution of Completed Order Amounts")
    plt.xlabel("Order amount (INR)")
    plt.ylabel("Number of orders")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "order_amount_distribution.png", dpi=150)
    plt.close()

    # Plot 2: Box plot and IQR outliers.
    plt.figure(figsize=(9, 4))
    plt.boxplot(amounts, vert=False)
    plt.title("Completed Order Amounts and Outliers")
    plt.xlabel("Order amount (INR)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "order_amount_boxplot.png", dpi=150)
    plt.close()

    # Plot 3: Order amount versus discount.
    plot_data = pd.DataFrame({
        "order_amount": pd.to_numeric(
            completed["order_amount"], errors="coerce"
        ),
        "discount_pct": discounts,
    }).dropna()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        plot_data["discount_pct"],
        plot_data["order_amount"],
        alpha=0.45,
    )
    plt.title("Order Amount versus Discount")
    plt.xlabel("Discount (%)")
    plt.ylabel("Order amount (INR)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "amount_vs_discount.png", dpi=150)
    plt.close()

    print(f"\nReport saved to: {OUTPUT_DIR / 'eda_report.txt'}")
    print(f"Charts saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
