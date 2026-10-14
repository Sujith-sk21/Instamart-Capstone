
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FEATURES = ["recency", "frequency", "monetary"]


def load_data():
    customers = pd.read_csv(CLEAN_DIR / "customers_clean.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders_clean.csv")

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )

    return customers, orders


def calculate_rfm(customers, orders):
    completed = orders[
        (orders["status"] == "completed")
        & orders["order_ts"].notna()
        & (orders["order_ts"] < SNAPSHOT)
    ].copy()

    rfm = completed.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )

    rfm["recency"] = (
        SNAPSHOT - rfm["last_order"]
    ).dt.days

    rfm = rfm.reset_index()
    rfm = rfm[[
        "customer_id", "recency", "frequency", "monetary"
    ]]

    rfm["frequency"] = rfm["frequency"].astype(int)
    rfm["monetary"] = rfm["monetary"].round(2)

    return rfm


def evaluate_kmeans(rfm, scaled):
    results = []

    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42,
        )
        labels = model.fit_predict(scaled)

        results.append({
            "k": k,
            "inertia": round(model.inertia_, 2),
            "silhouette": round(
                silhouette_score(scaled, labels), 3
            ),
        })

    return pd.DataFrame(results)


def build_segments(rfm, scaled):
    # K=4 is chosen based on the assignment's reference and business use.
    model = KMeans(
        n_clusters=4,
        n_init=10,
        random_state=42,
    )

    result = rfm.copy()
    result["segment_id"] = model.fit_predict(scaled)

    profiles = result.groupby("segment_id").agg(
        size=("customer_id", "size"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )

    premium_id = profiles["mean_monetary"].idxmax()

    # Cluster IDs are arbitrary labels, not rankings.
    names = {}
    for segment_id, row in profiles.iterrows():
        if segment_id == premium_id:
            names[segment_id] = "Premium"
        elif row["mean_frequency"] >= profiles["mean_frequency"].median():
            names[segment_id] = "Frequent Customers"
        elif row["mean_recency"] >= profiles["mean_recency"].median():
            names[segment_id] = "At-Risk Customers"
        else:
            names[segment_id] = "Growing Customers"

    result["segment_name"] = result["segment_id"].map(names)
    result["is_premium"] = result["segment_id"] == premium_id

    return result, profiles, model, names


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    customers, orders = load_data()
    rfm = calculate_rfm(customers, orders)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[FEATURES])

    evaluation = evaluate_kmeans(rfm, scaled)
    segments, profiles, model, names = build_segments(rfm, scaled)

    # PCA is applied to the scaled RFM matrix.
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(scaled)
    pca_ratios = pca.explained_variance_ratio_

    # Required output columns and order.
    segments[[
        "customer_id", "recency", "frequency", "monetary",
        "segment_id", "segment_name", "is_premium"
    ]].to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)

    evaluation.to_csv(
        OUTPUT_DIR / "clustering_evaluation.csv", index=False
    )
    profiles.to_csv(OUTPUT_DIR / "segment_profiles.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(evaluation["k"], evaluation["inertia"], marker="o")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.set_title("KMeans Elbow")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "kmeans_inertia.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(evaluation["k"], evaluation["silhouette"], marker="o")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette score")
    ax.set_title("KMeans Silhouette Scores")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "kmeans_silhouette.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for segment_id in sorted(segments["segment_id"].unique()):
        mask = segments["segment_id"].to_numpy() == segment_id
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            label=names[segment_id],
            alpha=0.7,
        )
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("Customer Segments (PCA)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "customer_segments_pca.png", dpi=150)
    plt.close(fig)

    with (OUTPUT_DIR / "segmentation_report.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("CUSTOMER SEGMENTATION REPORT\n")
        file.write("============================\n\n")
        file.write(f"Snapshot: {SNAPSHOT}\n")
        file.write(f"RFM customers: {len(rfm)}\n\n")
        file.write("RFM summary:\n")
        file.write(rfm[FEATURES].describe().round(2).to_string())
        file.write("\n\nKMeans evaluation:\n")
        file.write(evaluation.to_string(index=False))
        file.write("\n\nSegment profiles:\n")
        file.write(profiles.round(2).to_string())
        file.write("\n\nSegment names and actions:\n")
        for segment_id, name in names.items():
            if name == "Premium":
                action = "Offer loyalty rewards and personalised upselling."
            elif name == "Frequent Customers":
                action = "Encourage repeat purchases with bundles."
            elif name == "At-Risk Customers":
                action = "Send a targeted win-back offer."
            else:
                action = "Use recommendations to encourage a second purchase."
            file.write(
                f"Cluster {segment_id} ({name}): {action}\n"
            )
        file.write(
            "\nPCA explained variance ratios: "
            + ", ".join(f"{x:.3f}" for x in pca_ratios)
        )
        file.write(
            f"\nTotal PCA variance explained: {pca_ratios.sum():.3f}\n"
        )
        file.write(
            "\nCluster IDs are arbitrary labels, not a ranking.\n"
        )

    print(f"RFM customers: {len(rfm)}")
    print("\nRFM summary:")
    print(rfm[FEATURES].describe().round(2))
    print("\nKMeans evaluation:")
    print(evaluation.to_string(index=False))
    print("\nSegment profiles:")
    print(profiles.round(2).to_string())
    print("\nSegment names:", names)
    print("\nPCA explained variance:", pca_ratios.round(3))
    print(f"Total PCA variance: {pca_ratios.sum():.3f}")
    print(f"\nSaved customer segments to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
