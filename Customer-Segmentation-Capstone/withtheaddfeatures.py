import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error



# File paths


data_path = "/Users/katelynjones/Documents/Online Retail.xlsx"
output_folder = Path("/Users/katelynjones/Documents/masters_program_extended_features")
output_folder.mkdir(parents=True, exist_ok=True)



# Load and clean data


df = pd.read_excel(data_path)

df_clean = df.dropna(
    subset=["CustomerID", "InvoiceNo", "StockCode", "InvoiceDate"]
).copy()

df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")]
df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]

df_clean["CustomerID"] = df_clean["CustomerID"].astype(int).astype(str)
df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]



# RFM creation


snapshot_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df_clean.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]



# Additional customer features


avg_purchase_value = df_clean.groupby("CustomerID").agg({
    "TotalPrice": "sum",
    "InvoiceNo": "nunique"
}).reset_index()

avg_purchase_value["AvgPurchaseValue"] = (
    avg_purchase_value["TotalPrice"] / avg_purchase_value["InvoiceNo"]
)

avg_purchase_value = avg_purchase_value[["CustomerID", "AvgPurchaseValue"]]


product_diversity = df_clean.groupby("CustomerID").agg({
    "StockCode": "nunique"
}).reset_index()

product_diversity.columns = ["CustomerID", "ProductDiversity"]


invoice_dates = (
    df_clean.groupby(["CustomerID", "InvoiceNo"])["InvoiceDate"]
    .min()
    .reset_index()
    .sort_values(["CustomerID", "InvoiceDate"])
)

invoice_dates["DaysBetweenTransactions"] = (
    invoice_dates.groupby("CustomerID")["InvoiceDate"].diff().dt.days
)

avg_days_between = (
    invoice_dates.groupby("CustomerID")["DaysBetweenTransactions"]
    .mean()
    .reset_index()
)

avg_days_between.columns = ["CustomerID", "AvgDaysBetweenTransactions"]

avg_days_between["AvgDaysBetweenTransactions"] = (
    avg_days_between["AvgDaysBetweenTransactions"].fillna(0)
)


customer_features_ext = rfm.merge(avg_purchase_value, on="CustomerID", how="left")
customer_features_ext = customer_features_ext.merge(product_diversity, on="CustomerID", how="left")
customer_features_ext = customer_features_ext.merge(avg_days_between, on="CustomerID", how="left")

customer_features_ext.to_csv(
    output_folder / "extended_customer_features_for_testing.csv",
    index=False
)



# Extended feature summary statistics


extended_stats = customer_features_ext[
    [
        "Recency",
        "Frequency",
        "Monetary",
        "AvgPurchaseValue",
        "ProductDiversity",
        "AvgDaysBetweenTransactions"
    ]
].describe().transpose().round(2)

extended_stats.to_csv(
    output_folder / "extended_features_summary_statistics.csv"
)



# t-SNE BEFORE log transformation


extended_features_raw = customer_features_ext[
    [
        "Recency",
        "Frequency",
        "Monetary",
        "AvgPurchaseValue",
        "ProductDiversity",
        "AvgDaysBetweenTransactions"
    ]
]

scaler_ext_raw = StandardScaler()
extended_scaled_raw = scaler_ext_raw.fit_transform(extended_features_raw)

tsne_raw = TSNE(
    n_components=2,
    random_state=42,
    perplexity=50,
    learning_rate=200,
    max_iter=2000
)

extended_tsne_raw = tsne_raw.fit_transform(extended_scaled_raw)

extended_tsne_raw_df = pd.DataFrame(
    extended_tsne_raw,
    columns=["TSNE1", "TSNE2"]
)

plt.figure(figsize=(8, 6))
plt.scatter(
    extended_tsne_raw_df["TSNE1"],
    extended_tsne_raw_df["TSNE2"],
    c=customer_features_ext["Monetary"],
    alpha=0.7
)

plt.title("t-SNE Visualization (Extended Features - Before Log)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(label="Monetary")
plt.tight_layout()
plt.savefig(output_folder / "extended_tsne_before_log.png", dpi=300)
plt.show()
plt.close()



# Log transformation for clustering


for col in [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgPurchaseValue",
    "ProductDiversity",
    "AvgDaysBetweenTransactions"
]:
    customer_features_ext[f"{col}_log"] = np.log1p(customer_features_ext[col])


extended_log_features = customer_features_ext[
    [
        "Recency_log",
        "Frequency_log",
        "Monetary_log",
        "AvgPurchaseValue_log",
        "ProductDiversity_log",
        "AvgDaysBetweenTransactions_log"
    ]
]

scaler_ext = StandardScaler()
extended_scaled = scaler_ext.fit_transform(extended_log_features)


# t-SNE AFTER log transformation


tsne_log = TSNE(
    n_components=2,
    random_state=42,
    perplexity=50,
    learning_rate=200,
    max_iter=2000
)

extended_tsne_log = tsne_log.fit_transform(extended_scaled)

extended_tsne_log_df = pd.DataFrame(
    extended_tsne_log,
    columns=["TSNE1", "TSNE2"]
)

plt.figure(figsize=(8, 6))
plt.scatter(
    extended_tsne_log_df["TSNE1"],
    extended_tsne_log_df["TSNE2"],
    c=customer_features_ext["Monetary"],
    alpha=0.7
)

plt.title("t-SNE Visualization (Extended Features - After Log)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(label="Monetary")
plt.tight_layout()
plt.savefig(output_folder / "extended_tsne_after_log.png", dpi=300)
plt.show()
plt.close()



# K-means evaluation using extended features


extended_kmeans_results = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(extended_scaled)

    sil_score = silhouette_score(extended_scaled, labels)
    db_score = davies_bouldin_score(extended_scaled, labels)

    extended_kmeans_results.append({
        "Method": "KMeans_ExtendedFeatures",
        "Clusters": k,
        "SilhouetteScore": round(sil_score, 4),
        "DaviesBouldinIndex": round(db_score, 4)
    })

extended_kmeans_results_df = pd.DataFrame(extended_kmeans_results)

extended_kmeans_results_df.to_csv(
    output_folder / "extended_kmeans_evaluation_results.csv",
    index=False
)

print("\nExtended Feature K-means Evaluation Results:")
print(extended_kmeans_results_df)



# Best k clustering using extended features


best_k_ext = extended_kmeans_results_df.loc[
    extended_kmeans_results_df["SilhouetteScore"].idxmax(),
    "Clusters"
]

best_k_ext = int(best_k_ext)

kmeans_ext_best = KMeans(n_clusters=best_k_ext, random_state=42, n_init=10)
customer_features_ext["Cluster_Extended_Best"] = kmeans_ext_best.fit_predict(extended_scaled)


extended_best_summary = customer_features_ext.groupby("Cluster_Extended_Best").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean",
    "AvgPurchaseValue": "mean",
    "ProductDiversity": "mean",
    "AvgDaysBetweenTransactions": "mean",
    "CustomerID": "count"
}).round(2)

extended_best_summary.rename(columns={"CustomerID": "CustomerCount"}, inplace=True)

extended_best_summary["Percentage"] = (
    extended_best_summary["CustomerCount"] / len(customer_features_ext) * 100
).round(2)

extended_best_summary.to_csv(
    output_folder / "extended_best_cluster_summary.csv"
)



# k = 3 and k = 4 extended clustering summaries


for k in [3, 4]:
    kmeans_ext = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_col = f"Cluster_Extended_{k}"
    customer_features_ext[cluster_col] = kmeans_ext.fit_predict(extended_scaled)

    summary = customer_features_ext.groupby(cluster_col).agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean",
        "AvgPurchaseValue": "mean",
        "ProductDiversity": "mean",
        "AvgDaysBetweenTransactions": "mean",
        "CustomerID": "count"
    }).round(2)

    summary.rename(columns={"CustomerID": "CustomerCount"}, inplace=True)

    summary["Percentage"] = (
        summary["CustomerCount"] / len(customer_features_ext) * 100
    ).round(2)

    summary.to_csv(
        output_folder / f"extended_cluster_{k}_summary.csv"
    )

    temp_df = extended_tsne_log_df.copy()
    temp_df["Cluster"] = customer_features_ext[cluster_col]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        temp_df["TSNE1"],
        temp_df["TSNE2"],
        c=temp_df["Cluster"],
        cmap="tab10",
        alpha=0.7
    )

    plt.title(f"t-SNE Visualization (Extended Features, k={k})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.savefig(output_folder / f"extended_tsne_k{k}.png", dpi=300)
    plt.show()
    plt.close()


# t-SNE for best extended cluster solution

temp_best = extended_tsne_log_df.copy()
temp_best["Cluster"] = customer_features_ext["Cluster_Extended_Best"]

plt.figure(figsize=(8, 6))
plt.scatter(
    temp_best["TSNE1"],
    temp_best["TSNE2"],
    c=temp_best["Cluster"],
    cmap="tab10",
    alpha=0.7
)

plt.title(f"t-SNE Visualization (Extended Features, Best k={best_k_ext})")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.savefig(output_folder / "extended_tsne_best_k.png", dpi=300)
plt.show()
plt.close()



# Prediction comparison

# AvgPurchaseValue is NOT included because it is derived from Monetary/TotalPrice.
# Including it would create leakage.

# Baseline model: Recency + Frequency
X_base = customer_features_ext[["Recency", "Frequency"]]

# Extended no-leakage model
X_ext = customer_features_ext[
    [
        "Recency",
        "Frequency",
        "ProductDiversity",
        "AvgDaysBetweenTransactions"
    ]
]

y = customer_features_ext["Monetary"]

X_base_train, X_base_test, y_train, y_test = train_test_split(
    X_base,
    y,
    test_size=0.2,
    random_state=42
)

X_ext_train, X_ext_test, _, _ = train_test_split(
    X_ext,
    y,
    test_size=0.2,
    random_state=42
)

scaler_base = StandardScaler()
X_base_train_scaled = scaler_base.fit_transform(X_base_train)
X_base_test_scaled = scaler_base.transform(X_base_test)

scaler_ext_pred = StandardScaler()
X_ext_train_scaled = scaler_ext_pred.fit_transform(X_ext_train)
X_ext_test_scaled = scaler_ext_pred.transform(X_ext_test)


base_model = LinearRegression()
base_model.fit(X_base_train_scaled, y_train)
base_pred = base_model.predict(X_base_test_scaled)

extended_model = LinearRegression()
extended_model.fit(X_ext_train_scaled, y_train)
extended_pred = extended_model.predict(X_ext_test_scaled)

prediction_comparison = pd.DataFrame({
    "Model": [
        "Baseline Regression",
        "Extended Feature Regression"
    ],
    "Features": [
        "Recency, Frequency",
        "Recency, Frequency, ProductDiversity, AvgDaysBetweenTransactions"
    ],
    "Target": [
        "Monetary",
        "Monetary"
    ],
    "MAE": [
        round(mean_absolute_error(y_test, base_pred), 2),
        round(mean_absolute_error(y_test, extended_pred), 2)
    ],
    "RMSE": [
        round(np.sqrt(mean_squared_error(y_test, base_pred)), 2),
        round(np.sqrt(mean_squared_error(y_test, extended_pred)), 2)
    ],
    "MAPE": [
        round(mean_absolute_percentage_error(y_test, base_pred) * 100, 2),
        round(mean_absolute_percentage_error(y_test, extended_pred) * 100, 2)
    ],
    "R2": [
        round(r2_score(y_test, base_pred), 4),
        round(r2_score(y_test, extended_pred), 4)
    ]
})

prediction_comparison.to_csv(
    output_folder / "extended_prediction_comparison.csv",
    index=False
)

print("\nPrediction Comparison:")
print(prediction_comparison)