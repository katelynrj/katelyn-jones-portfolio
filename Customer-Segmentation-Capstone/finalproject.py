#working with excel files 
import pandas as pd
#for numerical operations, including log transformations 
import numpy as np
#manage folder/file paths 
from pathlib import Path
#creating plots and saving figures
import matplotlib.pyplot as plt
#standarize features 
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import seaborn as sns

# File paths

#store the location for online retail dataset
data_path = "/Users/katelynjones/Documents/Online Retail.xlsx"

#create a path object for the folder where output files will be saved
output_folder = Path("/Users/katelynjones/Documents/masters_program_output")
#create output folder
output_folder.mkdir(parents=True, exist_ok=True)


# Load data

df = pd.read_excel(data_path)

# Dataset summary table

summary_table = pd.DataFrame({
    "Attribute": df.columns,
    "DataType": df.dtypes.astype(str).values,
    "ExampleValue": df.iloc[0].values,
    "Description": [
        "Unique transaction ID",
        "Product code",
        "Product description",
        "Quantity of product purchased",
        "Date and time of transaction",
        "Price per unit",
        "Unique customer identifier",
        "Country of customer"
    ]
})

# Save the dataset summary table to file 
# index=False prevents pandas from writing the row index as an extra column

summary_table.to_csv(output_folder / "table_dataset_summary.csv", index=False)


# Missing data analysis


missing_rows = df[df.isnull().any(axis=1)]

missing_customer = df[df["CustomerID"].isnull()]


missing_rows.to_csv(output_folder / "rows_with_missing_values.csv", index=False)


# Basic cleaning

# Remove rows missing key customer-level info
df_clean = df.dropna(subset=["CustomerID", "InvoiceNo", "StockCode", "InvoiceDate"]).copy()

# Remove cancellation invoices
df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")]

# Remove rows with non-positive quantity or price
df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]

# Standardize CustomerID format
df_clean["CustomerID"] = df_clean["CustomerID"].astype(int).astype(str)

# Create transaction total
df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]


# RFM creation

# This is used as the reference point for calculating Recency
snapshot_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)

# Group transactions by CustomerID to create one row per customer
# Recency: days between the snapshot date and the customer's most recent purchase
# Frequency: number of unique invoices/transactions per customer
# Monetary: total amount spent by each customer


rfm = df_clean.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]



# t-SNE BEFORE log transformation 

# Standardize raw Recency, Frequency, and Monetary values
# This prevents large values like Monetary from dominating the visualization

scaler_raw = StandardScaler()
rfm_scaled_raw = scaler_raw.fit_transform(
    rfm[["Recency", "Frequency", "Monetary"]]
)

# Reduce the data to 2 components so it can be plotted on x/y axes
# Set random_state for repeating code 
# Perplexity controls how many nearby points t-SNE considers   
# Learning rate controls the step size during t-SNE optimization
# Maximum number of optimization iterations
tsne_raw = TSNE(
    n_components=2,
    random_state=42,
    perplexity=50,
    learning_rate=200,
    max_iter=2000
)


# Apply t-SNE to the scaled raw RFM data
rfm_tsne_raw = tsne_raw.fit_transform(rfm_scaled_raw)

tsne_raw_df = pd.DataFrame(rfm_tsne_raw, columns=["TSNE1", "TSNE2"])


# Create a scatter plot of the two t-SNE components
# Color each point by the customer's Monetary value

plt.figure(figsize=(8,6))
plt.scatter(
    tsne_raw_df["TSNE1"],
    tsne_raw_df["TSNE2"],
    c=rfm["Monetary"],
    alpha=0.7
)

plt.title("t-SNE Visualization (Raw RFM - Before Log)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(label="Monetary")
plt.tight_layout()
plt.savefig(output_folder / "tsne_rfm_raw.png", dpi=300)
plt.show()
plt.close()


# Log transformation (to reduce skew)
# Apply log(1 + x) transformation to RFM variables

rfm["Recency_log"] = np.log1p(rfm["Recency"])
rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
rfm["Monetary_log"] = np.log1p(rfm["Monetary"])

# Scale RFM features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(
    rfm[["Recency_log", "Frequency_log", "Monetary_log"]]
)
# Apply t-SNE
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=50,
    learning_rate=200,
    max_iter=2000
)

rfm_tsne = tsne.fit_transform(rfm_scaled)


# RFM-only K-means evaluation

# Create an empty list to store evaluation results for each k value

rfm_kmeans_results = []

# Test K-means clustering using k values from 2 through 8
# range(2, 9) includes 2 but stops before 9

# Create a K-means model with the current number of clusters
# random_state=42 makes results reproducible
# n_init=10 means K-means will run 10 different centroid initializations
# and keep the best result

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rfm_scaled)


    sil_score = silhouette_score(rfm_scaled, labels)
    db_score = davies_bouldin_score(rfm_scaled, labels)

    rfm_kmeans_results.append({
        "Method": "KMeans_RFM",
        "Clusters": k,
        "SilhouetteScore": round(sil_score, 4),
        "DaviesBouldinIndex": round(db_score, 4)
    })

rfm_kmeans_results_df = pd.DataFrame(rfm_kmeans_results)

print("\nRFM-only K-means results:")
print(rfm_kmeans_results_df)

rfm_kmeans_results_df.to_csv(output_folder / "rfm_kmeans_evaluation_results.csv", index=False)


# Outlier Filtering Test for Clustering
# Calculate the first quartile of Monetary values
# Calculate the third quartile of Monetary values

Q1 = rfm["Monetary"].quantile(0.25)
Q3 = rfm["Monetary"].quantile(0.75)


# IQR measures the spread of the middle 50% of the Monetary data
IQR = Q3 - Q1


# Create an upper outlier limit using the common IQR rule
# Values above Q3 + 1.5*IQR are high-end outliers
upper_limit = Q3 + 1.5 * IQR

# Create a filtered RFM dataset that removes extreme Monetary outliers
# .copy() avoids modifying the original rfm DataFrame

rfm_cluster_filtered = rfm[
    rfm["Monetary"] <= upper_limit
].copy()

# Log transform
rfm_cluster_filtered["Recency_log"] = np.log1p(rfm_cluster_filtered["Recency"])
rfm_cluster_filtered["Frequency_log"] = np.log1p(rfm_cluster_filtered["Frequency"])
rfm_cluster_filtered["Monetary_log"] = np.log1p(rfm_cluster_filtered["Monetary"])

# Scale features
filtered_scaled = scaler.fit_transform(
    rfm_cluster_filtered[
        ["Recency_log", "Frequency_log", "Monetary_log"]
    ]
)

# KMeans with k=2
kmeans_filtered = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)
# Fit K-means to the filtered data and get cluster labels
filtered_labels = kmeans_filtered.fit_predict(filtered_scaled)

# Evaluation metrics
filtered_silhouette = silhouette_score(
    filtered_scaled,
    filtered_labels
)

filtered_db = davies_bouldin_score(
    filtered_scaled,
    filtered_labels
)

print("\nFiltered Clustering Results:")
print("Silhouette Score:", round(filtered_silhouette, 4))
print("Davies-Bouldin Index:", round(filtered_db, 4))


# Select best k for final RFM clustering
# Find the row with the highest silhouette score
# Then retrieve the corresponding number of clusters

best_k_rfm = rfm_kmeans_results_df.loc[
    rfm_kmeans_results_df["SilhouetteScore"].idxmax(), "Clusters"
]

# Convert best_k_rfm to an integer
best_k_rfm = int(best_k_rfm)

# Create the final K-means model using the best k value
kmeans_rfm = KMeans(n_clusters=best_k_rfm, random_state=42, n_init=10)

# Fit the final K-means model and store cluster labels in the rfm DataFrame

rfm["Cluster_RFM"] = kmeans_rfm.fit_predict(rfm_scaled)


# Additional clustering (k = 3 and k = 4)

kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm["Cluster_3"] = kmeans_3.fit_predict(rfm_scaled)

kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster_4"] = kmeans_4.fit_predict(rfm_scaled)


# Save k=3 and k=4 summaries
# Loop through both additional cluster label columns


for cluster_col in ["Cluster_3", "Cluster_4"]:
    summary = rfm.groupby(cluster_col).agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean",
        "CustomerID": "count"
    }).round(2)

    # Rename CustomerID count column to CustomerCount for clarity

    summary.rename(columns={"CustomerID": "CustomerCount"}, inplace=True)
    # Calculate the percentage of total customers in each cluster
    summary["Percentage"] = (summary["CustomerCount"] / len(rfm) * 100).round(2)

    

    
    summary.to_csv(output_folder / f"{cluster_col.lower()}_summary.csv")

#Create t-SNE plots for k=3 and k=4
plt.figure(figsize=(8,6))

for label_col, name in [("Cluster_3", "k=3"), ("Cluster_4", "k=4")]:
    temp_df = pd.DataFrame(rfm_tsne, columns=["TSNE1", "TSNE2"])
    temp_df["Cluster"] = rfm[label_col]
    plt.figure(figsize=(8,6))

    # Convert cluster labels to strings for coloring
    temp_df["Cluster"] = temp_df["Cluster"].astype(str)

    # Plot each cluster separately
    for cluster in temp_df["Cluster"].unique():
        cluster_data = temp_df[temp_df["Cluster"] == cluster]

        plt.scatter(
            cluster_data["TSNE1"],
            cluster_data["TSNE2"],
            label=cluster,
            alpha=0.7
        )

    plt.legend(title="Cluster")

    plt.title(f"t-SNE Visualization ({name})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(label="Cluster")
    plt.tight_layout()

    
    plt.savefig(output_folder / f"tsne_{label_col.lower()}.png", dpi=300)

    plt.show()
    plt.close()

# RFM-only cluster summary

# Group customers by their final RFM K-means cluster
# Then calculate the average Recency, Frequency, Monetary value,
# and count the number of customers in each cluster

rfm_cluster_summary = rfm.groupby("Cluster_RFM").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean",
    "CustomerID": "count"
}).round(2)

rfm_cluster_summary.rename(columns={"CustomerID": "CustomerCount"}, inplace=True)

# Count the total number of customers in the RFM dataset

total_customers = len(rfm)


# Calculate what percentage of all customers belongs to each cluster

rfm_cluster_summary["Percentage"] = (
    rfm_cluster_summary["CustomerCount"] / total_customers * 100
).round(2)
# Sort the cluster summary by cluster number

rfm_cluster_summary = rfm_cluster_summary.sort_index()


rfm_cluster_summary.to_csv(output_folder / "rfm_kmeans_cluster_summary.csv")

# t-SNE with RFM clusters
# Sort the cluster summary by cluster number


tsne_rfm_df = pd.DataFrame(rfm_tsne, columns=["TSNE1", "TSNE2"])
# Add the final RFM K-means cluster label for each customer

tsne_rfm_df["Cluster"] = rfm["Cluster_RFM"]
# Create a new figure for the cluster visualization


# Plot the t-SNE points
# Each point represents one customer
# Color represents the customer's assigned cluster

plt.figure(figsize=(8,6))
# Convert cluster labels to strings
tsne_rfm_df["Cluster"] = tsne_rfm_df["Cluster"].astype(str)

# Plot each cluster separately
for cluster in tsne_rfm_df["Cluster"].unique():
    cluster_data = tsne_rfm_df[
        tsne_rfm_df["Cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["TSNE1"],
        cluster_data["TSNE2"],
        label=cluster,
        alpha=0.7
    )

plt.legend(title="Cluster")

plt.title("t-SNE Visualization of RFM-Based Clusters")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.savefig(output_folder / "tsne_rfm_clusters.png", dpi=300)
plt.show()


# TSNE colored by monetary value
# Convert to DataFrame
tsne_df = pd.DataFrame(rfm_tsne, columns=["TSNE1", "TSNE2"])

plt.figure(figsize=(8,6))
plt.scatter(tsne_df["TSNE1"], tsne_df["TSNE2"], c=rfm["Monetary"], alpha=0.7)
plt.title("t-SNE Visualization of RFM Features")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(label="Monetary")
plt.tight_layout()
plt.savefig(output_folder / "tsne_rfm.png", dpi=300)
plt.show()



# OPTICS clustering

# min_samples=5 means at least 5 nearby points are needed to form a dense region
# xi=0.03 controls how steep a density change must be to separate clusters
# min_cluster_size=0.03 means each cluster must contain at least 3% of the data

optics = OPTICS(
    min_samples=5,
    xi=0.03,
    min_cluster_size=0.03
)

# Fit OPTICS to the scaled RFM data and return cluster labels
optics_labels = optics.fit_predict(rfm_scaled)


# t-SNE visualization of OPTICS clusters
tsne_optics_df = pd.DataFrame(rfm_tsne, columns=["TSNE1", "TSNE2"])
tsne_optics_df["Cluster"] = optics_labels



plt.figure(figsize=(8,6))
# Plot the t-SNE coordinates colored by OPTICS cluster labels

# Convert cluster labels to strings
tsne_optics_df["Cluster"] = tsne_optics_df["Cluster"].astype(str)

# Plot each cluster separately
for cluster in tsne_optics_df["Cluster"].unique():
    cluster_data = tsne_optics_df[
        tsne_optics_df["Cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["TSNE1"],
        cluster_data["TSNE2"],
        label=cluster,
        alpha=0.7
    )

plt.legend(title="Cluster")


plt.title("t-SNE Visualization (OPTICS Clusters)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.savefig(output_folder / "tsne_optics_clusters.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()



rfm["Cluster_OPTICS"] = optics_labels

# Group customers by OPTICS cluster and calculate average RFM values
# Also count how many customers are in each OPTICS group

optics_summary = rfm.groupby("Cluster_OPTICS").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean",
    "CustomerID": "count"
}).round(2)



optics_summary.rename(columns={"CustomerID": "CustomerCount"}, inplace=True)



total_customers = len(rfm)
# Calculate percentage of customers in each OPTICS cluster

optics_summary["Percentage"] = (
    optics_summary["CustomerCount"] / total_customers * 100
).round(2)


optics_summary.to_csv(output_folder / "optics_summary.csv")

# Silhouette Score for OPTICS

# Create a mask that excludes points labeled as noise

mask = optics_labels != -1
# Keep only non-noise OPTICS labels
labels_clean = optics_labels[mask]
# Keep only the scaled data points that were not labeled as noise

data_clean = rfm_scaled[mask]
# Only calculate silhouette score if OPTICS found more than one real cluster

if len(set(labels_clean)) > 1:
    sil_score = silhouette_score(data_clean, labels_clean)
    print("\nSilhouette Score (OPTICS):", round(sil_score, 3))
else:
    print("\nSilhouette Score: Not enough clusters")
    

# Create clean tables for report


# Missing values table
missing_table = df.isnull().sum().reset_index()
missing_table.columns = ["Column", "MissingCount"]

missing_table.to_csv(output_folder / "table_missing_values.csv", index=False)

# RFM summary statistics 
rfm_stats = rfm[["Recency", "Frequency", "Monetary"]].describe().transpose()

rfm_stats = rfm_stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]

rfm_stats = rfm_stats.round(2)


rfm_stats.to_csv(output_folder / "table_rfm_summary.csv")

# Distribution of Monetary Value
plt.figure(figsize=(8, 5))
plt.hist(rfm["Monetary"], bins=100)

plt.title("Distribution of Monetary Value")
plt.xlabel("Monetary")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig(output_folder / "hist_monetary.png", dpi=300)

plt.show()
plt.close()

# Boxplot of Monetary Value (w/o extreme outliers)


plt.figure(figsize=(8, 5))
plt.boxplot(rfm["Monetary"], showfliers=False)  # hides extreme outliers
plt.title("Boxplot of Monetary Value (Without Extreme Outliers)")
plt.ylabel("Monetary")
plt.tight_layout()
plt.savefig(output_folder / "boxplot_monetary_clean.png", dpi=300)
plt.show()


# Boxplot of Monetary Value (w/ outliers)
plt.figure(figsize=(8,5))
plt.boxplot(rfm["Monetary"])
plt.title("Boxplot of Monetary Value (Including Outliers)")
plt.ylabel("Monetary")


plt.savefig(output_folder / "boxplot_monetary_with_outliers.png", dpi=300)

plt.show()

# Recency histogram
plt.figure(figsize=(8,5))
plt.hist(rfm["Recency"], bins=50)
plt.title("Distribution of Recency")
plt.xlabel("Recency (Days)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(output_folder / "hist_recency.png", dpi=300)
plt.show()

# Frequency Histogram
plt.figure(figsize=(8,5))
plt.hist(rfm["Frequency"], bins=75)


plt.title("Distribution of Frequency")
plt.xlabel("Frequency")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig(output_folder / "hist_frequency.png", dpi=300)
plt.show()


# Feature engineering - additional customer level variables

# Group transactions by customer
# Calculate:
# - total amount spent
# - number of unique invoices/transactions

avg_purchase_value = df_clean.groupby("CustomerID").agg({
    "TotalPrice": "sum",
    "InvoiceNo": "nunique"
}).reset_index()



avg_purchase_value["AvgPurchaseValue"] = (
    avg_purchase_value["TotalPrice"] / avg_purchase_value["InvoiceNo"]
)


# Keep only CustomerID and the new AvgPurchaseValue feature

avg_purchase_value = avg_purchase_value[["CustomerID", "AvgPurchaseValue"]]

# Product diversity
# Count how many unique products each customer purchased
# nunique counts distinct StockCode values


product_diversity = df_clean.groupby("CustomerID").agg({
    "StockCode": "nunique"
}).reset_index()
# Rename the column for readability
product_diversity.columns = ["CustomerID", "ProductDiversity"]

# Average time between transactions
# Group by both CustomerID and InvoiceNo
# Then get the earliest timestamp for each invoice
# This creates one timestamp per transaction
invoice_dates = (
    df_clean.groupby(["CustomerID", "InvoiceNo"])["InvoiceDate"]
    .min()
    .reset_index()
    .sort_values(["CustomerID", "InvoiceDate"])
)

# Calculate the number of days between consecutive transactions
# diff() computes the difference between current and previous invoice dates
invoice_dates["DaysBetweenTransactions"] = (
    invoice_dates.groupby("CustomerID")["InvoiceDate"].diff().dt.days
)

# Calculate the average number of days between purchases per customer

avg_days_between = (
    invoice_dates.groupby("CustomerID")["DaysBetweenTransactions"]
    .mean()
    .reset_index()
)

# Rename the column


avg_days_between.columns = ["CustomerID", "AvgDaysBetweenTransactions"]

# Customers with only one transaction have no previous purchase
# so diff() creates NaN values
# Replace those missing values with 0

avg_days_between["AvgDaysBetweenTransactions"] = (
    avg_days_between["AvgDaysBetweenTransactions"].fillna(0)
)


# Merge new features

customer_features = rfm.merge(avg_purchase_value, on="CustomerID", how="left")
customer_features = customer_features.merge(product_diversity, on="CustomerID", how="left")
customer_features = customer_features.merge(avg_days_between, on="CustomerID", how="left")


customer_features.to_csv(output_folder / "customer_features_extended.csv", index=False)



# Correlation Analysis (RFM)


# Select only the original RFM variables
rfm_corr = customer_features[["Recency", "Frequency", "Monetary"]]

# Calculate pairwise Pearson correlation values
# round(2) limits values to 2 decimal places
corr_matrix = rfm_corr.corr().round(2)

# Save the correlation matrix as a CSV file

corr_matrix.to_csv(output_folder / "rfm_correlation_matrix.csv")

# Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True)

plt.title("Correlation Matrix (RFM Features)")

plt.tight_layout()
plt.savefig(output_folder / "rfm_correlation_heatmap.png", dpi=300)
plt.show()
plt.close()

# Scatter Plots


# Recency vs Monetary
plt.figure()
plt.scatter(customer_features["Recency"], customer_features["Monetary"], alpha=0.5)
plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.title("Recency vs Monetary")
plt.tight_layout()
plt.savefig(output_folder / "recency_vs_monetary.png", dpi=300)
plt.show()
plt.close()

# Frequency vs Monetary
plt.figure()
plt.scatter(customer_features["Frequency"], customer_features["Monetary"], alpha=0.5)
plt.xlabel("Frequency")
plt.ylabel("Monetary")
plt.title("Frequency vs Monetary")
plt.tight_layout()
plt.savefig(output_folder / "frequency_vs_monetary.png", dpi=300)
plt.show()
plt.close()

# Zoomed Frequency vs Monetary

zoom_df = customer_features[
    (customer_features["Frequency"] <= 50) &
    (customer_features["Monetary"] <= 25000)
]

plt.figure(figsize=(8,6))

plt.scatter(
    zoom_df["Frequency"],
    zoom_df["Monetary"],
    alpha=0.5
)

plt.xlabel("Frequency")
plt.ylabel("Monetary")
plt.title("Frequency vs Monetary (Zoomed View)")

plt.xlim(0, 50)
plt.ylim(0, 25000)

plt.tight_layout()

plt.savefig(
    output_folder / "frequency_vs_monetary_zoomed.png",
    dpi=300
)

plt.show()
plt.close()

# Recency vs Frequency
plt.figure()
plt.scatter(customer_features["Recency"], customer_features["Frequency"], alpha=0.5)
plt.xlabel("Recency")
plt.ylabel("Frequency")
plt.title("Recency vs Frequency")
plt.tight_layout()
plt.savefig(output_folder / "recency_vs_frequency.png", dpi=300)
plt.show()
plt.close()


# Prediction task - Predict Monetary using Recency and Frequency


# Features - only Recency and Frequency
X = customer_features[["Recency", "Frequency"]]

# Target - Monetary value
y = customer_features["Monetary"]

# Train/test split
# 80% training data
# 20% testing data
# random_state=42 ensures reproducible splits
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale Recency and Frequency
scaler_pred = StandardScaler()
# fit_transform learns scaling parameters and applies them

X_train_scaled = scaler_pred.fit_transform(X_train)

# Apply the SAME scaling parameters to testing data
# transform only applies existing scaling
X_test_scaled = scaler_pred.transform(X_test)

# Train regression model
model_monetary = LinearRegression()
model_monetary.fit(X_train_scaled, y_train)

# Predict Monetary
y_pred = model_monetary.predict(X_test_scaled)

# Evaluate regression model
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

# MAPE measures average percentage prediction error
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

prediction_results = pd.DataFrame({
    "Model": ["Linear Regression"],
    "Features": ["Recency, Frequency"],
    "Target": ["Monetary"],
    "MAE": [round(mae, 2)],
    "RMSE": [round(rmse, 2)],
    "MAPE": [round(mape, 2)],
    "R2": [round(r2, 4)]
})

prediction_results.to_csv(
    output_folder / "prediction_monetary_rf_results.csv",
    index=False
)

# Save actual vs predicted values
actual_vs_predicted = pd.DataFrame({
    "ActualMonetary": y_test.values,
    "PredictedMonetary": y_pred
}).round(2)

actual_vs_predicted.to_csv(
    output_folder / "prediction_monetary_actual_vs_predicted.csv",
    index=False
)

# Plot actual vs predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)

plt.title("Actual vs Predicted Monetary Value")
plt.xlabel("Actual Monetary")
plt.ylabel("Predicted Monetary")

plt.tight_layout()
plt.savefig(output_folder / "actual_vs_predicted_monetary.png", dpi=300)
plt.show()
plt.close()


# Refined Regression Model

# Create a filtered dataset that removes extreme outliers
# This keeps customers with:
# - Frequency <= 50
# - Monetary <= 25000

filtered_df = customer_features[
    (customer_features["Frequency"] <= 50) &
    (customer_features["Monetary"] <= 25000)
]


# Predictor variables
# Recency and Frequency are used to predict spending behavior

# Features
X_filtered = filtered_df[["Recency", "Frequency"]]

# Target
y_filtered = filtered_df["Monetary"]

# Split filtered data into:
# - 80% training data
# - 20% testing data
# random_state=42 keeps results reproducible
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_filtered,
    y_filtered,
    test_size=0.2,
    random_state=42
)

# Scale predictor variables

scaler_filtered = StandardScaler()

Xf_train_scaled = scaler_filtered.fit_transform(Xf_train)
Xf_test_scaled = scaler_filtered.transform(Xf_test)

# Train model
filtered_model = LinearRegression()

filtered_model.fit(Xf_train_scaled, yf_train)

# Predict
yf_pred = filtered_model.predict(Xf_test_scaled)


# Metrics
filtered_mae = mean_absolute_error(yf_test, yf_pred)

filtered_rmse = np.sqrt(mean_squared_error(yf_test, yf_pred))

filtered_r2 = r2_score(yf_test, yf_pred)

filtered_mape = mean_absolute_percentage_error(yf_test, yf_pred) * 100

# Save results
filtered_results = pd.DataFrame({
    "Model": ["Filtered Linear Regression"],
    "Features": ["Recency, Frequency"],
    "Target": ["Monetary"],
    "FrequencyLimit": [50],
    "MonetaryLimit": [25000],
    "MAE": [round(filtered_mae, 2)],
    "RMSE": [round(filtered_rmse, 2)],
    "MAPE": [round(filtered_mape, 2)],
    "R2": [round(filtered_r2, 4)]
})

filtered_results.to_csv(
    output_folder / "prediction_filtered_results.csv",
    index=False
)

# Plot actual vs predicted
plt.figure(figsize=(8,6))

plt.scatter(yf_test, yf_pred, alpha=0.6)

plt.title("Actual vs Predicted Monetary Value (Filtered Data)")
plt.xlabel("Actual Monetary")
plt.ylabel("Predicted Monetary")

plt.tight_layout()

plt.savefig(
    output_folder / "actual_vs_predicted_filtered.png",
    dpi=300
)

plt.show()
plt.close()

# Summary table for new features

new_features_stats = customer_features[
    ["AvgPurchaseValue", "ProductDiversity", "AvgDaysBetweenTransactions"]
].describe().transpose().round(2)


new_features_stats.to_csv(output_folder / "table_new_features_summary.csv")

plt.figure(figsize=(8,5))
plt.hist(customer_features["ProductDiversity"], bins=75)
plt.title("Distribution of Product Diversity")
plt.xlabel("Number of Unique Products")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(output_folder / "hist_product_diversity.png", dpi=300)
plt.show()

plt.figure(figsize=(8,5))
plt.hist(customer_features["AvgPurchaseValue"], bins=75)

plt.title("Distribution of Average Purchase Value")
plt.xlabel("Average Purchase Value")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig(output_folder / "hist_avg_purchase_value.png", dpi=300)
plt.show()


# Time Series - Sales over time


# Create a new column containing only the date
# This removes the time portion from InvoiceDate
df_clean["InvoiceDateOnly"] = df_clean["InvoiceDate"].dt.date

# Group transactions by date
# Then calculate total daily revenue

daily_sales = df_clean.groupby("InvoiceDateOnly")["TotalPrice"].sum().reset_index()

# Rename columns

daily_sales.columns = ["Date", "Sales"]

# Convert Date column back into pandas datetime format

daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])

# Sort
daily_sales = daily_sales.sort_values("Date")

# Plot sales over time 

plt.figure(figsize=(10,5))
plt.plot(daily_sales["Date"], daily_sales["Sales"])

plt.title("Daily Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.savefig(output_folder / "timeseries_daily_sales.png", dpi=300)
plt.show()

# Plot Rolling average (7-day)
daily_sales["RollingMean"] = daily_sales["Sales"].rolling(window=7).mean()

plt.figure(figsize=(10,5))
plt.plot(daily_sales["Date"], daily_sales["Sales"], label="Daily Sales")
plt.plot(daily_sales["Date"], daily_sales["RollingMean"], label="7-Day Avg")

plt.legend()
plt.title("Daily Sales with Rolling Average")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.tight_layout()
plt.savefig(output_folder / "timeseries_trend.png", dpi=300)
plt.show()

# Plot monthly sales trend 

monthly_sales = df_clean.set_index("InvoiceDate").resample("ME")["TotalPrice"].sum()

# Remove final incomplete month

monthly_sales = monthly_sales.iloc[:-1]

plt.figure(figsize=(10,5))
plt.plot(monthly_sales.index, monthly_sales.values)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.ticklabel_format(style='plain', axis='y')

plt.tight_layout()

monthly_path = output_folder / "timeseries_monthly.png"
plt.savefig(monthly_path, dpi=300)


plt.show()
plt.close()


# ARIMA Forecasting


# Set Date as index and force daily frequency
# asfreq("D") ensures every calendar day exists in the time series
daily_sales_ts = daily_sales.set_index("Date").asfreq("D")

# Fill missing days with 0 sales
daily_sales_ts["Sales"] = daily_sales_ts["Sales"].fillna(0)

# Split data
train_data = daily_sales_ts.iloc[:-7]
test_data = daily_sales_ts.iloc[-7:]

# Create ARIMA forecasting model
# order=(4,1,1) means:
# - 4 autoregressive terms
# - 1 differencing step
# - 1 moving average term
model = ARIMA(train_data["Sales"], order=(4, 1, 1))

model_fit = model.fit()

# Forecast final 7 known days
forecast = model_fit.forecast(steps=7)

# Create comparison table
forecast_comparison = pd.DataFrame({
    "Date": test_data.index,
    "ActualSales": test_data["Sales"].values,
    "PredictedSales": forecast.values
})

# Round values
forecast_comparison["ActualSales"] = (
    forecast_comparison["ActualSales"].round(2)
)

forecast_comparison["PredictedSales"] = (
    forecast_comparison["PredictedSales"].round(2)
)

# Calculate forecast metrics
forecast_mae = mean_absolute_error(
    forecast_comparison["ActualSales"],
    forecast_comparison["PredictedSales"]
)

forecast_rmse = np.sqrt(mean_squared_error(
    forecast_comparison["ActualSales"],
    forecast_comparison["PredictedSales"]
))

forecast_nonzero = forecast_comparison["ActualSales"] != 0

forecast_mape = mean_absolute_percentage_error(
    forecast_comparison.loc[forecast_nonzero, "ActualSales"],
    forecast_comparison.loc[forecast_nonzero, "PredictedSales"]
) * 100

# Save table
forecast_comparison.to_csv(
    output_folder / "arima_forecast_validation.csv",
    index=False
)

forecast_metrics = pd.DataFrame({
    "Model": ["ARIMA(4,1,1)"],
    "ForecastHorizon": ["7 Days"],
    "MAE": [round(forecast_mae, 2)],
    "RMSE": [round(forecast_rmse, 2)],
    "MAPE": [round(forecast_mape, 2)]
})

forecast_metrics.to_csv(
    output_folder / "arima_forecast_metrics.csv",
    index=False
)

# Plot forecast vs actual
plt.figure(figsize=(10,5))

plt.plot(
    train_data.index,
    train_data["Sales"],
    label="Training Data"
)

plt.plot(
    test_data.index,
    test_data["Sales"],
    label="Actual Sales"
)

plt.plot(
    test_data.index,
    forecast,
    label="Predicted Sales"
)

plt.legend()

plt.title("ARIMA(4,1,1) Forecast Validation")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.tight_layout()

plt.savefig(
    output_folder / "arima_forecast_validation.png",
    dpi=300
)

plt.show()
plt.close()

# Zoomed forecast comparison

plt.figure(figsize=(10,5))

plt.plot(
    test_data.index,
    test_data["Sales"],
    marker="o",
    label="Actual Sales"
)

plt.plot(
    test_data.index,
    forecast,
    marker="o",
    label="Predicted Sales"
)

plt.legend()

plt.title("ARIMA(4,1,1) Forecast vs Actual Sales (Final 7 Days)")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.tight_layout()

plt.savefig(
    output_folder / "arima_forecast_zoomed.png",
    dpi=300
)

plt.show()
plt.close()


# ARIMA Forecast Validation
# Mid-August 2011


# Define validation period
august_start = pd.to_datetime("2011-08-15")
august_end = pd.to_datetime("2011-08-21")

# Training data ends before the validation period
train_august = daily_sales_ts[daily_sales_ts.index < august_start]

# Actual sales during the validation period
test_august = daily_sales_ts[
    (daily_sales_ts.index >= august_start) &
    (daily_sales_ts.index <= august_end)
]

# Fit ARIMA(4,1,1)
model_august = ARIMA(train_august["Sales"], order=(4, 1, 1))
model_august_fit = model_august.fit()

# Forecast the 7-day validation period
forecast_august = model_august_fit.forecast(steps=len(test_august))

# Create comparison table
forecast_august_comparison = pd.DataFrame({
    "Date": test_august.index,
    "ActualSales": test_august["Sales"].values,
    "PredictedSales": forecast_august.values
})

forecast_august_comparison["ActualSales"] = (
    forecast_august_comparison["ActualSales"].round(2)
)

forecast_august_comparison["PredictedSales"] = (
    forecast_august_comparison["PredictedSales"].round(2)
)

# Calculate metrics
forecast_august_mae = mean_absolute_error(
    forecast_august_comparison["ActualSales"],
    forecast_august_comparison["PredictedSales"]
)

forecast_august_rmse = np.sqrt(mean_squared_error(
    forecast_august_comparison["ActualSales"],
    forecast_august_comparison["PredictedSales"]
))

forecast_august_nonzero = (
    forecast_august_comparison["ActualSales"] != 0
)

forecast_august_mape = mean_absolute_percentage_error(
    forecast_august_comparison.loc[
        forecast_august_nonzero,
        "ActualSales"
    ],
    forecast_august_comparison.loc[
        forecast_august_nonzero,
        "PredictedSales"
    ]
) * 100

# Save comparison table
forecast_august_comparison.to_csv(
    output_folder / "arima_forecast_validation_august.csv",
    index=False
)

# Save metrics table
forecast_august_metrics = pd.DataFrame({
    "Model": ["ARIMA(4,1,1)"],
    "ValidationPeriod": ["2011-08-15 to 2011-08-21"],
    "ForecastHorizon": ["7 Days"],
    "MAE": [round(forecast_august_mae, 2)],
    "RMSE": [round(forecast_august_rmse, 2)],
    "MAPE": [round(forecast_august_mape, 2)]
})

forecast_august_metrics.to_csv(
    output_folder / "arima_forecast_metrics_august.csv",
    index=False
)

# Plot full forecast validation
plt.figure(figsize=(10,5))

plt.plot(
    train_august.index,
    train_august["Sales"],
    label="Training Data"
)

plt.plot(
    test_august.index,
    test_august["Sales"],
    label="Actual Sales"
)

plt.plot(
    test_august.index,
    forecast_august,
    label="Predicted Sales"
)

plt.legend()
plt.title("ARIMA(4,1,1) Forecast Validation - Mid-August 2011")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.tight_layout()

plt.savefig(
    output_folder / "arima_forecast_validation_august.png",
    dpi=300
)

plt.show()
plt.close()

# Zoomed forecast comparison
plt.figure(figsize=(10,5))

plt.plot(
    test_august.index,
    test_august["Sales"],
    marker="o",
    label="Actual Sales"
)

plt.plot(
    test_august.index,
    forecast_august,
    marker="o",
    label="Predicted Sales"
)

plt.legend()
plt.title("ARIMA(4,1,1) Forecast vs Actual Sales - Mid-August 2011")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.tight_layout()

plt.savefig(
    output_folder / "arima_forecast_zoomed_august.png",
    dpi=300
)

plt.show()
plt.close()

# Association Rule Mining by Cluster

# Product descriptions are required for association rules
df_rules = df_clean.dropna(subset=["Description"]).copy()

# Add k=3 cluster labels back to transaction-level data
df_rules = df_rules.merge(
    rfm[["CustomerID", "Cluster_3"]],
    on="CustomerID",
    how="left"
)

# Store top rules from all clusters
all_top_rules = []

# Loop through each k=3 cluster
for cluster_id in sorted(df_rules["Cluster_3"].dropna().unique()):


    # Filter transactions for this cluster
    df_cluster = df_rules[df_rules["Cluster_3"] == cluster_id]

    # Create basket format
    basket = (
        df_cluster.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )

    # Convert to True/False format for Apriori
    basket = basket.map(lambda x: True if x > 0 else False)

    # Skip clusters with too few transactions
    if basket.shape[0] < 10:
        continue

    # Generate frequent itemsets
    frequent_items = apriori(
        basket,
        min_support=0.02,
        use_colnames=True
    )

    if frequent_items.empty:
        
        continue

    # Generate association rules
    rules = association_rules(
        frequent_items,
        metric="lift",
        min_threshold=1.0
    )

    if rules.empty:
        
        continue

    # Remove duplicate rules (A→B vs B→A)
    rules["rule_pair"] = rules.apply(
        lambda x: tuple(sorted([tuple(x["antecedents"]), tuple(x["consequents"])])),
        axis=1
    )

    rules = rules.drop_duplicates(subset="rule_pair")
    rules = rules.drop(columns=["rule_pair"])

    # Sort strongest rules first
    rules = rules.sort_values(by="lift", ascending=False)

    # Save full rules for this cluster
    rules.to_csv(
        output_folder / f"association_rules_cluster_{int(cluster_id)}.csv",
        index=False
    )

    # Create clean top 10 table for report
    top_rules = rules.head(10)[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ].copy()

    top_rules["Cluster_3"] = int(cluster_id)

    # Convert readable text
    top_rules["antecedents"] = top_rules["antecedents"].apply(
        lambda x: ", ".join(list(x))
    )
    top_rules["consequents"] = top_rules["consequents"].apply(
        lambda x: ", ".join(list(x))
    )

    # Round metrics
    top_rules[["support", "confidence", "lift"]] = top_rules[
        ["support", "confidence", "lift"]
    ].round(4)

    # Save clean top rules for this cluster
    top_rules.to_csv(
        output_folder / f"top_rules_cluster_{int(cluster_id)}.csv",
        index=False
    )

    # Add to combined table
    all_top_rules.append(top_rules)

    

# Save combined top rules table
if all_top_rules:
    all_cluster_top_rules = pd.concat(all_top_rules, ignore_index=True)

    all_cluster_top_rules.to_csv(
        output_folder / "all_clusters_top_rules.csv",
        index=False
    )

    print("\nCombined Top Association Rules by Cluster:")
    print(all_cluster_top_rules)
else:
    print("\nNo association rules were generated for any cluster.")