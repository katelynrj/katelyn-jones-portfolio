# Customer Segmentation and Behavioral Analysis

## Project Overview

This project analyzes customer purchasing behavior using the UCI Online Retail dataset. The objective was to identify meaningful customer segments, predict customer spending, forecast future sales, and uncover product purchasing relationships using machine learning and statistical techniques.

---

## Business Objective

Retail organizations often struggle to identify high-value customers, predict purchasing behavior, and understand product relationships. This project combines customer segmentation, predictive modeling, forecasting, and association rule mining to support data-driven decision-making.

---

## Dataset

**Source:** UCI Machine Learning Repository – Online Retail Dataset

**Dataset Characteristics**

* 541,909 transaction records
* Customer purchase history
* Product descriptions
* Transaction dates
* Country information

### Data Cleaning

The dataset was cleaned by:

* Removing cancelled orders
* Removing missing customer identifiers
* Removing invalid quantities and prices
* Creating transaction-level revenue calculations

---

## Tools & Technologies

* Python
* Pandas
* NumPy
* Scikit-Learn
* Statsmodels
* Mlxtend
* Matplotlib
* Seaborn

---

## Methodology

### 1. Feature Engineering

Created traditional RFM variables:

* Recency
* Frequency
* Monetary Value

Additional behavioral features:

* Product Diversity
* Average Purchase Value
* Average Days Between Transactions

### 2. Customer Segmentation

Applied:

* K-Means Clustering
* OPTICS Clustering
* t-SNE Visualization

### 3. Predictive Modeling

Built Linear Regression models to predict customer spending behavior.

### 4. Forecasting

Implemented ARIMA forecasting models to predict future sales trends.

### 5. Association Rule Mining

Used Apriori algorithms to identify frequently purchased product combinations.

---

## Key Results

### Customer Segmentation

* K-Means achieved the strongest clustering performance.
* Two major customer groups emerged:

  * Active High-Value Customers
  * Inactive Low-Value Customers

### Predictive Modeling

* Frequency showed the strongest relationship with spending behavior.
* Removing extreme outliers improved model performance substantially.

### Forecasting

* ARIMA successfully captured overall sales trends.
* Forecast accuracy decreased during extreme sales spikes.

### Product Relationships

* Association rule mining identified strong product bundling opportunities.
* Several tea set products demonstrated extremely strong co-purchase behavior.

---

## Business Impact

This analysis demonstrates how customer analytics can be used to:

* Improve customer targeting
* Support personalized marketing campaigns
* Identify product bundling opportunities
* Improve sales forecasting
* Support inventory planning

---

## Repository Contents

* `finalproject.py` — Main project workflow
* `withtheaddfeatures.py` — Extended feature engineering experiments
* `customer_behavior_presentation.pdf` — Final presentation

---

## Future Improvements

* Seasonal forecasting models
* Advanced machine learning algorithms
* Real-time recommendation systems
* Additional behavioral feature exploration
