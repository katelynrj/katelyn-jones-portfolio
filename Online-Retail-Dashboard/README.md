# Online Retail Executive Dashboard
Interactive Power BI dashboard designed to analyze retail sales performance, customer activity, product trends, and geographic revenue distribution.

## Dataset

This dashboard was built using the Online Retail dataset from the UCI Machine Learning Repository, which contains transactional sales data from a UK-based online retailer between December 2010 and December 2011.

Source:
https://archive.ics.uci.edu/ml/datasets/online+retail

## Data Preparation

- Removed canceled transactions (Invoice numbers beginning with "C")
- Removed rows with negative quantities or prices
- Created revenue calculations using Quantity × Unit Price
- Created custom Month-Year and Sort Month fields for time-series analysis
- Built DAX measures for Revenue, Orders, Customers, Units Sold, and Average Order Value

## Business Objective
Analyze retail sales performance across products, customers, and countries.

## Tools Used
- Power BI
- Power Query
- DAX

## KPIs
- Total Revenue
- Total Orders
- Total Customers
- Units Sold
- Average Order Value

## Dashboard Screenshot 1 - Executive Dashboard 

<img width="640" height="357" alt="OnlineRetailExecutiveDashboard" src="https://github.com/user-attachments/assets/3859c8f4-33cd-4e33-aaba-99e7c53fec65" />

## Key Insights

### Revenue Performance
- Total revenue exceeded **$8.9 million** across approximately **19,000 orders**.
- Sales peaked during **Q4 2011**, with November generating the highest monthly revenue.

### Customer Insights
- The business served approximately **4,000 unique customers**.
- Average order value was approximately **$481 per transaction**.

### Geographic Performance
- The **United Kingdom** generated the majority of revenue.
- International sales represented a smaller portion of overall sales.

### Product Performance
- Revenue was concentrated among a small group of products.
- The top-selling products generated a disproportionately large share of total revenue.

## Dashboard Screenshot 2 - Netherlands Filter Example

<img width="632" height="355" alt="Netherlands" src="https://github.com/user-attachments/assets/59f3d485-28d2-4e5b-ac68-26cc68619ae1" />

### Key Insights

- The Netherlands generated approximately **$285K in revenue** from **94 orders** placed by **9 customers**.
- Average order value exceeded **$3,000**, substantially higher than the overall dashboard average.
- Revenue peaked during **August and October 2011**, indicating potential seasonal demand.
- Sales were concentrated among a small number of products, led by **Rabbit Night Light** and several lunch box product lines.
- A relatively small customer base generated substantial revenue, highlighting the importance of high-value international customers.

## Dashboard Screenshot 2 - Monthly Analysis Example (November 2011)

<img width="640" height="356" alt="nov11" src="https://github.com/user-attachments/assets/e6754bcf-cb0f-414b-9c80-8b7e8c34ab5e" />


### Key Insights

- November 2011 generated approximately **$1.2 million in revenue**, making it the highest-performing month in the dataset.
- Sales were driven by approximately **3,000 orders** from **2,000 customers**, demonstrating strong customer engagement during the period.
- The business sold approximately **669,000 units** in a single month while maintaining an average order value of **$437**.
- The **United Kingdom** remained the dominant market, contributing the vast majority of monthly revenue.
- Seasonal and gift-oriented products, including **Rabbit Night Light** and **Paper Chain Kit 50's Christmas**, ranked among the highest-revenue products, suggesting increased holiday purchasing activity.
- The concentration of revenue in November highlights the importance of Q4 sales performance and holiday-season demand.

## Future Enhancements

- Product Performance dashboard page
- Customer Insights dashboard page
- Geographic analysis dashboard page
- Additional DAX measures and trend analysis
