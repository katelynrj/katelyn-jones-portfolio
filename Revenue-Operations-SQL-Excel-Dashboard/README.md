# Revenue Operations Dashboard | SQL & Excel

## Project Overview

This project analyzes sales pipeline performance using SQL and Excel. The goal was to evaluate revenue generation, sales effectiveness, pipeline health, and manager performance using a CRM sales opportunities dataset.

The analysis includes KPI development, SQL-based revenue calculations, Excel dashboard creation, and an interactive opportunity explorer built with Pivot Tables and slicers.

This project demonstrates how SQL and Excel can be used to transform raw CRM data into actionable business insights for sales and revenue operations teams.

## Dataset

This project uses the CRM Sales Opportunities dataset available on Kaggle:

Dataset: CRM Sales Opportunities  
Source: https://www.kaggle.com/datasets/innocentmfa/crm-sales-opportunities

**The dataset contains sales opportunity records, account information, sales team assignments, products, deal stages, and revenue values used to analyze pipeline performance and revenue generation.
**---

## Tools Used

- SQL (SQLite)
- DB Browser for SQLite
- Microsoft Excel
- Pivot Tables
- Pivot Charts
- Slicers
- Data Visualization

---

## Business Questions

This project answers the following questions:

- How much revenue has been generated from closed-won opportunities?
- What is the overall sales win rate?
- Which products generate the most revenue?
- Which industries contribute the most revenue?
- Which managers generate the highest revenue?
- What does the current pipeline look like across sales stages?

---

## Key Metrics

| Metric | Value |
|----------|----------:|
| Revenue | $10,005,534 |
| Total Opportunities | 8,800 |
| Closed-Won Rate | 63.15% |
| Average Won Deal Size | $2,361 |
| Active Pipeline | 2,089 |

---

## Dashboard Screenshot

<img width="1155" height="798" alt="Revenue_Operations_Dashboard" src="https://github.com/user-attachments/assets/62dee940-c9d3-4a5e-8edf-e891e394507e" />


---

## Key Findings

- Revenue exceeded $10 million across closed-won opportunities.
- The sales organization achieved a 63.15% closed-won rate.
- GTXPro generated the highest product revenue ($3.5M).
- Retail was the highest revenue-producing industry.
- Melvin Marxen generated the highest manager revenue ($2.25M).
- More than 2,000 opportunities remain active in the sales pipeline.

---

## SQL Techniques Demonstrated

- Aggregations
- Joins
- Filtering
- CASE Statements
- GROUP BY
- ORDER BY
- KPI Calculations

Example query used to calculate Closed-Won Rate:

```sql
SELECT
    ROUND(
        100.0 *
        SUM(CASE WHEN deal_stage = 'Won' THEN 1 ELSE 0 END)
        /
        SUM(CASE WHEN deal_stage IN ('Won','Lost') THEN 1 ELSE 0 END),
        2
    ) AS closed_win_rate
FROM sales_pipeline;
```

---

## Interactive Opportunity Explorer

An additional Excel report was created using Pivot Tables and slicers that allows users to filter opportunities by:

- Manager
- Sales Agent
- Product
- Deal Stage

This provides a user-friendly way to explore sales performance and opportunity details.

<img width="772" height="641" alt="Opportunity_Explorer" src="https://github.com/user-attachments/assets/8433d651-4cca-473c-9946-eb9577e1e3ad" />

---

## Files Included

- queries.sql
- Revenue_Operations_Dashboard.xlsx
- dashboard_screenshot.png
- README.md

---
