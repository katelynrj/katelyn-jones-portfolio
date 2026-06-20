-- Revenue Operations Analysis
-- KPI Queries

-- KPI 1: Total Revenue Won

SELECT
    ROUND(SUM(close_value),0) AS total_revenue
FROM sales_pipeline
WHERE deal_stage = 'Won';

-- Result: $10,005,534

-- KPI 2: Total Opportunities

SELECT
    COUNT(*) AS total_opportunities
FROM sales_pipeline;


-- Result: 8,800

-- KPI 3: Win Rate

SELECT
    ROUND(
        100.0 *
        SUM(CASE WHEN deal_stage = 'Won' THEN 1 ELSE 0 END)
        /
        COUNT(*),
        2
    ) AS win_rate
FROM sales_pipeline;

-- Result: 48.16%

SELECT
    deal_stage,
    COUNT(*) AS opportunities
FROM sales_pipeli