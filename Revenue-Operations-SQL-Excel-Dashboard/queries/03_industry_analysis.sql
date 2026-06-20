-- Revenue by Industry

SELECT
    a.sector,
    ROUND(SUM(sp.close_value),0) AS revenue
FROM sales_pipeline sp
JOIN accounts a
    ON sp.account = a.account
WHERE sp.deal_stage = 'Won'
GROUP BY a.sector
ORDER BY revenue DESC;

--Results:
"""retail	1867528.0
technolgy	1515487.0
medical	1359595.0
software	1077934.0
finance	950908.0
marketing	922321.0
entertainment	689007.0
telecommunications	653574.0
services	533006.0
employment	436174.0
"""
