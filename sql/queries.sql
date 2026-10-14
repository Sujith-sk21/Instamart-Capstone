
-- Q1. Count customers in each city and gender combination.
SELECT city, gender, COUNT(*) AS customer_count
FROM customers
GROUP BY city, gender
ORDER BY city, gender;


-- Q2. Total completed-order revenue by city and product category.
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue_inr DESC;


-- Q3. Products price_inrd above the average product price_inr.
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;


-- Q4. Customers with no completed orders.
SELECT c.customer_id, c.city
FROM customers c
WHERE NOT EXISTS (
    SELECT 1
    FROM orders o
    WHERE o.customer_id = c.customer_id
      AND o.status = 'completed'
)
ORDER BY c.customer_id;


-- Q5. Customers whose completed-order spending exceeds
-- the average completed-order spending per customer.
WITH customer_spending AS (
    SELECT
        c.customer_id,
        COALESCE(SUM(o.order_amount), 0) AS total_spend
    FROM customers c
    LEFT JOIN orders o
        ON c.customer_id = o.customer_id
       AND o.status = 'completed'
    GROUP BY c.customer_id
)
SELECT customer_id, ROUND(total_spend, 2) AS total_spend_inr
FROM customer_spending
WHERE total_spend > (
    SELECT AVG(total_spend) FROM customer_spending
)
ORDER BY total_spend DESC;


-- Q6. Rank customers by completed-order spending within each city.
WITH customer_spending AS (
    SELECT
        c.city,
        c.customer_id,
        COALESCE(SUM(o.order_amount), 0) AS total_spend
    FROM customers c
    LEFT JOIN orders o
        ON c.customer_id = o.customer_id
       AND o.status = 'completed'
    GROUP BY c.city, c.customer_id
)
SELECT
    city,
    customer_id,
    ROUND(total_spend, 2) AS total_spend_inr,
    RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS spending_rank,
    ROW_NUMBER() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC, customer_id
    ) AS row_number
FROM customer_spending
ORDER BY city, spending_rank, customer_id;


-- Q7. Daily completed-order revenue and its seven-day moving average.
WITH daily_revenue AS (
    SELECT
        date(order_ts) AS order_day,
        SUM(order_amount) AS revenue_inr
    FROM orders
    WHERE status = 'completed'
    GROUP BY date(order_ts)
)
SELECT
    order_day,
    ROUND(revenue_inr, 2) AS revenue_inr,
    ROUND(
        AVG(revenue_inr) OVER (
            ORDER BY order_day
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS moving_average_7_days
FROM daily_revenue
ORDER BY order_day;
