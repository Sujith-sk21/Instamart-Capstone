# Instamart Customer Intelligence & Support Copilot

## Overview

This capstone project builds a data-driven customer intelligence and support system using synthetic Instamart data. It combines data cleaning, exploratory analysis, customer segmentation, lapse prediction, support ticket triage, policy retrieval, and a bounded refund-support agent.

The project uses Python, pandas, scikit-learn, SQLite, TF-IDF retrieval, and the supplied offline mock language model.

## Reproduce

### Requirements

- Python 3.14 or a compatible Python version
- Git
- The supplied `generate_data.py` and `mock_llm.py` files

### Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Generate the synthetic data:

```powershell
python generate_data.py
```

Run the project modules:

```powershell
python src\clean_data.py
python src\build_database.py
python src\validate_data.py
python src\eda.py
python src\segmentation.py
python src\lapse_prediction.py
python src\ticket_triage.py
python src\policy_rag.py
python src\test_policy_rag.py
```

Run commands from the project root directory. Some later modules depend on the outputs produced by earlier modules.

## Part A — Data Foundation and EDA

Part A prepares the synthetic customer and order data for analysis.

Key components:

- Data cleaning and validation scripts
- A SQLite database containing customer, product, and order data
- SQL queries in `sql/queries.sql`
- Exploratory analysis reports and plots

Key outputs include:

- `data/clean/customers_clean.csv`
- `data/clean/orders_clean.csv`
- `data/instamart.db`
- `outputs/validation.txt`
- `outputs/eda_report.txt`

The validation run reported `ALL CHECKPOINTS PASSED`.

The EDA report analysed completed orders with available order amounts. It reported a mean order amount of INR 269.92 and a median of INR 135.00.

## Part B — Customer Segmentation and Lapse Prediction

Part B builds customer-level recency, frequency, and monetary features, applies K-means clustering, and trains models to predict customer lapse.

Key components:

- RFM feature engineering
- Clustering evaluation and customer segment profiles
- Lapse prediction using logistic regression and random forest
- A retention list for selected customers
- Saved model and evaluation reports

Key outputs include:

- `outputs/customer_segments.csv`
- `outputs/clustering_evaluation.csv`
- `outputs/segment_profiles.csv`
- `outputs/segmentation_report.txt`
- `outputs/lapse_report.txt`
- `outputs/retention_list.csv`
- `outputs/lapse_model.joblib`

The current lapse evaluation selected a random forest model and reported a test F1 score of approximately 0.689. The retention list is intended to support follow-up prioritisation.

## Part C — Support Ticket Triage

Part C classifies synthetic support tickets into support categories using the supplied offline mock model and a keyword-based fallback.

Key components:

- Input text preparation and personal-data masking
- Zero-shot and few-shot classification runs
- Output validation and fallback handling
- Per-ticket triage results and summary metrics

Key outputs include:

- `outputs/triage_results.csv`
- `outputs/triage_report.json`

The current run reported 85.625% accuracy for zero-shot classification and 100% for few-shot classification on the labelled tickets included in the evaluation. These results are specific to the supplied synthetic dataset and mock model.

## Part D — Policy RAG and Refund-Support Agent

Part D retrieves policy text and demonstrates a bounded agent for refund support.

Key components:

- Loading and chunking five policy documents
- TF-IDF and cosine-similarity retrieval
- Policy answers with source references
- Order lookup and refund calculation tools
- Refund validation and human escalation
- Iteration, token, and time limits
- JSONL execution records and agent checkpoints

Key outputs include:

- `outputs/policy_answer.json`
- `outputs/policy_evaluation.csv`
- `outputs/agent_scenarios.json`
- `outputs/refund_ledger.jsonl`

The scenario test run passed all six tested scenarios: automatic refund, human escalation, ineligible delivery time, cancelled order, missing order, and maximum iteration limit.

The demonstration issued a refund of INR 128.25 for a synthetic eligible order. This is a local simulation, not a real payment transaction.

## Limitations and Next Steps

- All customer, order, product, and support data are synthetic.
- The supplied mock language model is an offline simulation and does not represent production LLM performance.
- Evaluation metrics depend on the generated dataset, split, and test cases; they should not be interpreted as guaranteed real-world performance.
- TF-IDF retrieval is a lightweight baseline and may miss relevant policy passages or return incomplete evidence.
- Refund actions are simulated locally. A production implementation would require authenticated services, secure payment integration, idempotency, audit controls, and human approval workflows.
- Further work should improve evaluation coverage, handle ambiguous or missing ticket text explicitly, validate all policy-answer scenarios, and add monitoring for model and data quality.

## Repository Structure

```text
Instamart-Capstone/
├── data/
│   ├── clean/
│   ├── policies/
│   └── instamart.db
├── outputs/
├── sql/
│   └── queries.sql
├── src/
├── generate_data.py
├── mock_llm.py
├── requirements.txt
├── .gitignore
└── README.md
```

All reported monetary amounts use INR. Outputs are generated locally from the supplied scripts and synthetic data.
