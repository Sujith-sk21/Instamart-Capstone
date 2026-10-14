Instamart Customer Intelligence & Support Copilot

Overview

This capstone project builds a data-driven customer intelligence and support system using synthetic Instamart data. It covers data cleaning, exploratory data analysis, customer segmentation, lapse prediction, support ticket triage, policy retrieval, and a bounded refund-support agent.

Technology stack: Python, pandas, scikit-learn, SQLite, TF-IDF retrieval, and the supplied offline mock language model.

All data and transactions are synthetic or simulated. No real refunds or external LLM API calls are performed.

Reproduce

Requirements

Python 3.14 or a compatible Python version

Git

The supplied generate_data.py and mock_llm.py files

Setup

Run these commands from the project root in PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python generate_data.py

Run the project

python src\clean_data.py
python src\build_database.py
python src\validate_data.py
python src\eda.py
python src\segmentation.py
python src\lapse_prediction.py
python src\ticket_triage.py
python src\policy_rag.py
python src\test_policy_rag.py

Run the modules from the project root and in order, because later stages depend on earlier outputs. The Part D script generates policy evaluation results, agent scenario results, execution logs, refund ledger entries, and checkpoints.

Part A — Data Foundation and EDA

Part A cleans and validates the synthetic customer and order data, builds a SQLite database, and performs exploratory analysis.

Key components:

Data cleaning and validation

SQLite database and SQL queries in sql/queries.sql

Exploratory analysis reports and plots

Key outputs:

data/clean/customers_clean.csv

data/clean/orders_clean.csv

data/instamart.db

outputs/validation.txt

outputs/eda_report.txt

The recorded validation run reported ALL CHECKPOINTS PASSED. The EDA report analysed completed orders with available order amounts and reported a mean of INR 269.92 and a median of INR 135.00.

Part B — Customer Segmentation and Lapse Prediction

Part B engineers recency, frequency, and monetary (RFM) features, applies K-means clustering, and evaluates models for predicting customer lapse.

Key outputs:

outputs/customer_segments.csv

outputs/clustering_evaluation.csv

outputs/segment_profiles.csv

outputs/segmentation_report.txt

outputs/lapse_report.txt

outputs/retention_list.csv

outputs/lapse_model.joblib

The recorded evaluation selected a random forest and reported a test F1 score of approximately 0.689. The retention list supports prioritisation of customers for follow-up. These metrics apply to the synthetic dataset and its evaluation setup.

Part C — Support Ticket Triage

Part C classifies synthetic support tickets using the supplied offline mock model and a keyword-based fallback.

Key components:

Input text preparation and personal-data masking

Zero-shot and few-shot classification

Output validation and fallback handling

Per-ticket results and summary metrics

Key outputs:

outputs/triage_results.csv

outputs/triage_report.json

The recorded run reported 85.625% zero-shot accuracy and 100% few-shot accuracy on the labelled tickets included in the evaluation. These results are specific to the synthetic dataset and mock model and should not be interpreted as real-world performance guarantees.

Part D — Policy RAG and Refund-Support Agent

Part D retrieves policy evidence and demonstrates a bounded agent for refund support.

Key components:

Five policy documents, chunking, and TF-IDF/cosine-similarity retrieval

Policy answers with source references

Order lookup and refund calculation tools

Refund eligibility checks and human escalation

Iteration, token, and timeout limits

JSONL execution records, tool logs, and checkpoints

Crash/resume demonstration and zero-overlap evaluation

Acceptance scenario results

The latest recorded run passed all seven scenarios:

Scenario

Expected outcome

Result

S1 — Automatic refund

Refund issued

PASS — INR 128.25

S2 — Human escalation

Escalation for refund above INR 1,000

PASS — INR 1,999 requires approval

S3 — Delivery within 45 minutes

Refund denied

PASS

S4 — Cancelled order

Refund denied

PASS

S5 — Missing order

Order not found

PASS

S6 — Bounded iteration test

FALLBACK_MAX_ITERATIONS

PASS

S7 — Default-limit loop test

FALLBACK_MAX_ITERATIONS

PASS

Fallback and recovery demonstrations

Demonstration

Recorded outcome

Token limit

FALLBACK_TOKEN_LIMIT

Timeout

FALLBACK_TIMEOUT

Crash/resume

PASS — exactly 4 total tool executions

Zero-overlap evaluation

7 questions tested; unsupported lunar-delivery question identified as a no-match

The crash/resume demonstration recorded two tool executions before the simulated crash and four total after resumption. The final answer was REFUND: INR 128.25 issued.

Key outputs

outputs/policy_answer.json

outputs/policy_evaluation.csv

outputs/zero_overlap_evaluation.csv

outputs/agent_scenarios.json

outputs/agent_runs.jsonl

outputs/agent_tool_log.jsonl

outputs/refund_ledger.jsonl

state/ — agent checkpoints

The refund ledger records an issued refund of INR 128.25 and an escalation requiring human approval for INR 1,999. These are simulated local actions, not real payment transactions.

Limitations and Next Steps

All customer, order, product, and support data are synthetic.

The supplied mock language model is an offline simulation and does not represent production LLM performance.

Evaluation metrics depend on the generated data, evaluation split, and test cases.

TF-IDF retrieval is a lightweight baseline and may miss relevant passages or return incomplete evidence.

Refund actions are simulated locally. A production system would need authentication, secure payment integration, idempotency, audit controls, and human approval workflows.

Future improvements include broader evaluation coverage, explicit handling of missing ticket text, stronger retrieval evaluation, and monitoring of model and data quality.

Repository Structure

Instamart-Capstone/
├── data/
│   ├── clean/
│   ├── policies/
│   └── instamart.db
├── outputs/
├── sql/
│   └── queries.sql
├── src/
├── state/
├── generate_data.py
├── mock_llm.py
├── requirements.txt
├── .gitignore
└── README.md