"""Part D scenario tests for the Instamart refund agent."""

import json
import sqlite3
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mock_llm
import policy_rag



ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"
OUTPUT_PATH = ROOT / "outputs" / "agent_scenarios.json"
LEDGER_PATH = ROOT / "outputs" / "test_refund_ledger.jsonl"


def find_order(connection, conditions):
    query = f"""
        SELECT order_id, status, order_amount, delivery_minutes
        FROM orders
        WHERE {conditions}
        ORDER BY order_id
        LIMIT 1
    """
    row = connection.execute(query).fetchone()
    return dict(row) if row else None


def run_scenario(name, order, expected_prefix, chunks):
    if order is None:
        return {
            "scenario": name,
            "status": "SKIPPED",
            "reason": "No matching order found.",
        }

    mock_llm.reset()

    state = policy_rag.run_refund_agent(
        user_request=(
            f"Please process a refund for order {order['order_id']}."
        ),
        chunks=chunks,
        ledger_path=LEDGER_PATH,
        run_id=name,
    )

    answer = state.get("final_answer") or ""

    return {
        "scenario": name,
        "order_id": order["order_id"],
        "expected_prefix": expected_prefix,
        "actual_answer": answer,
        "tools_executed": len(state.get("executed_tools", [])),
        "status": "PASS" if answer.startswith(expected_prefix) else "FAIL",
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    documents = policy_rag.load_policies()
    chunks = policy_rag.chunk_policies(documents)

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        scenarios = [
            (
                "S1_auto_refund",
                "status = 'completed' AND delivery_minutes > 45 "
                "AND order_amount > 0 AND order_amount <= 1000",
                "REFUND:",
            ),
            (
                "S2_human_escalation",
                "status = 'completed' AND delivery_minutes > 45 "
                "AND order_amount > 1000",
                "ESCALATE:",
            ),
            (
                "S3_within_45_minutes",
                "status = 'completed' AND delivery_minutes <= 45",
                "DENIED:",
            ),
            (
                "S4_cancelled_order",
                "status = 'cancelled'",
                "DENIED:",
            ),
        ]

        results = []

        for name, conditions, expected_prefix in scenarios:
            order = find_order(connection, conditions)
            results.append(
                run_scenario(name, order, expected_prefix, chunks)
            )

    mock_llm.reset()

    missing_state = policy_rag.run_refund_agent(
        user_request="Please refund order ORD-999999.",
        chunks=chunks,
        ledger_path=LEDGER_PATH,
        run_id="S5_missing_order",
    )

    missing_answer = missing_state.get("final_answer") or ""

    results.append({
        "scenario": "S5_missing_order",
        "actual_answer": missing_answer,
        "tools_executed": len(
            missing_state.get("executed_tools", [])
        ),
        "status": (
            "PASS" if missing_answer.startswith("NOT_FOUND:") else "FAIL"
        ),
    })

    mock_llm.reset()

    loop_state = policy_rag.run_refund_agent(
        user_request="Test the bounded agent loop [LOOP].",
        chunks=chunks,
        ledger_path=LEDGER_PATH,
        run_id="S6_max_iterations",
        max_iterations=3,
    )

    loop_answer = loop_state.get("final_answer") or ""

    results.append({
        "scenario": "S6_max_iterations",
        "actual_answer": loop_answer,
        "tools_executed": len(
            loop_state.get("executed_tools", [])
        ),
        "status": (
            "PASS"
            if loop_answer == "FALLBACK_MAX_ITERATIONS"
            else "FAIL"
        ),
    })

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print(f"\nSaved results to: {OUTPUT_PATH}")
    print(
        "\nPassed:",
        sum(result["status"] == "PASS" for result in results),
        "of",
        len(results),
        "scenarios",
    )


if __name__ == "__main__":
    main()