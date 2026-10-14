
from pathlib import Path
import json
import re
import time
import random

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"

sys.path.insert(0, str(ROOT))
import mock_llm


ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
TOKEN_BUDGET = 6000


def redact_pii(text):
    """Redact email addresses, Indian mobile numbers, and order IDs."""
    text = str(text)
    text = re.sub(
        r"[\w.+-]+@[\w-]+\.[\w.]+",
        "[EMAIL]",
        text,
    )
    text = re.sub(
        r"\b[6-9]\d{9}\b",
        "[PHONE]",
        text,
    )
    text = re.sub(
        r"\bORD-\d{6}\b",
        "[ORDER_ID]",
        text,
        flags=re.IGNORECASE,
    )
    return text


def estimate_tokens(text):
    """Estimate tokens consistently with the offline mock model."""
    return mock_llm.est_tokens(text)


def truncate_text(text, max_tokens=MAX_INPUT_TOKENS):
    """Keep ticket input within the configured token limit."""
    words = str(text).split()
    while estimate_tokens(" ".join(words)) > max_tokens and words:
        words.pop()
    return " ".join(words)


def parse_category(content):
    """Validate JSON output, with a safe keyword fallback."""
    if not content:
        return None, "invalid"

    try:
        result = json.loads(content)
        category = result.get("category")

        if category in ALLOWED_CATEGORIES:
            return category, "json"
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # Fallback: accept a category only if exactly one allowed category
    # appears as a standalone word in the response.
    matches = re.findall(
        r"\b(billing|delivery|technical|fraud)\b",
        content.lower(),
    )
    matches = list(dict.fromkeys(matches))

    if len(matches) == 1:
        return matches[0], "keyword"

    return None, "invalid"


def call_with_retry(messages):
    """Retry transient model errors with exponential backoff and jitter."""
    for attempt in range(MAX_RETRIES):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, BASE_DELAY)

            # Keep the demo deterministic and offline.
            time.sleep(delay)

    raise RuntimeError("Model request failed after retries")


def make_messages(ticket_text, few_shot=False):
    system = (
        "CLASSIFY customer support tickets. Return JSON only with keys "
        '"category" and "reason". Category must be one of: billing, '
        "delivery, technical, fraud."
    )

    if few_shot:
        system += (
            "\nEXAMPLES:\n"
            '{"category":"fraud","reason":"unauthorized activity"}\n'
            '{"category":"billing","reason":"payment or refund"}\n'
            '{"category":"technical","reason":"app or login problem"}\n'
            '{"category":"delivery","reason":"late or missing delivery"}'
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": ticket_text},
    ]


def classify_one(ticket_text, few_shot=False):
    cleaned = redact_pii(ticket_text)
    cleaned = truncate_text(cleaned)

    messages = make_messages(cleaned, few_shot=few_shot)
    prompt_tokens = sum(
        estimate_tokens(message.get("content", ""))
        for message in messages
    )

    if prompt_tokens > TOKEN_BUDGET:
        return {
            "category": "technical",
            "method": "budget_fallback",
            "tokens": prompt_tokens,
            "valid": False,
        }

    response = call_with_retry(messages)
    content = response.choices[0].message.content
    category, method = parse_category(content)

    
    if category is None:
        # Recover the category from the redacted ticket text.
        text_lower = cleaned.lower()

        if any(word in text_lower for word in [
            "unauthorized", "fraud", "hacked", "stolen"
        ]):
            category = "fraud"
        elif any(word in text_lower for word in [
            "charged", "invoice", "bill", "refund",
            "deducted", "payment"
        ]):
            category = "billing"
        elif any(word in text_lower for word in [
            "arrived", "rider", "delivery", "delivered", "late"
        ]):
            category = "delivery"
        else:
            category = "technical"

        method = "keyword_fallback"


    return {
        "category": category,
        "method": method,
        "tokens": response.usage.total_tokens,
        "valid": method in {"json", "keyword"},
    }


def load_tickets():
    """Find and load the supplied ticket CSV."""
    candidates = [
        RAW_DIR / "tickets.csv",
        RAW_DIR / "support_tickets.csv",
    ]

    for path in candidates:
        if path.exists():
            return pd.read_csv(path), path

    raise FileNotFoundError(
        "Could not find data/raw/tickets.csv or "
        "data/raw/support_tickets.csv."
    )


def find_text_column(df):
    candidates = [
        "ticket_text", "message", "description",
        "ticket_description", "text", "issue",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    object_columns = df.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    if not object_columns:
        raise ValueError("No text column found in the ticket CSV.")

    return object_columns[0]


def find_label_column(df):
    candidates = ["category", "label", "true_category", "expected_category"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def run_experiment(tickets, text_column, few_shot):
    mock_llm.reset()
    records = []

    for _, row in tickets.iterrows():
        text = row[text_column]

        try:
            result = classify_one(text, few_shot=few_shot)
        except mock_llm.TransientLLMError:
            result = {
                "category": "technical",
                "method": "retry_exhausted_fallback",
                "tokens": 0,
                "valid": False,
            }

        records.append(result)

    return pd.DataFrame(records)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tickets, input_path = load_tickets()
    text_column = find_text_column(tickets)
    label_column = find_label_column(tickets)

    print("Input file:", input_path)
    print("Ticket count:", len(tickets))
    print("Text column:", text_column)
    print("Label column:", label_column)

    # Redaction check.
    redacted = tickets[text_column].fillna("").astype(str).map(redact_pii)
    remaining_email = redacted.str.contains(
        r"[\w.+-]+@[\w-]+\.[\w.]+", regex=True
    ).sum()
    remaining_phone = redacted.str.contains(
        r"\b[6-9]\d{9}\b", regex=True
    ).sum()
    remaining_order_id = redacted.str.contains(
        r"\bORD-\d{6}\b", case=False, regex=True
    ).sum()

    print("\nPII redaction checks:")
    print("Remaining emails:", int(remaining_email))
    print("Remaining phones:", int(remaining_phone))
    print("Remaining order IDs:", int(remaining_order_id))

    zero_shot = run_experiment(
        tickets.assign(**{text_column: redacted}),
        text_column,
        few_shot=False,
    )
    few_shot_results = run_experiment(
        tickets.assign(**{text_column: redacted}),
        text_column,
        few_shot=True,
    )

    output = tickets.copy()
    output["redacted_text"] = redacted
    output["zero_shot_category"] = zero_shot["category"]
    output["zero_shot_method"] = zero_shot["method"]
    output["zero_shot_tokens"] = zero_shot["tokens"]
    output["few_shot_category"] = few_shot_results["category"]
    output["few_shot_method"] = few_shot_results["method"]
    output["few_shot_tokens"] = few_shot_results["tokens"]

    output.to_csv(OUTPUT_DIR / "triage_results.csv", index=False)

    print("\nZero-shot method counts:")
    print(zero_shot["method"].value_counts().to_string())
    print("\nFew-shot method counts:")
    print(few_shot_results["method"].value_counts().to_string())

    if label_column:
        truth = tickets[label_column].astype(str).str.lower()
        for name, predicted in [
            ("Zero-shot", zero_shot["category"]),
            ("Few-shot", few_shot_results["category"]),
        ]:
            valid = truth.isin(ALLOWED_CATEGORIES)
            if valid.any():
                accuracy = (
                    truth[valid].to_numpy()
                    == predicted[valid].to_numpy()
                ).mean()
                print(f"{name} accuracy: {accuracy:.3%}")
            else:
                print(f"{name} accuracy: no valid labels")
    else:
        print(
            "\nNo known label column was found. "
            "Accuracy cannot be computed until the correct label column is identified."
        )

    report = {
        "ticket_count": len(tickets),
        "input_file": str(input_path.relative_to(ROOT)),
        "text_column": text_column,
        "label_column": label_column,
        "remaining_emails": int(remaining_email),
        "remaining_phones": int(remaining_phone),
        "remaining_order_ids": int(remaining_order_id),
        "zero_shot_methods": zero_shot["method"].value_counts().to_dict(),
        "few_shot_methods": few_shot_results["method"].value_counts().to_dict(),
    }

    with (OUTPUT_DIR / "triage_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\nSaved outputs/triage_results.csv")
    print("Saved outputs/triage_report.json")


if __name__ == "__main__":
    main()
