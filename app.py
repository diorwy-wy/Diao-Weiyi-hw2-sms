"""
ISOM5160 Homework 2 — SMS Spam / Smishing Alert App
Course: ISOM5160-L3 Python Programming for Business Applications, Fall 2026 (PG ICSM)

INSTRUCTIONS (read carefully)
- Complete the marked STUDENT sections and the function bodies below.
- Keep these function names: load_model, load_test_samples, classify_text,
  compute_accuracy, main.
- Call main() at the bottom of this file (the starter already does this).
- In GitHub and on Streamlit Cloud, this file must be named app.py.
- Submit to Canvas: a zip file containing app.py, requirements.txt, and
  Homework2_<StudentID>_link.txt (Streamlit Cloud URL and GitHub URL).
  Put those three files at the root of the zip, not inside a subfolder.
- Do NOT fine-tune a model. Do NOT scrape the web.
- Do NOT change DATASET_ID, MODEL_ID, N_TEST, or SPLIT_SEED.
- When you deploy, open Advanced settings and set Python to 3.12.
  If the app is already deployed, delete it and redeploy — the Python
  version cannot be changed in place.
- If the app calls a Hugging Face model, store a fine-grained Hugging Face
  API token in Streamlit Cloud secrets as HF_TOKEN. Do not put the token
  in app.py, requirements.txt, or any public file.
- First run may download the model and dataset (needs network). CPU is OK.

SECRET PHRASE (must appear on the app page): ISOM5160-HW2-SMS
"""

# =============================================================================
# STUDENT ANSWERS BLOCK (fill in; keep this in the SAME .py file)
# =============================================================================
# Student ID: 21318838
#
# GenAI acknowledgment (required if you used GenAI; otherwise write "None"):
#   - Tools used: ChatGPT
#   - How you used them: it explained the starter structure, helped me draft
#     the marked parts of main()
#
# Q1. Who is the primary business user of this app?
#     (one short sentence; e.g., an employee or a security analyst reviewing
#      inbound SMS for spam or scam-like messages)
#   YOUR_ANSWER:A fraud-prevention analyst in a bank's customer-protection team,
#     screening suspicious inbound SMS before customers act on them.
#
# Q2. Look at the accuracy from your 100-sample test. Is accuracy enough
#     for a business to rely on this app, and why or why not?
#     2–4 short sentences max. You MUST cite the accuracy figure from
#     YOUR app run. Comment on the cost of a false positive (blocking a
#     legitimate SMS, such as a bank or one-time code message) versus a
#     false negative (missing a spam or scam-like message).
#   YOUR_ANSWER:My run scored 99.00% (99 of 100), which suits triage but not auto-blocking. 
#    A false positive blocks a real bank or one-time-code SMS, which can lock a customer out and cost more in support than the spam it saves. 
#    A false negative lets a scam through; my one miss was a spam text labelled legitimate with 92.73% confidence. 
#    So I would use this app to flag messages for an analyst, not to block them automatically.
#
# =============================================================================

import streamlit as st
from datasets import load_dataset
from transformers import pipeline

# Locked by the instructor — do not change these four values.
DATASET_ID = "ucirvine/sms_spam"
MODEL_ID = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
N_TEST = 100
SPLIT_SEED = 42

# Course identity (required)
STUDENT_ID = "21318838"  # STUDENT: replace with your real student ID
COURSE_SECRET = "ISOM5160-HW2-SMS"  # do not change


def map_label(raw_label):
    """Map model or dataset labels to the two display names used in this app."""
    text = str(raw_label).strip().lower()
    if text in {"ham", "legitimate", "label_0", "0"}:
        return "legitimate"
    if text in {"spam", "smishing", "label_1", "1"}:
        return "spam"
    return str(raw_label)


@st.cache_resource
def load_model(token):
    """Load the SMS text-classification model once and reuse it."""
    # This function is provided. Keep device="cpu" and
    # model_kwargs={"low_cpu_mem_usage": False} so Streamlit Cloud can load
    # the model on CPU.
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        token=token,
        device="cpu",
        truncation=True,
        max_length=512,
        model_kwargs={"low_cpu_mem_usage": False},
    )
    return classifier


@st.cache_data
def load_test_samples():
    """
    Load the Hugging Face dataset and take a fixed set of 100 samples.

    The Hub copy has only a train split, so this homework holds out 100
    rows after a shuffle with SPLIT_SEED. Do not change the seed or N_TEST,
    so every student is tested on the same 100 messages.
    """
    raw = load_dataset(DATASET_ID, split="train")
    picked = raw.shuffle(seed=SPLIT_SEED).select(range(N_TEST))
    texts = list(picked["sms"])
    true_labels = [map_label(label) for label in picked["label"]]
    return texts, true_labels


def classify_text(classifier, text):
    result = classifier(text)[0]
    display_label = map_label(result["label"])
    score = float(result["score"])
    return display_label, score


def compute_accuracy(pred_labels, true_labels):
    if len(true_labels) == 0:
        return 0.0
    matches = sum(1 for pred, truth in zip(pred_labels, true_labels) if pred == truth)
    accuracy = matches / len(true_labels)
    return accuracy


def main():
    st.title("ISOM5160")
    st.write("Homework 2 — SMS Spam / Smishing Alert")
    # STUDENT: also show Student ID and COURSE_SECRET on the page.
    st.write(f"**Student ID:** {STUDENT_ID}")
    st.write(f"**Course secret:** {COURSE_SECRET}")

    # Use a Hugging Face fine-grained token from Streamlit Cloud secrets
    # if one is set. Do not put the token in this file.
    hf_token = None
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        # No secrets.toml in this environment (e.g. a first local run).
        # The model is public, so the app can continue without a token.
        hf_token = None

    # STUDENT: load the model with load_model(hf_token)
    classifier = load_model(hf_token)

    alert_text = st.text_area(
        "SMS message",
        value=(
            "WINNER!! You have been selected to receive a $900 prize. "
            "Claim now at http://prize-alert.example or the offer expires."
        ),
        height=140,
    )

    if st.button("Classify SMS"):
        if not alert_text.strip():
            st.warning("Please paste an SMS message first.")
        else:
            label, score = classify_text(classifier, alert_text)
            if label == "spam":
                st.error(f"Prediction: {label}")
            else:
                st.success(f"Prediction: {label}")
            st.write(f"Confidence: {score:.2%}")

    st.subheader("Test on 100 Hugging Face samples")
    st.write(
        "Click the button to load 100 messages from the Hugging Face "
        "dataset and measure accuracy."
    )
    if st.button("Run 100-sample test"):
        texts, true_labels = load_test_samples()
        predictions, confidences = [], []

        # Classify one message at a time; only runs when the button is clicked.
        bar = st.progress(0.0, text="Classifying messages…")
        for i, text in enumerate(texts, start=1):
            label, score = classify_text(classifier, text)
            predictions.append(label)
            confidences.append(score)
            if i % 10 == 0:
                bar.progress(
                    i / len(texts), text=f"Classified {i}/{len(texts)} messages"
                )
        bar.progress(1.0, text="Finished")
        bar.empty()

        # Build one readable row per message for the results table.
        rows = []
        for i, (text, truth, pred, conf) in enumerate(
            zip(texts, true_labels, predictions, confidences), start=1
        ):
            preview = " ".join(str(text).split())
            if len(preview) > 60:
                preview = preview[:57] + "..."
            rows.append(
                {
                    "#": i,
                    "Text (short)": preview,
                    "Ground truth": truth,
                    "Prediction": pred,
                    "Confidence": round(float(conf), 4),
                    "Match": "yes" if pred == truth else "no",
                }
            )

        # Keep the result in session_state so it survives Streamlit reruns
        # instead of classifying all 100 messages again on every interaction.
        st.session_state["hw2_accuracy"] = compute_accuracy(predictions, true_labels)
        st.session_state["hw2_rows"] = rows

    if "hw2_rows" in st.session_state:
        accuracy = st.session_state["hw2_accuracy"]
        rows = st.session_state["hw2_rows"]
        correct = sum(1 for row in rows if row["Match"] == "yes")
        st.metric("Accuracy on the 100-sample test", f"{accuracy:.2%}")
        st.caption(
            f"{correct} of {len(rows)} messages matched the ground truth. "
            f"Fixed sample: seed {SPLIT_SEED}, first {N_TEST} rows of the train split."
        )
        st.dataframe(rows)


if __name__ == "__main__":
    main()
