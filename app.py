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
# Student ID: 21343455
#
# GenAI acknowledgment (required if you used GenAI; otherwise write "None"):
#   - Tools used: None
#   - How you used them: None
#
# Q1. Who is the primary business user of this app?
#     (one short sentence; e.g., an employee or a security analyst reviewing
#      inbound SMS for spam or scam-like messages)
#   YOUR_ANSWER:
#   A security analyst who determines whether the inbound SMS message is legitimate or spam.
#
# Q2. Look at the accuracy from your 100-sample test. Is accuracy enough
#     for a business to rely on this app, and why or why not?
#     2–4 short sentences max. You MUST cite the accuracy figure from
#     YOUR app run. Comment on the cost of a false positive (blocking a
#     legitimate SMS, such as a bank or one-time code message) versus a
#     false negative (missing a spam or scam-like message).
#   YOUR_ANSWER:
#   No, accuracy alone is not sufficient for a business to rely on this app, 
#   because it hides other types of errors that the model makes and ignores class imbalance.   
#   The app accuracy is 99%.
#   The cost of a false positive causes immediate operational failure and customer frustration.
#   The cost of a false negative causes higher risk in the exposure of system vulnerability.
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
STUDENT_ID = "21343455"  # STUDENT: replace with your real student ID
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
    """Return (display_label, confidence) for one SMS string."""
    # STUDENT: call the pipeline on this one text, map the raw label with
    # map_label(...), and return the display label and the confidence score.
    result = classifier(text)
    display_label = map_label(result[0]['label'])
    score = result[0]['score']
    return display_label, score


def compute_accuracy(pred_labels, true_labels):
    """Return accuracy as a float between 0 and 1."""
    # STUDENT: accuracy = (number of matches) / (number of samples).
    num_of_matches = 0
    num_of_samples = len(pred_labels)
    for index in range(num_of_samples):
        if pred_labels[index] == true_labels[index]:
            num_of_matches += 1
    accuracy = num_of_matches/num_of_samples
    return accuracy


def main():
    st.title("ISOM5160")
    st.write("Homework 2 — SMS Spam / Smishing Alert")
    # STUDENT: also show Student ID and COURSE_SECRET on the page.
    st.write(f"Student ID: {STUDENT_ID} | Course Secret: {COURSE_SECRET}")
    # Use a Hugging Face fine-grained token from Streamlit Cloud secrets
    # if one is set. Do not put the token in this file.
    hf_token = None
    if "HF_TOKEN" in st.secrets:
        hf_token = st.secrets["HF_TOKEN"]

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
        # STUDENT: if the text box is not empty, call classify_text,
        # then show the display label and the confidence.
        if len(alert_text) != 0:
            display_label, score = classify_text(classifier, alert_text)
            st.write(f"The message is classified as {display_label} with a confidence score of {score}")

    st.subheader("Test on 100 Hugging Face samples")
    st.write(
        "Click the button to load 100 messages from the Hugging Face "
        "dataset and measure accuracy."
    )
    if st.button("Run 100-sample test"):
        # STUDENT: call load_test_samples(), classify each message with
        # classify_text, then compute_accuracy. Show the accuracy and a
        # results table with at least these columns:
        # # | Text (short) | Ground truth | Prediction | Confidence | Match
        pred_label_previews = []
        pred_score_previews = []
        match_previews = []        
        text_previews, true_label_previews = load_test_samples()
        for index in range(len(text_previews)):
            label, score = classify_text(classifier,text_previews[index])
            pred_label_previews.append(label)
            pred_score_previews.append(score)
            match_previews.append(label == true_label_previews[index])
        accuracy = compute_accuracy(pred_label_previews, true_label_previews)
        st.write(f"The accuracy is {accuracy}.")
        st.write("A results table is shown below: ")
        table_data = {
            "Text(short)": text_previews,
            "Ground truth": true_label_previews,
            "Prediction": pred_label_previews,
            "Confidence": pred_score_previews,
            "Match": match_previews,
        }
        st.table(table_data, border="horizontal")


if __name__ == "__main__":
    main()
