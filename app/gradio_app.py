import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"]   = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "2"

from pathlib import Path

import joblib
import gradio as gr

from app.training.bge_model import BGELogisticRegressionModel       # noqa: F401
from app.training.bge_mlp_model import BGEMLPModel                   # noqa: F401

MODEL_DIR = Path(__file__).parent / "model"

MODELS = {
    "TF-IDF + Logistic Regression": MODEL_DIR / "sentiment_model.joblib",
    "TF-IDF + SVM":                  MODEL_DIR / "sentiment_model_svm.joblib",
    "BGE + Logistic Regression":     MODEL_DIR / "sentiment_model_bge.joblib",
    "BGE + MLP (TensorFlow)":        MODEL_DIR / "sentiment_model_bge_mlp.joblib",
}

CATEGORIES = [
    "Not specified",
    "Clothing",
    "Electronics",
    "Home & Garden",
    "Sports",
    "Beauty",
    "Toys",
    "Books",
    "Food & Grocery",
    "Other",
]

print("Loading models...")
loaded = {}
for name, path in MODELS.items():
    if path.exists():
        loaded[name] = joblib.load(path)
        print(f"  ✓  {name}")
    else:
        loaded[name] = None
        print(f"  ✗  {name}  (not trained yet — skipped)")


def _verdict_md(is_positive: bool) -> str:
    return "### 🟢 POSITIVE" if is_positive else "### 🔴 NEGATIVE"


def analyse(review_text: str, category: str):
    if not review_text.strip():
        return [{"no input": 1.0}] * len(MODELS) + [""] * len(MODELS)

    label_outputs   = []
    verdict_outputs = []
    for name, model in loaded.items():
        if model is None:
            label_outputs.append({"Model not trained yet": 1.0})
            verdict_outputs.append("### ⚪ NOT TRAINED")
        else:
            probs     = model.predict_proba([review_text])[0]
            pos_prob  = round(float(probs[1]), 4)
            neg_prob  = round(float(probs[0]), 4)
            threshold = getattr(model, "THRESHOLD", 0.5)
            label_outputs.append({"positive": pos_prob, "negative": neg_prob})
            verdict_outputs.append(_verdict_md(pos_prob >= threshold))
    return label_outputs + verdict_outputs


with gr.Blocks(title="E-Commerce Sentiment Analyser", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # E-Commerce Sentiment Analyser
        Compare four different ML models on the same review — from a simple TF-IDF baseline
        to a TensorFlow neural network on top of BGE sentence embeddings.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            review_input = gr.Textbox(
                label="Review Text",
                placeholder="Paste a product review here…",
                lines=6,
            )
            category_input = gr.Dropdown(
                choices=CATEGORIES,
                value="Not specified",
                label="Product Category (optional)",
            )
            analyse_btn = gr.Button("Analyse Sentiment", variant="primary", size="lg")

    gr.Markdown("---")
    gr.Markdown("### Results")

    model_names     = list(loaded.keys())
    output_labels   = []
    output_verdicts = []

    with gr.Row():
        for name in model_names[:2]:
            with gr.Column():
                gr.Markdown(f"**{name}**")
                vrd = gr.Markdown("")
                lbl = gr.Label(num_top_classes=2, label=name)
                output_verdicts.append(vrd)
                output_labels.append(lbl)

    with gr.Row():
        for name in model_names[2:]:
            with gr.Column():
                gr.Markdown(f"**{name}**")
                vrd = gr.Markdown("")
                lbl = gr.Label(num_top_classes=2, label=name)
                output_verdicts.append(vrd)
                output_labels.append(lbl)

    gr.Markdown("---")
    gr.Markdown(
        """
        **Model legend**
        | Model | Vectorisation | Classifier |
        |---|---|---|
        | TF-IDF + Logistic Regression | Sparse TF-IDF (50k features) | Logistic Regression |
        | TF-IDF + SVM | Sparse TF-IDF (50k features) | LinearSVC (calibrated) |
        | BGE + Logistic Regression | `BAAI/bge-base-en-v1.5` (768-dim) | Logistic Regression |
        | BGE + MLP (TensorFlow) | `BAAI/bge-base-en-v1.5` (768-dim) | Dense(128) → Dense(64) → sigmoid |
        """
    )

    all_outputs = output_labels + output_verdicts

    analyse_btn.click(
        fn=analyse,
        inputs=[review_input, category_input],
        outputs=all_outputs,
    )

    review_input.submit(
        fn=analyse,
        inputs=[review_input, category_input],
        outputs=all_outputs,
    )

    gr.Examples(
        examples=[
            ["Absolutely wonderful — silky, comfortable and true to size. Will definitely buy again!", "Clothing"],
            ["Terrible quality. Fell apart after one wash. Complete waste of money.", "Clothing"],
            ["It's okay I guess. Nothing special but does the job. Wouldn't rush to buy again.", "Electronics"],
            ["Fast delivery, great packaging, product exactly as described. Five stars!", "Home & Garden"],
            ["Not as advertised. The colour looks completely different in person and the sizing runs very small.", "Clothing"],
        ],
        inputs=[review_input, category_input],
    )


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
