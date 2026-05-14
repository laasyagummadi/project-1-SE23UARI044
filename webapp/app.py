from flask import Flask, request, jsonify, render_template
import re

app = Flask(__name__, template_folder="templates")

# ---------------------------
# PROMPT OPTIMIZATION
# ---------------------------
ROLE_PREFIX = "You are a precise and concise summarization assistant.\n\n"
COT_SUFFIX = "\n\nThink step by step before summarizing."

def optimize_prompt(text):
    return ROLE_PREFIX + text + COT_SUFFIX


# ---------------------------
# CHUNKING
# ---------------------------
def split_sentences(text):
    if not text:
        return []
    return re.split(r'(?<=[.!?]) +', text.strip())

def chunk_text(text, size=5):
    sentences = split_sentences(text)
    return [" ".join(sentences[i:i+size]) for i in range(0, len(sentences), size)]


# ---------------------------
# MOCK MODEL (FAST + SAFE)
# ---------------------------
def generate_summary(text):
    sentences = split_sentences(text)
    return " ".join(sentences[:2]) if sentences else ""


# ---------------------------
# METRICS
# ---------------------------
def rouge_l(gen, ref):
    if not ref:
        return 0
    g = gen.split()
    r = ref.split()
    return round(len(set(g) & set(r)) / len(r), 3)

def rouge_1(gen, ref):
    return rouge_l(gen, ref)

def bleu_score(gen, ref):
    if not ref:
        return 0
    g = gen.split()
    r = ref.split()
    return round(len(set(g) & set(r)) / len(r), 3)


# ---------------------------
# ROUTES
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()

    article = data.get("article", "").strip()
    reference = data.get("reference", "").strip()

    if not article:
        return jsonify({"error": "Please enter text"}), 400

    chunks = chunk_text(article)

    baseline_summaries = []
    optimized_summaries = []

    for chunk in chunks:
        # Baseline
        baseline_summaries.append(generate_summary(chunk))

        # Optimized
        optimized_prompt = optimize_prompt(chunk)
        optimized_summaries.append(generate_summary(optimized_prompt))

    baseline_final = " ".join(baseline_summaries)
    optimized_final = " ".join(optimized_summaries)

    scores = {}

    if reference:
        scores = {
            "baseline": {
                "rouge_l": rouge_l(baseline_final, reference),
                "bleu": bleu_score(baseline_final, reference)
            },
            "optimized": {
                "rouge_l": rouge_l(optimized_final, reference),
                "bleu": bleu_score(optimized_final, reference)
            }
        }

    return jsonify({
        "baseline": baseline_final,
        "optimized": optimized_final,
        "scores": scores
    })


# ---------------------------
# RUN APP
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)