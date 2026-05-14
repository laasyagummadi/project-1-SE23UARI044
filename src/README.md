# Improving Generative AI Systems through Prompt Optimization

A modular pipeline to improve the reliability of generative AI systems using prompt optimization and chunk-based processing, evaluated through BLEU and ROUGE metrics.

## Overview
This project improves the reliability of generative AI models by:
- Reducing prompt sensitivity
- Handling long text using chunking
- Evaluating outputs using BLEU and ROUGE

---

## Methodology
1. Data preprocessing  
2. Chunk-based text splitting  
3. Prompt optimization  
4. Model inference (FLAN-T5 / GPT)  
5. Evaluation using BLEU & ROUGE  

---

## Dataset
A curated dataset of text articles categorized based on length (short, medium, long) to evaluate model performance under varying input sizes.

---

## Results
Evaluation results are stored in the `results/` directory and include:
- BLEU scores  
- ROUGE scores  
- Graph visualizations  

The results show that chunk-based processing improves consistency for long inputs, while prompt optimization reduces variability in generated outputs.

---

## How to Run

Run the pipeline using:

```bash
pip install -r requirements.txt
cd src
python pipeline.py
