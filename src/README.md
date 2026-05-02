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
Custom dataset of articles categorized by length.

---

## Results
Evaluation results are stored in the `results/` directory and include:
- BLEU scores  
- ROUGE scores  
- Graph visualizations  

---

## How to Run

```bash
pip install -r requirements.txt
cd src
python pipeline.py
