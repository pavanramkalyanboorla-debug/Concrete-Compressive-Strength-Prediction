<div align="center">

# 🧱 Concrete Mix Optimizer

**AI-powered concrete mix design with engineering constraints, optimization, and explainability**

[![FastAPI](https://img.shields.io/badge/API-FastAPI-green)]()
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)]()
[![Docker](https://img.shields.io/badge/docker-ready-blue)]()
[![HuggingFace](https://img.shields.io/badge/🤗-Live_App-yellow)]()

</div>

---

> **This project replaces traditional trial-and-error concrete mix design with an AI-driven optimization system.**

---

## 🚀 What This Project Actually Does

Most ML projects in this domain stop at:

> “Given a mix → predict strength”

That’s not useful in real engineering.

### This system does the reverse:

> **Given a target strength → generate a valid, cost-efficient mix that satisfies real-world constraints**

---

## 🧠 Core Idea

This is not just ML.

It’s a **decision system** combining:

- 🧱 Civil engineering principles (absolute volume method)
- 🛑 Hard constraints (durability, workability, mix limits)
- 🤖 Machine learning (strength prediction)
- 🔍 Optimization (search best mix)
- 📊 Explainability (why the mix works)

---

## 🔥 Why This Stands Out

| Typical ML Project | This Project |
|-------------------|-------------|
| Predicts output | Designs input |
| Ignores physics | Uses engineering constraints |
| Single answer | Trade-off exploration (Pareto) |
| Black box | Explainable with SHAP |

👉 This is closer to **engineering decision intelligence**, not just regression.

---

## ⚙️ How It Works

### 1. Feature Engineering
From raw mix:

- Water-Binder Ratio  
- Log(Age)  
- Cement × Age  
- SCM Ratio  

---

### 2. Preprocessing (Critical Fix)

A major issue was identified and fixed:

- Earlier: log transformation was applied in EDA but not during prediction  
- Result: inconsistent inputs → wrong predictions  

### ✅ Fix:
- `log1p` is now part of the **preprocessor pipeline**
- Same transformation is applied in:
  - Training
  - Optimization
  - Prediction
  - SHAP

---

### 3. Model Training

From your actual logs:

- Dataset: 1130 samples  
- Features: 12 (8 raw + 4 engineered)  
- Algorithm: Gradient Boosting (best selected)  
