# COMPAS Recidivism Analysis — Explainability Audit
 
A continuation of the Lecture 1 COMPAS analysis, this notebook applies explainability and interpretability methods to a gradient-boosted tree model trained on the COMPAS recidivism dataset. SHAP values, LIME explanations, and DiCE counterfactuals are used to audit model behavior across racial groups.
 
The analysis is part of Individual Homework 2 for DNSC 6330: Responsible Machine Learning at The George Washington University.
 
---

## AI Use Disclaimer
 
Portions of this notebook were developed with the assistance of an AI assistant. AI tools were used to help translate R code into Python, debug errors, and structure code comments. All outputs were reviewed, tested, and verified by me. The analysis, interpretation of results, and governance memo reflect my own understanding of the material.
 
 
## Purpose
 
Building on the fairness diagnostics from Assignment 1, this notebook examines *why* the model makes individual predictions and whether affected defendants have access to meaningful recourse. The analysis compares explanation methods across four defendants — highest and lowest risk in each of two racial groups — and produces a governance memo addressed to a hypothetical court auditor.
 
---
 
## Libraries Used
 
| Library | Purpose |
|---|---|
| `pandas` | Data manipulation and filtering |
| `numpy` | Numerical operations |
| `scikit-learn` | Model pipeline, preprocessing, gradient-boosted tree, train/test split |
| `shap` | SHAP values, beeswarm summary plot, waterfall plots |
| `lime` | Local interpretable model-agnostic explanations |
| `dice-ml` | Counterfactual explanation generation |
| `matplotlib` | Plotting and visualization |
 
---
 
## How to Reproduce
 
### 1. Clone the repository
 
```bash
git clone <your-repo-url>
cd <repo-folder>
```
 
### 2. Install dependencies
 
```bash
pip install pandas numpy scikit-learn shap lime dice-ml matplotlib
```
 
### 3. Run the notebook
 
Open `Lecture_02_Alignment_Python.ipynb` in Jupyter or Google Colab and run all cells top to bottom. The dataset is loaded directly from ProPublica's public GitHub repository so no local data files are needed.
 
```
Dataset URL: https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
```
 
---
 
## Analytical Pipeline
 
### 1. Model Setup
- Rebuilds the preprocessing pipeline and gradient-boosted tree classifier from Lecture 1
- Defines numeric and categorical feature lists
- Performs an 80/20 train/test split with stratification on the recidivism outcome
 
### 2. SHAP Analysis
- Computes SHAP values for all test set instances using a model-agnostic explainer
- Produces a global beeswarm summary plot showing feature importance across the full test set
- Generates individual waterfall plots for four defendants: highest and lowest risk in each of the Black and white racial groups
- Key finding: `race_African-American` carries a small but consistent positive SHAP value, pushing predicted risk above average even after controlling for other features
 
### 3. LIME Analysis
- Fits a LIME explainer on the encoded training data
- Generates local explanations for the same four defendants
- Compares LIME and SHAP feature attributions, identifying where they agree and where they diverge
- Key finding: LIME and SHAP broadly agree on high-impact features like age and priors but diverge on lower-ranked features, reflecting LIME's local-only fidelity guarantee
 
### 4. DiCE Counterfactuals
- Generates three counterfactuals per defendant showing the minimal feature changes required to flip the prediction
- Features allowed to vary: `age`, `priors_count`, `c_charge_degree`
- Immutable features explicitly excluded: `race`, `sex`
- Key finding: the most common recourse path for high-risk defendants requires increasing age to 69-70, which is not actionable recourse
 
### 5. Governance Memo
- 300-word memo addressed to a hypothetical court auditor summarizing model behavior, method limitations, and monitoring recommendations
 
---
 
## Key Findings
 
SHAP analysis confirms a persistent race signal in the model. Black defendants face a false positive rate of 0.367 compared to 0.104 for white defendants, and `race_African-American` consistently pushes individual predictions above the average even when race is not the top-ranked feature. This points to proxy discrimination through correlated features like `priors_count` rather than direct use of race.
 
DiCE counterfactuals did not require changes to immutable features, which is a positive finding. However the most common recourse recommendation for high-risk defendants was a large increase in age, which is not actionable. Counterfactual generation should be constrained to genuinely actionable features in any real deployment.
 
---
 
## Notes on Method Limitations
 
- SHAP values reflect marginal contributions relative to the average prediction, not causal effects. A zero SHAP value for race does not mean the model is race-neutral.
- LIME only guarantees local fidelity. Results should not be used to describe global model behavior.
- DiCE counterfactuals can be mathematically valid but practically useless if they require changes to features a defendant cannot control.
- All three methods are diagnostic tools. They must be embedded in a documented audit process to constitute meaningful governance.
