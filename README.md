# COMPAS Recidivism Analysis — Explainability Audit
 
A continuation of the Lecture 1 COMPAS analysis, this notebook applies explainability and interpretability methods to a gradient-boosted tree model trained on the COMPAS recidivism dataset. SHAP values, LIME explanations, and DiCE counterfactuals are used to audit model behavior across racial groups.
 
The analysis is part of Individual Homework 2 for DNSC 6330: Responsible Machine Learning at The George Washington University.

## Table of Contents

1. [AI Use Disclaimer](#ai-use-disclaimer)
2. [Purpose](#purpose)
3. [Libraries Used](#libraries-used)
4. [How to Reproduce](#how-to-reproduce)
5. [Analytical Pipeline](#analytical-pipeline)
6. [Governance Memo](#governance-memo)
7. [Notes on Method Limitations](#notes-on-method-limitations)
 
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
 
## Key Findings/Governance Memo
 
This memo summarizes findings from an explainability audit of a gradient-boosted tree model trained on the COMPAS recidivism dataset. Three methods were applied: SHAP values, LIME, and DiCE counterfactuals. The goal was to assess whether the model's predictions are transparent, fair, and amenable to actionable recourse.

What the Explanations Reveal

SHAP analysis shows that days_b_screening_arrest, decile_score, and score_text are the strongest global predictors. Notably, race_African-American carries a small but consistent positive SHAP value across the test set, meaning being Black pushes predicted recidivism risk above average even after controlling for other features. This is consistent with the FPR disparity identified in the diagnostic analysis, where Black defendants faced a false positive rate of 0.367 compared to 0.104 for white defendants.
LIME explanations for individual defendants were broadly consistent with SHAP on high-impact features like age and priors, but diverged on lower-ranked features. This divergence is expected given that LIME only guarantees local fidelity, but it means the two methods cannot be used interchangeably for individual-level recourse decisions.

Counterfactual Analysis

DiCE counterfactuals did not require changes to immutable features like race or sex to flip predictions, which is a positive finding. However, the most common recourse path for high-risk defendants was a large increase in age, sometimes to 69 or 70. This is not actionable. A defendant cannot change their age, making these counterfactuals technically valid but practically useless for recourse purposes.

Recommendations

First, constrain future counterfactual generation to genuinely actionable features only, excluding age and decile score. Second, the persistent race signal in SHAP values warrants a full proxy discrimination audit examining whether features like priors_count are encoding racial disparities from systemic over-policing. Third, explainability outputs should be reviewed on a rolling basis rather than treated as a one-time compliance exercise.
 
---
 
## Notes on Method Limitations
 
- SHAP values reflect marginal contributions relative to the average prediction, not causal effects. A zero SHAP value for race does not mean the model is race-neutral.
- LIME only guarantees local fidelity. Results should not be used to describe global model behavior.
- DiCE counterfactuals can be mathematically valid but practically useless if they require changes to features a defendant cannot control.
- All three methods are diagnostic tools. They must be embedded in a documented audit process to constitute meaningful governance.
