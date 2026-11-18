# Predicting New User Uninstallation in Digital Wallet Apps with Explainable AI

Code for my undergraduate thesis and the IEEE paper:

> F. Lee and N. Zulkarnain, "Predicting New User Uninstallation in Digital Wallet Applications with Explainable Artificial Intelligence," *2026 International Conference on Information Management and Technology (ICIMTech)*, Jakarta, Indonesia, 2026.
> DOI: [10.1109/ICIMTech202671078.2026.11698998](https://doi.org/10.1109/ICIMTech202671078.2026.11698998) · [IEEE Xplore](https://ieeexplore.ieee.org/document/11698998)

## Problem
New users of a digital wallet app often uninstall it within the first 30 days, many of them before making a single transaction. Most churn research focuses on dormant existing users. This project predicts **early-stage uninstallation** and explains *why* the model flags a user, so CRM teams can act on it.

## Approach
| Step | What was done |
|---|---|
| Data | 132,374 new users with app interaction, transaction and acquisition-source attributes (original data is confidential; this repo ships a synthetic version, see below) |
| Preprocessing | Missing-value handling, amount cleaning, one-hot encoding, `Register_Weekday` from sign-up date |
| Feature engineering | `Engagement_Score` (PN click + card click + cash-in + payment), `Total_Transaction_Amount` (top-up + product amount) |
| Class imbalance | `scale_pos_weight` ≈ 4.42 (class weighting instead of SMOTE, so SHAP explains the real distribution) |
| Model | XGBoost, 80:20 stratified split |
| Tuning | Optuna (Bayesian optimization), 50 trials, 5-fold stratified CV |
| Explainability | SHAP: global (summary, bar, dependence) and local (waterfall, force) |

## Results (as reported in the paper)
| Metric | Baseline | Tuned |
|---|---|---|
| Accuracy | 0.8192 | **0.8244** |
| Precision | 0.5077 | **0.5193** |
| Recall | 0.6501 | 0.6391 |
| F1-score | 0.5701 | **0.5730** |
| ROC-AUC | 0.7972 | 0.7972 |

**Key insight:** acquisition channel is by far the strongest driver. Users from **paid campaigns who make zero transactions in their first days** are the highest-risk group, while transaction amount and engagement push users toward staying.

**Business recommendations:** dedicated onboarding for paid-channel users, first-transaction incentives and push notifications for low-engagement users, and onboarding timed by registration weekday.

## ⚠️ About the data
The original dataset belongs to the company where the research was done and is **not included**.
`synthetic_new_users.csv` is a **synthetic dataset with the same schema**: rows were resampled to a different size and class ratio, sign-up dates moved to a fictitious month (only the weekday is kept), amounts perturbed with random noise, partner/biller names replaced by generic categories, and some flags randomly flipped.
It is meant to make the pipeline runnable. **Metrics produced by the notebook differ from the paper** and do not reflect any real business figures.

## Repository contents
| File | Description |
|---|---|
| `churn_prediction_xgboost_shap.ipynb` | Full pipeline: cleaning → feature engineering → baseline → Optuna tuning → evaluation → SHAP (executed on synthetic data) |
| `synthetic_new_users.csv` | Synthetic dataset (50,000 rows, `;`-separated) |
| `requirements.txt` | Python dependencies |

## How to run
```bash
pip install -r requirements.txt
jupyter notebook churn_prediction_xgboost_shap.ipynb
```
Or open it in Google Colab and upload `synthetic_new_users.csv` next to the notebook.

## Related
- Live demo (Streamlit): https://churn-prediction-xai-digital-wallet.streamlit.app/ · app source: [fintech-churn-risk-app](https://github.com/fernandoolee/fintech-churn-risk-app)
- Author: **Fernando Lee**, Information Systems, BINUS University · [GitHub](https://github.com/fernandoolee)
