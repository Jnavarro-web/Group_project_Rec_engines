# GROUP 2 — Apartment Recommendation System
### Inside Airbnb Madrid · Recommender Systems Course

## Team
Sofia Navarro · Marco Vidal · Lena Brandt · James O'Brien · Yuki Tanaka

---

## Dataset Setup (Do This First)

The notebooks use the real Inside Airbnb Madrid dataset. **You must download the files before running any notebook.**

**Download from:** http://insideairbnb.com/get-the-data.html  
→ Scroll to **Madrid, Community of Madrid, Spain**  
→ Download: **`listings.csv.gz`** (the detailed version, ~20MB) and **`reviews.csv`** (~50MB)

Place both files in the `data/` folder:
```
GROUP_2_project/
└── data/
    ├── listings.csv.gz     ← download this
    └── reviews.csv         ← download this
```

> **Note:** `reviews.csv` contains only `listing_id` and `date` — no reviewer IDs.  
> This is the standard Inside Airbnb format. Our CF notebook (02) explains how we handle this.

---

## Project Overview

End-to-end recommendation system comparing four approaches on 25,000 Madrid Airbnb listings and 1,275,992 dated reviews.

| Approach | Method | Key strength |
|----------|--------|-------------|
| Non-Personalized | Bayesian Average | Interpretable baseline |
| Collaborative Filtering | SVD on temporal co-occurrence matrix | Discovers cross-neighbourhood similar listings |
| Content-Based | 45-dim features + TF-IDF | Works for any listing, cold-start robust |
| Context-Aware | Pre-filtering CARS (season × trip purpose) | Best ranking quality |

**Best result:** Context-Aware achieves Precision@10 = 0.258 — 3× the non-personalized baseline.

---

## Repository Structure

```
GROUP_2_project/
├── data/
│   ├── listings.csv.gz        ← download from Inside Airbnb
│   └── reviews.csv            ← download from Inside Airbnb
├── figures/                   ← auto-generated when notebooks run
├── 01_non_personalized.ipynb  ← EDA (8 charts) + Bayesian recommender
├── 02_collaborative_filtering.ipynb  ← temporal SVD item-item CF
├── 03_content_based.ipynb     ← feature engineering + query interface
├── 04_context_aware.ipynb     ← seasonal CARS + context analysis
├── 05_evaluation.ipynb        ← comparison table + A/B test + bias
├── GROUP_2_report.md          ← full written report (§1–12)
├── GROUP_2_executive_summary.md
└── README.md
```

---

## Setup

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter
mkdir -p data figures
# → place listings.csv.gz and reviews.csv in data/
```

## Running the Notebooks

Run in order (each notebook is self-contained but 02–05 follow 01's conventions):

```bash
jupyter notebook
```

Or execute non-interactively:
```bash
jupyter nbconvert --to notebook --execute --inplace 01_non_personalized.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_collaborative_filtering.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_content_based.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_context_aware.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_evaluation.ipynb
```

Expected runtime per notebook: 2–5 minutes (depends on machine).  
All figures save automatically to `figures/`.

---

## Key Results

| Metric | Non-Pers. | CF (SVD) | Content | Context-Aware |
|--------|-----------|----------|---------|---------------|
| Precision@10 | 0.087 | 0.214 | 0.231 | **0.258** |
| Recall@10 | 0.043 | 0.118 | 0.134 | **0.161** |
| NDCG@10 | 0.071 | 0.198 | 0.219 | **0.247** |
| Coverage | 6.2% | 48.3% | **87.2%** | 34.1% |
| Diversity | 0.201 | **0.634** | 0.581 | 0.612 |

Simulated A/B test: Context-Aware vs Content-Based → **+22.7% lift (p < 0.05)**.  
Business case: ~€269,000 annual revenue uplift for a 50K-MAU platform.

---

## Notes for Students

- **No reviewer IDs in the dataset** — this is expected. Notebook 02 explains our temporal CF approach.
- The **query interface in Notebook 03** (Demo 1–3) is designed to be shown live during the presentation.
- All figures are saved to `figures/` — use them directly in your slides.
- The **standardised comparison table** is in Notebook 05, Cell "Final Summary Table".
- Results may differ slightly from the report (±0.01) depending on the exact dataset snapshot downloaded.

## AI Usage Disclosure

GitHub Copilot used for pandas/scipy boilerplate. Claude (Anthropic) used for grammar review of report sections. All algorithms, analysis, and business estimates produced by team members.
