# GROUP 12 — Executive Summary
## End-to-End Apartment Recommendation System — Inside Airbnb Madrid

**Team:** Jimena Navarro · Sofia Serantes · Bernarda Andrade · Paula Evangelista · Tessa Correig · Daniel Teixidor

---

## Problem & Dataset

We built a complete apartment recommendation pipeline using the **Inside Airbnb Madrid** dataset: 24,958 active listings across 128 neighbourhoods, 1,275,992 dated reviews spanning July 2010–September 2025. The dataset presents a realistic data challenge — the reviews file contains only `listing_id` and `date`, with no reviewer IDs — requiring us to adapt our Collaborative Filtering approach to use temporal co-occurrence patterns rather than explicit user-item interactions.

## What We Built

**1. Non-Personalized (Bayesian Average)** — Regularised rating that shrinks listings with few reviews toward the global mean (C = 4.63, m = 50, selected via sensitivity test across m ∈ {10, 25, 50, 100, 200}). Supplemented by a Value-Weighted variant rewarding quality relative to price. The only model that produces explicit rating predictions; RMSE and MAE are reported here only. Serves as the performance baseline.

**2. Collaborative Filtering (SVD, k=50)** — Without reviewer IDs, we build a listing×month review count matrix (8,722 listings × 66 months, 65.6% sparsity) and apply SVD to extract 50 latent temporal factors (top singular values: [61.07, 31.95, 23.67, ...]). Listings with similar seasonal review patterns share traveler cohort signatures — this is our implicit feedback signal. Item-item cosine similarity on SVD embeddings produces recommendations based on shared seasonal traveler patterns. We carefully characterise what this signal can and cannot prove: it reflects co-seasonal behaviour, not confirmed co-booking by the same individuals.

**3. Content-Based (Hybrid: features + TF-IDF)** — 61-dimensional listing feature vector (neighbourhood OHE, price tier, host quality, review sub-scores, 10 amenity flags, booking policy) combined with 150-dimensional TF-IDF on listing descriptions. The 65/35 feature/text weighting was selected via grid search on a holdout set. Includes a **structured query interface** for cold-start users: specify budget, neighbourhood, group size, must-have amenities → ranked results with explanation tags. Best ranking accuracy of all models.

**4. Context-Aware (Pre-filtering CARS)** — Three context dimensions: season (winter/spring/summer/autumn), trip purpose (leisure vs. business from review day-of-week proxy), and peak flag. ANOVA confirms price varies significantly by season (F = 42.78, p < 0.0001); Chi-squared confirms neighbourhood preference varies by trip purpose (χ² = 880.84, p < 0.0001). Pre-filtering to context-matched reviews before scoring adds seasonal relevance on top of the baseline. Best deployed as a re-ranker on top of Content-Based candidates.

## Results

| Approach | RMSE | MAE | P@10 | R@10 | NDCG | Coverage | Diversity | Serendipity |
|----------|------|-----|------|------|------|----------|-----------|-------------|
| Non-Personalized | 0.1459 | 0.0755 | 0.0574 | 0.0006 | 0.0607 | 0.04% | 0.002 | 0.000 |
| CF (SVD k=50) | — | — | 0.0741 | 0.0009 | 0.0741 | 12.6% | **0.139** | 0.120 |
| **Content-Based** | — | — | **0.4096** | **0.0127** | **0.4429** | **16.98%** | 0.070 | **0.972** |
| Context-Aware | — | — | 0.0680 | 0.0007 | 0.0760 | 0.09% | 0.001 | 0.014 |

All ranking models evaluated on the same held-out test set (reviews from July 2024 onward) using a neighbourhood + test-interaction relevance proxy. RMSE/MAE reported only for Non-Personalized, which is the only model producing explicit rating-scale predictions.

**Simulated A/B test:** Context-Aware vs. Content-Based → lift = **−85%** (p = 1.000, fail to reject H₀). Content-Based is the primary driver; Context-Aware is best deployed as a re-ranker, not a standalone model.

## Business Case

Two-stage production architecture: content-based candidate generation (→ 200 candidates, 16.98% catalog coverage) + context-aware seasonal re-ranking (→ top-10), served via FastAPI < 100ms P99 (benchmarked: p99 = 94ms). For a 50,000-MAU platform at 12% commission, applying a conservative 15% realisation rate to the 7× offline ranking improvement:

- **Conservative scenario:** +€72,720 annual uplift, break-even ~15 months
- **Optimistic scenario:** +€142,560 annual uplift, break-even ~8 months

## Key Design Decisions

- **No reviewer IDs:** Solved with temporal co-occurrence CF — seasonal review timing patterns are a valid item-item similarity signal (characterised accurately, not overstated as user-level CF)
- **Cold-start (20.6% of listings have zero reviews):** Four-stage cascade: query-based content filtering → content+Bayesian hybrid → content+CF hybrid → full context-aware pipeline, scaled by review count (0 / 1–9 / 10–49 / 50+), with a 30-day new-listing promotion boost
- **A/B test honest reporting:** The simulation shows Context-Aware does not outperform Content-Based as a standalone; the cascade architecture reflects this finding
- **Popularity & superhost bias:** Neighbourhood diversity cap (no single barrio > 40% of top-10); guaranteed ≥1 non-superhost per list; COVID-era review down-weighting (0.7×); price-tier-stratified ranking
