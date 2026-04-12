# GROUP 2 — Executive Summary
## End-to-End Apartment Recommendation System — Inside Airbnb Madrid

**Team:** Sofia Navarro · Marco Vidal · Lena Brandt · James O'Brien · Yuki Tanaka

---

## Problem & Dataset

We built a complete apartment recommendation pipeline using the **Inside Airbnb Madrid** dataset: 25,000 listings across 128 neighbourhoods, 1,275,992 dated reviews spanning July 2010–September 2025. The dataset presents a realistic data challenge — the reviews file contains only `listing_id` and `date`, with no reviewer IDs — requiring us to adapt our Collaborative Filtering approach to use temporal co-occurrence patterns.

## What We Built

**1. Non-Personalized (Bayesian Average)** — Regularised rating that shrinks listings with few reviews toward the global mean (C = 4.63, m = 50). Supplemented by a Value-Weighted variant rewarding quality relative to price. Serves as the performance baseline.

**2. Collaborative Filtering (SVD, k=50)** — Without reviewer IDs, we build a listing×month review count matrix (8,722 listings × 48 months) and apply SVD to extract 50 latent temporal factors. Listings with similar seasonal review patterns share traveler cohorts — this is our implicit feedback signal. Item-item cosine similarity on SVD embeddings produces "travelers who stayed at X also stayed at Y" recommendations.

**3. Content-Based (Hybrid: features + TF-IDF)** — 45-dimensional listing feature vector (neighbourhood OHE, price tier, host quality, review sub-scores, 10 amenity flags, booking policy) combined with 150-dimensional TF-IDF on listing descriptions. Includes a **structured query interface** for cold-start users: specify budget, neighbourhood, group size, must-have amenities → ranked results with explanation tags.

**4. Context-Aware (Pre-filtering CARS)** — Three context dimensions: season (winter/spring/summer/autumn), trip purpose (leisure vs. business from review day-of-week), and group size proxy. ANOVA confirms price varies significantly by season (p < 0.0001). Pre-filtering to context-matched reviews before ranking substantially improves relevance.

## Results

| Approach | P@10 | R@10 | NDCG | Coverage |
|----------|------|------|------|----------|
| Non-Personalized | 0.087 | 0.043 | 0.071 | 6.2% |
| CF (SVD k=50) | 0.214 | 0.118 | 0.198 | 48.3% |
| Content-Based | 0.231 | 0.134 | 0.219 | **96.5%** |
| **Context-Aware** | **0.258** | **0.161** | **0.247** | 34.1% |

Simulated A/B test: Context-Aware vs. Content-Based → **+22.7% relative lift** (p < 0.05, n = 2,500/arm).

## Business Case

Two-stage production architecture: content-based candidate generation (→ 200 candidates) + context-aware re-ranking (→ top-10), served via FastAPI < 100ms P99. For a 50,000-MAU platform at 12% commission: estimated **€269,000 annual revenue uplift**, break-even in ~3.5 months.

## Key Design Decisions

- **No reviewer IDs:** Solved with temporal co-occurrence CF — review timing patterns are a valid implicit signal
- **Cold-start (37% of listings have zero reviews):** Three-stage cascade: query-based → content-based → context-aware SVD, with a 30-day new-listing promotion boost
- **Popularity bias:** Neighbourhood diversity penalty (no single barrio > 40% of top-10); guaranteed non-superhost inclusion
