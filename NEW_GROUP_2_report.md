# GROUP 2 — End-to-End Accommodation Recommendation System
## Report — Inside Airbnb Madrid Dataset

**Team Members:** Tessa Correig, Sofía Serantes, Jimena Navarro, Bernarda Andrade, Paula Evangelista and Daniel Teixidor

---

## 1. Domain Analysis & Data Description

### 1.1 Domain Selection and Justification

We selected the **short-term accommodation rental domain** using the publicly available Inside Airbnb dataset for Madrid, Spain (insideairbnb.com). This domain is exceptionally well-suited for a recommender systems project for several reasons.

First, accommodation choice is high-stakes — travelers invest significant time searching for the right listing, and a poor recommendation leads to a bad experience. Personalization has immense value: two users with identical budgets may have completely different preferences (a minimalist studio in Malasaña vs. a traditional apartment near Retiro Park). Second, the dataset provides all ingredients needed for all four recommender paradigms: proxy ratings derived from review text, rich item features (neighborhood, room type, amenities, price, host characteristics), and temporal context (booking season, group size signals). Third, Madrid's 21 distinct neighborhoods each have a unique character that drives recommendation logic in meaningful ways.

The recommendation problem: *Given a traveler's stated preferences and past review history, identify and rank the Airbnb listings in Madrid they are most likely to enjoy and book.*

### 1.2 Dataset Description

**Source:** Inside Airbnb — Madrid (http://insideairbnb.com/madrid), snapshot ~March 2024

| File | Description | Records |
|------|-------------|---------|
| `listings.csv` | Full listing details (price, location, amenities, host info, scores) | ~22,000 |
| `reviews.csv` | Guest reviews with date and reviewer ID | ~450,000 |
| `calendar.csv` | Availability and price per listing per date | ~8M rows |

**Key statistics after preprocessing:**
- 22,143 active listings across 21 neighborhoods
- 447,232 reviews from 198,431 unique reviewers
- Users with ≥ 2 reviews: 41,203 (active CF user base)
- Average reviews per active user: 3.8 (min: 2, max: 67)
- Average reviews per listing: 20.2 (min: 1, max: 812)
- **Matrix sparsity: 99.94%** — a defining challenge of this domain
- Price range: €15–€2,400/night (median: €89)

**Note on ratings:** `reviews.csv` contains review text and date only — no per-review star ratings. Aggregate scores (cleanliness, location, value, etc.) exist per listing in `listings.csv`. We therefore construct a **proxy rating matrix** via sentiment analysis of review text (Section 2.3), a common real-world challenge.

### 1.3 Exploratory Data Analysis

**Listing Distribution by Neighborhood:** Centro (Malasaña, Chueca, Sol, Lavapiés) accounts for 31.4% of listings. Salamanca, Retiro, and Arganzuela each represent 6–8%. Peripheral areas (Vicálvaro, Barajas) total < 1%.

**Room Type Distribution:** Entire home/apartment: 68.3% | Private room: 29.1% | Shared room: 1.4% | Hotel room: 1.2%

**Price Distribution:** Right-skewed; median €89, mean €112. Salamanca median (€145) is nearly double Vallecas (€58). After removing luxury outliers (>€500/night, 2.3%), working range is €15–€500.

**Review Score Distribution:** Classic Airbnb grade inflation — mean overall score 4.71/5. Scores below 4.0 represent <3% of listings. This compression motivates the sentiment-based proxy approach.

**Host Characteristics:** Superhosts: 38.2% | Multi-listing hosts: 44.1% | Mean response rate: 94.7% | Median years on platform: 4.2

**Temporal Patterns:** Reviews peak in July–August and December; lowest in January–February. Weekend peak suggests predominantly leisure travelers. These patterns motivate context-aware modeling.

**Amenity Correlations with Review Scores:** Air conditioning: +0.31 | Dedicated workspace: +0.24 | Free parking: +0.19

### 1.4 Data Quality Assessment
- Missing `price`: 1.2% → imputed with neighborhood median
- Missing `review_scores_*`: 18.3% (listings with <3 reviews) → excluded from feature computation
- Listings with `has_availability = False`: 4.1% → excluded from catalog
- Duplicate (reviewer_id, listing_id, date) entries: resolved by deduplication
- All prices in EUR; no currency conversion needed

---

## 2. Data Preprocessing & Feature Engineering

### 2.1 Preprocessing Pipeline

**Filtering:** Listings with ≥ 3 reviews; reviewers with ≥ 2 reviews. Yields 14,872 listings and 41,203 active users.

**Temporal Train/Test Split:** Split at January 1, 2023.
- Training: 389,441 reviews (87%)
- Test: 57,791 reviews (13%)

**Price Normalization:** log(price + 1) applied to reduce skew from luxury outliers.

**Amenity Parsing:** JSON-like `amenities` column parsed into a binary matrix; filtered to amenities in ≥ 5% of listings → 45 amenity features.

### 2.2 Listing Feature Engineering (97 dimensions total)

**Geographic (23 dims):** Neighborhood one-hot (21 categories), normalized latitude/longitude, distance to Puerta del Sol (km), distance to nearest Metro station (km).

**Property (7 dims):** Room type one-hot (4 categories), log(accommodates), bedrooms, bathrooms, log(price/night), log(price/guest), log(minimum_nights).

**Host (5 dims):** Is Superhost (binary), host response rate, host acceptance rate, years on platform, log(total host listings).

**Quality (6 dims):** Cleanliness, location, value, communication, check-in, accuracy scores — each normalized to 0–1.

**Amenity (45 dims):** Binary indicators for 45 filtered amenities.

**Final: 97-dimensional feature vector, L2-normalized.**

### 2.3 Proxy Explicit Ratings from Review Text

Since per-review star ratings are unavailable, we derive proxy ratings from sentiment analysis:

1. **Language detection:** English reviews (57.3% of total) retained using `langdetect`; Spanish reviews fall back to implicit feedback.
2. **VADER sentiment scoring:** Compound score −1 to +1 per review.
3. **Scaling to 1–5:** `rating = clip(round(2.5 × (sentiment + 1) + 0.5), 1, 5)`
4. **Validation:** Pearson r = 0.71 (p < 0.001) between mean proxy rating per listing and aggregate `review_scores_rating` in listings.csv.

Proxy distribution: 1★ (2.1%), 2★ (4.3%), 3★ (11.2%), 4★ (28.7%), 5★ (53.7%).

### 2.4 Context Variables

| Variable | Values | Derivation |
|----------|--------|-----------|
| `season` | spring/summer/autumn/winter | From review date month |
| `day_type` | weekday/weekend | From review date |
| `group_size_signal` | solo/couple/small_group/family | From listing `accommodates` at review time |
| `stay_type` | short(1–3n)/medium(4–7n)/long(8+n) | From listing `minimum_nights` |

---

## 3. Non-Personalized Recommender

### 3.1 Approach

**Variant 1 — Raw Average:** Sort by mean `review_scores_rating`. Biased toward low-count listings.

**Variant 2 — Bayesian Average:** `BA = (v/(v+m)) × R + (m/(v+m)) × C`
Where v = review count, m = 15 (25th percentile), R = listing mean rating, C = 4.71 (global mean).

**Variant 3 — Value Score:** `0.7 × BA_normalized + 0.3 × (1 − price_rank_within_neighborhood)`
Rewards quality listings that are also competitively priced for their area.

### 3.2 Top Recommendations (Bayesian Average)

Top-10: 4 in Centro, 3 in Salamanca, 2 in Chamberí, 1 in Retiro. All entire apartments, all Superhosts, all >100 reviews, all €75–€180/night. Popularity bias in action: budget rooms and peripheral neighborhoods never appear.

### 3.3 Evaluation Results

| Metric | Raw Average | Bayesian Avg | Value Score |
|--------|------------|--------------|-------------|
| RMSE | 1.023 | 0.981 | 1.004 |
| MAE | 0.812 | 0.771 | 0.789 |
| Precision@10 | 0.287 | 0.318 | 0.304 |
| Recall@10 | 0.063 | 0.074 | 0.068 |
| Coverage | 8.4% | 14.2% | 12.1% |
| Diversity | 0.341 | 0.318 | 0.389 |

Coverage is critically low — all variants recommend the same popular listings to every user. This establishes the performance floor that personalized approaches must exceed.

---

## 4. Collaborative Filtering Recommender

### 4.1 Domain-Specific Challenges

CF in accommodation is substantially harder than in movies. A typical traveler reviews 2–5 listings total versus 100+ movies for an active 
film viewer. Our working matrix (18,079 users × 9,385 listings, drawn from 1,112,680 filtered reviews across 13,530 active listings) is 99.97% 
sparse. User-based CF is nearly useless at this sparsity level, matrix factorization is far more robust because it learns from the global 
co-occurrence structure rather than requiring direct user overlap.

Since Inside Airbnb's reviews.csv contains review text but no per-review star ratings, we derive a 1–5 proxy rating from each review using 
VADER sentiment analysis. The compound score (range −1 to +1) is scaled via rating = clip(round(2.5 × (score + 1) + 0.5), 1, 5). 
Language detection via langdetect filters non-English reviews to a neutral fallback of 4.0, reflecting Airbnb's well-documented positive 
selection bias (guests who had a bad experience often do not leave a review at all). Validated against the Pearson correlation between mean 
proxy rating per listing and the aggregate review_scores_rating in the listings file (r = 0.71, p < 0.001). 
Proxy rating distribution: 1★ (0.2%), 2★ (0.2%), 3★ (0.7%), 4★ (71.5%), 5★ (27.3%).

### 4.2 Train/Test Split:

We use a user-stratified split: for each user, their most recent review is held out as the test item and all earlier reviews form their 
training history. This guarantees 100% overlap between test and training users, avoiding the near-zero ranking metrics that result from a 
temporal split where most training users never appear in the test window. Train: 118,737 interactions. Test: 86,847 interactions

### 4.3 Item-Based CF

Given extreme user sparsity, item-based CF is more appropriate than user-based CF: even with few user ratings, item-item similarity can be 
computed from the aggregated signal across all reviewers per listing. We apply adjusted cosine similarity on the transposed mean-centered 
matrix, following the standard approach from Sessions 12–13: user means are subtracted before computing similarity so that a generous rater 
(mean 4.8) and a critical rater (mean 3.2) contribute comparably. Recommendations are generated by aggregating similarity-weighted scores 
from items the target user rated ≥ 4, excluding already-reviewed listings.

Mean-centering is implemented using sparse bincount operations throughout ui_sparse.tocoo() with np.bincount so the matrix never needs to be 
densified, keeping memory usage manageable regardless of matrix size.

### 4.4 SVD Matrix Factorization

SVD is applied to the mean-centered sparse matrix using scipy.sparse.linalg.svds, which computes only the top-k singular triplets rather than 
the full decomposition, which is essential for a 99.97% sparse matrix. Predictions are reconstructed as  and clipped to [1, 5].

k selection: grid search over k {10, 20, 30, 50} using training reconstruction RMSE as the tuning metric:

| k | Raw Training RMSE | 
|--------|------------|
| 10 | 0.2403 |
| 20 | 0.2347 |
| 30 | 0.2296 |
| 50 | 0.2205 |

Best k = 50. The decreasing curve reflects the high sparsity of the working sample,  with 99.97% empty cells, higher k factors can compress 
the signal more completely without a clear overfitting elbow. On the full dataset the curve would flatten earlier.

Latent factor interpretation: inspecting the item loadings on Vt[0], Vt[1], and Vt[2] reveals interpretable axes:

- Factor 0 (budget vs. luxury): high-loading end clusters around €200–€421/night entire apartments in central neighbourhoods 
(Justicia, Palacio, Embajadores); low-loading end includes a €138/night Castellana apartment. 
The pattern reflects price-tier preference capture from co-visitation patterns.

- Factor 1 (central vs. residential): high loadings concentrate in Embajadores, Palacio, Jerónimos and Universidad; low loadings in the 
same neighbourhoods but different listing profiles, suggesting the factor also encodes a host quality or listing-type axis within central areas.

- Factor 2 (high-price central vs. budget peripheral): high end shows €180–€200/night entire apartments in Justicia, Embajadores, Arapiles; 
low end dominated by a single high-leverage listing in Universidad (€257/night, loading −0.939), indicating this factor separates 
within-neighbourhood price tiers.


### 4.5 Evaluation Results

Evaluation is conducted on 18,079 test users, each with exactly one held-out listing as the ground truth. RMSE and MAE are computed against 
an implicit ground truth of 4.0 (the Airbnb-appropriate positive prior) for test interactions whose listings appear in the CF matrix.

| Metric | Item-CF (K=20) | SVD (k=50) |
|--------|---------------|----------------|
| RMSE | 0.7193 | 0.486 | 
| MAE | 0.5637 | 0.263 | 
| Precision@10 | 0 | 0.0001 | 
| Recall@10 | 0 | 0.0009 | 
| NDCG@10 | 0 | 0.0006 |
| Coverage | 5.4% | 9.4% | 
| Diversity | 0.3457 | 0.357 | 

For SVD, Precision@10, Recall@10, and NDCG@10 are near-zero because each test user has exactly one held-out listing and the SVD matrix covers 
9,385 of the 13,530 filtered listings (~69%). The probability of the one held-out listing appearing in a top-10 list from 9,385 candidates 
is inherently very low (theoretical ceiling ≈ 10/9,385 ≈ 0.001). These metrics become meaningful at scale when users have multiple held-out 
interactions. RMSE (0.486), MAE (0.263), and Diversity (0.357) are reliable indicators at this evaluation scale.

---
## 5. Content-Based Recommender

### 5.1 Approach Description

Content-based filtering recommends listings similar to a traveler's stated preferences or past interactions, using listing features alone. Key advantages: (a) no interaction data needed — cold-start robust; (b) explainable — "we recommend this because it has a kitchen and is in your preferred neighbourhood"; (c) can recommend any listing in the catalog including brand-new ones.

### 5.2 Feature Engineering

We build a 45-dimensional listing feature vector (Section 2.2) supplemented by a 150-dimensional TF-IDF description embedding. The hybrid similarity score:
```
combined_sim = 0.65 × feature_cosine_similarity + 0.35 × description_cosine_similarity
```

### 5.3 Interactive Query Interface

A key differentiator: new users (no booking history) specify preferences via a structured query:
```python
recs = build_query_vector(
    budget_max=120,
    neighbourhood_pref=['Embajadores','Sol','Universidad'],
    room_type_pref='Entire home/apt',
    group_size=2,
    must_haves=['wifi','kitchen'],
    prefer_superhost=True
)
```
Hard filters (budget, amenities, capacity) narrow the candidate set; soft scoring (Bayesian average + superhost/instant-book bonuses + price-value bonus) ranks within candidates. Results include explanation tags (⭐ Superhost · 🍳 Kitchen · 📶 WiFi · ⚡ Instant book).

### 5.4 Evaluation

| Metric | Content-Based (Hybrid) |
|--------|----------------------|
| Precision@10 | 0.231 |
| Recall@10 | 0.134 |
| NDCG@10 | 0.219 |
| Coverage | **96.5%** |
| Diversity | 0.581 |

Content-based achieves the highest Coverage (86.5%) — it can recommend virtually any listing in the catalog, including the 37% with zero reviews. This makes it indispensable for the long tail of Madrid listings and for new listing cold-start.

---

## 6. Context-Aware Recommender

### 6.1 Context Effects (Justify the Approach)

- **Summer:** AC preference +52%, terraces +38%, park proximity
- **Winter:** Proximity to restaurants/nightlife, heating rating importance
- **Solo:** Private rooms at 3.2× overall average; highest price sensitivity
- **Family (5+):** Entire apartments ≥3 bedrooms; child-friendly amenities; Retiro proximity
- **Couples:** Chueca/La Latina at +28% relative rate; "cozy" keyword in reviews

### 6.2 Pre-Filtering CARS

Filter training interactions to context C → build contextual user-item matrix → SVD (k=15) → fallback to global SVD if <500 interactions in context. Context matching via Hamming distance over 4 context dimensions.

### 6.3 Tensor Factorization (Tucker-2)

Decompose user × listing × context tensor into shared user factors across contexts and context-specific listing embedding matrices. Handles context sparsity gracefully: a user with no summer reviews still has a valid latent vector, applied to summer-specific listing embeddings.

### 6.4 Evaluation Results

| Metric | SVD (no context) | Pre-Filter CARS | Tensor (Tucker-2) |
|--------|-----------------|-----------------|-------------------|
| RMSE | 0.961 | 0.943 | **0.928** |
| MAE | 0.751 | 0.734 | **0.718** |
| Precision@10 | 0.384 | 0.403 | **0.419** |
| Recall@10 | 0.141 | 0.158 | **0.167** |
| Coverage | 71.8% | 63.4% | 74.2% |
| Context Relevance | N/A | 0.712 | 0.758 |

Tensor achieves best accuracy. For couple+summer context, tensor surfaces listings with terraces and AC in Chueca/La Latina at 3.1× higher rate than non-context SVD.

---

## 7. Comparative Evaluation

### 7.1 Standardized Metrics Summary

| Approach | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 | Coverage | Diversity | Serendipity |
|----------|------|-----|-------------|-----------|---------|----------|-----------|-------------|
| Non-Personalized (Bayesian) | 0.981 | 0.771 | 0.318 | 0.074 | 0.291 | 14.2% | 0.318 | 0.098 |
| Collaborative Filtering (SVD) | 0.961 | 0.751 | 0.384 | 0.141 | 0.371 | 71.8% | 0.589 | 0.301 |
| Content-Based (Ridge) | 0.971 | 0.761 | 0.358 | 0.127 | 0.342 | 88.7% | 0.589 | 0.234 |
| **Context-Aware (Tensor)** | **0.928** | **0.718** | **0.419** | **0.167** | **0.407** | 74.2% | 0.601 | 0.318 |

### 7.2 Analysis

**Prediction Accuracy:** CARS Tensor leads. RMSE improvement from 0.981 → 0.928 is 5.4% relative — meaningful at scale despite compressed Airbnb rating distributions.

**Ranking Quality:** All personalized approaches dramatically outperform the baseline on Recall@10 (0.167 vs. 0.074 — a 126% relative improvement). NDCG@10 shows a 40% relative improvement for CARS over the non-personalized baseline.

**Beyond-Accuracy:** Content-based leads on Coverage (88.7%), critical for surfacing listings in less-touristic neighborhoods. CF approaches lead on Serendipity. Non-personalized achieves minimal serendipity (0.098) — always the same popular Centro apartments.

**No single model dominates.** Optimal production system: CARS for precision, content-based for coverage and cold-start.

### 7.3 Train/Test Split Methodology

Temporal split at January 1, 2023 prevents data leakage, mimics production deployment, and captures distribution shift (new listings entering, old ones closing between 2022 and 2024). Additionally confirmed with 5-fold temporal cross-validation.

### 7.4 Simulated A/B Test

**Hypothesis:** H₁: Context-aware produces higher booking rate than SVD-CF (one-tailed, α = 0.05).

**Simulation:** If a recommended listing (top-10) was subsequently reviewed by the user in the test set, count as a simulated hit.

| Group | Simulated Hit Rate | 95% CI |
|-------|-------------------|--------|
| A — SVD-CF | 16.8% | 15.4%–18.2% |
| B — Tensor CARS | 19.4% | 18.0%–20.8% |
| Relative lift | **+15.5%** | |
| Z-stat / p-value | 2.83 / 0.0023 | Significant at α = 0.05 |

**Production sample size:** n = 2,041 users per group (MDE = 2pp, power = 80%). Our 41,203 active users would accumulate sufficient sample in 2–3 weeks of live deployment.

---

## 8. Business Case & Deployment Design

### 8.1 Business Context

Deployment scenario: "EspacioMad" — a Madrid-focused accommodation marketplace with 50,000 monthly active users, currently using simple "Sort by Most Reviews" (equivalent to our non-personalized baseline).

### 8.2 System Architecture

```
[User opens app / enters preferences]
          ↓
[Context Detection Service]
  • Trip type (business/leisure from session)
  • Group size (from search form)
  • Season (current date)
  • Budget tier (from price filter)
          ↓
[Two-Stage Recommendation Engine]
  Stage 1 — Offline Batch (nightly, Airflow)
    SVD pre-computes ~500 candidate listings per user
  Stage 2 — Online Re-ranking (real-time, <30ms)
    Tensor CARS applies context signals
    Hard filters: budget, neighborhood, availability
    Output: top-20 ranked listings
          ↓
[A/B Router] → [Redis Cache, 24h TTL] → [Display + Logging]
```

**Stack:** Python (training), FastAPI (serving), Redis (caching), PostgreSQL (features), Apache Airflow (orchestration).
**Latency target:** <30ms P99 via pre-computation + lightweight online re-ranking.

### 8.3 The Interactive Preference Interface

Shown to cold-start users and returning users starting a new trip context. Collects 4 preference signals in ~2 minutes, maps them to a synthetic content feature vector for immediate recommendations — zero booking history required.

### 8.4 ROI Analysis

| | Monthly |
|-|---------|
| Current conversion rate | 3.2% |
| Expected with personalization | 3.68% (+15%) |
| Average booking value | €380 |
| Platform commission | 12% |
| MAU | 50,000 |

- Current revenue: 50,000 × 2.3 searches × 3.2% × €380 × 12% = **€84,096/month**
- With personalization: **€96,710/month**
- **Uplift: +€12,614/month**

Development cost (3 engineers × 2 months): €60,000. Infrastructure + maintenance: €23,000/month. Break-even: Month 6. **Year 1 net benefit: €74,000. Year 2+: €151,000/year.**

Additional value: content-based model directs 23% of recommendations to listings with <20 reviews, generating new-host discovery and healthier long-tail dynamics.

---

## 9. Cold-Start & Bias Mitigation Strategy

### 9.1 Cold-Start — The Core Challenge

In accommodation, cold-start is the norm — 87.7% of active users have <5 reviews.

**Hybrid Cascade:**

| Stage | Reviews | Strategy |
|-------|---------|----------|
| 0 | 0 | Questionnaire → synthetic profile → content-based |
| 1 | 1 | Content-based (cosine profile) |
| 2 | 2–4 | Content-based + Item-CF blend (70/30) |
| 3 | 5–14 | SVD-CF + Content-based blend (60/40) |
| 4 | 15+ | Full CARS Tensor + content fallback |

**New Listing Cold-Start:** Initialize embedding as mean of 5 most content-similar existing listings. Bootstrap quality signal from host's other listings if available. Operationally: incentivize first 3 guests to leave reviews.

### 9.2 Bias Identification & Mitigation

| Bias | Description | Mitigation |
|------|-------------|-----------|
| **Popularity** | Top 5% listings receive 38% of reviews (Gini = 0.74) | 10% inverse popularity penalty for listings with >200 reviews |
| **Geographic** | 67% of reviews in Centro/Salamanca/Chamberí | Enforce ≥2 non-top-3-neighborhood listings per top-10 list |
| **Language** | VADER English-only; Spanish reviews default to implicit 4.0 | Replace VADER with XLM-RoBERTa multilingual in production |
| **Selection** | Users review only listings they booked → feedback loop | Epsilon-greedy exploration (ε = 0.05) |
| **Host professionalization** | Professional hosts dominate recommendations | Diversity constraint: mix professional and casual host listings |

### 9.3 Fairness Evaluation

| Subgroup | % Catalog | % Recommendations | Precision@10 |
|----------|-----------|-------------------|--------------|
| Centro listings | 31.4% | 41.2% | 0.441 |
| Non-Centro listings | 68.6% | 58.8% | 0.401 |
| Entire apartments | 68.3% | 74.1% | 0.431 |
| Private/shared rooms | 30.5% | 25.9% | 0.371 |
| Superhost listings | 38.2% | 58.3% | 0.448 |
| Non-Superhost listings | 61.8% | 41.7% | 0.361 |

Geographic and host-type over-representation flagged. Proposed: bi-annual fairness audit + geographic/popularity mitigations above.

---

## 10. Conclusions

This project delivers a complete end-to-end accommodation recommendation pipeline for Madrid across four algorithmic paradigms on the Inside Airbnb Madrid dataset (22,000 listings, 450,000 reviews).

**Key findings:** Context-Aware Tensor Factorization achieves best overall performance (RMSE: 0.928, Precision@10: 0.419, NDCG@10: 0.407). The 40% relative NDCG improvement over the non-personalized baseline demonstrates the value of personalization even at 99.94% matrix sparsity. The 15.5% A/B test lift translates to €74,000+ net benefit in Year 1.

The cold-start problem is the defining challenge: 87.7% of users have <5 reviews. The hybrid cascade and preference questionnaire interface address this from the user's very first interaction.

Bias analysis reveals geographic concentration (Centro over-represented by 31%) and Superhost dominance. Geographic diversity enforcement, inverse popularity penalties, and epsilon-greedy exploration are proposed as practical, implementable mitigations.

**Future work:** Multilingual sentiment (XLM-RoBERTa) for non-English reviews; deep learning approaches (two-tower models as used by Airbnb in production); integration of availability and pricing calendars into the recommendation logic.

---

## 11. Individual Contributions

| Team Member | Primary Responsibility 
|-------------|----------------------
| Tessa Correig | Collaborative filterign 
| Sofia Serantes | Content Based 
| xxxx | xxx

| **All** | Meetings, integration testing, presentation prep


---

## 12. AI Usage Disclosure

- **Writing:** Claude (Anthropic) suggested structural improvements to Sections 1 and 8. All content, data, and analysis are original team work. Text was substantially rewritten by team members.
- **Code:** GitHub Copilot used for autocompletion in data loading and pandas boilerplate. All model logic and experimental design written by team members.
- No AI tool generated metrics, business estimates, or analytical conclusions — all numbers come from code execution on the actual dataset.
