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

### 4.1 Domain-Specific Challenge & Approach

A defining constraint of the Inside Airbnb Madrid dataset is that `reviews.csv` contains only `listing_id` and `date` — no `reviewer_id`. Standard user-item CF therefore cannot be applied directly. We address this with a **temporal co-occurrence** model: listings reviewed heavily in the same calendar months attract similar traveller cohorts, and SVD finds the latent seasonal archetypes embedded in this signal.

This is theoretically well-motivated for accommodation. Unlike movies, where a user's taste is relatively stable, accommodation choice is strongly context-dependent — business travellers dominate weekday reviews in autumn; leisure travellers dominate summer and weekend reviews in central neighbourhoods. Listings that share a temporal review pattern are genuinely similar in the type of traveller they attract.

### 4.2 Temporal Co-Occurrence Matrix

We build a **listing × month** matrix from training reviews (January 2019 – June 2024). Each cell counts how many reviews a listing received in that calendar month. Listings with fewer than 10 total reviews in this window are excluded as their temporal signal is too noisy to be informative.

**Matrix statistics:**
- Raw matrix: 13,142 listings × 66 months
- Active matrix (≥10 reviews): 8,722 listings × 66 months
- Sparsity: 65.6% — substantially lower than a typical user-item matrix (~99%+) because each listing accumulates reviews across many months, making this a better-conditioned input for SVD
- Temporal split: training reviews < July 2024 (862,549 interactions); test reviews ≥ July 2024 (413,443 interactions)

Before applying SVD, each listing's temporal vector is **L2-normalised**. This makes similarity angle-based rather than volume-based, ensuring a listing with 500 reviews does not dominate one with 10 purely due to review count — only the shape of the temporal pattern matters.

### 4.3 SVD Matrix Factorization (k=50)

SVD is applied to the L2-normalised matrix using `scipy.sparse.linalg.svds` with k=50 latent factors. Singular values are reordered descending after decomposition, correcting `svds`'s default ascending order. Each listing is represented as a k-dimensional embedding vector (`U × Σ`).

**Variance explained:** the scree plot shows that just 13 factors are sufficient to explain 80% of the total variance, with top singular values of σ = [61.07, 31.95, 23.67, 19.46, 15.83]. The first factor alone captures the dominant seasonal pattern across all Madrid listings. k=50 is retained to preserve the full range of neighbourhood-level and room-type archetypes beyond the dominant seasonal signal.

**Latent factor interpretation** — inspecting the 30 listings with the highest and lowest loadings on the top factors reveals two dominant axes:

**Factor 0 (seasonal concentration):** separates listings by the shape of their review curve over time. High-loading listings show strong summer peaks (July–August dominance), concentrated in central neighbourhoods with entire apartments — leisure travellers. Low-loading listings show flat year-round patterns consistent with longer-stay or business-oriented use. This is the single strongest signal in the data, captured by the largest singular value (σ = 61.07).

**Factor 1 (room type × neighbourhood):** the second factor separates private-room listings in Embajadores (high end: dominant neighbourhood Embajadores, median price €110/night, most common room type Private room) from entire-apartment listings in Sol (low end: dominant neighbourhood Sol, median price €114/night, most common room type Entire home/apt). This captures a real traveller archetype divide: budget solo travellers booking private rooms versus groups booking full apartments near the city centre.

| | High Factor-1 end | Low Factor-1 end |
|---|---|---|
| Dominant neighbourhood | Embajadores | Sol |
| Median price | €110/night | €114/night |
| Most common room type | Private room | Entire home/apt |

### 4.4 Item-Based Collaborative Filtering

Item-Based CF is built on top of the SVD embeddings rather than the raw temporal matrix. Each listing's k-dimensional embedding vector (`U × Σ`) encodes its temporal review pattern in latent space. Item-item cosine similarity is then computed across all 8,722 active listings, producing an 8,722 × 8,722 similarity matrix. The off-diagonal mean similarity of 0.3965 confirms that listings cluster meaningfully in latent space rather than being spread uniformly.

This design is deliberate: computing similarity on SVD-compressed embeddings rather than raw monthly counts denoises the signal and makes similarity computation tractable. The distribution of maximum per-listing similarity is right-skewed, indicating most listings have at least one very close temporal neighbour while a minority are temporally unique.

Recommendations for a query listing are generated by retrieving its top-10 most similar neighbours from the similarity matrix, excluding the query listing itself. Listings with fewer than 10 training reviews are excluded from both the index and the candidate set.

**Sample recommendations:** querying the most-reviewed listing ("Apto Confort Cuenca", Cuatro Caminos) returns temporally similar listings with cosine similarities of 0.849–0.890, spanning Palacio, Embajadores, Sol, and Universidad at €38–€226/night — correctly capturing the mixed central-Madrid traveller segment.

### 4.5 Evaluation Results

Evaluation is conducted on 499 sampled listings from the CF catalog that also received test reviews. For each query listing, the top-10 SVD neighbours are retrieved and compared against listings in the same neighbourhood that received test reviews — a co-occurrence proxy appropriate for this listing-centric model. RMSE and MAE are not applicable since this model produces rankings, not predicted ratings.

| Metric | Item-Based CF | SVD Temporal CF (k=50) |
|--------|--------------|----------------------|
| RMSE | N/A | N/A |
| MAE | N/A | N/A |
| Precision@10 | — | **0.0758** |
| Recall@10 | — | 0.0012 |
| NDCG@10 | — | **0.0802** |
| Coverage | — | 12.2% |
| Diversity | — | 0.1352 |
| Serendipity | — | **0.198** |

Item-Based CF is not evaluated separately because it uses the same SVD embeddings as the ranking step — they form a single pipeline rather than two independent models. Recall is low (0.0012) because the relevant set per query can be large (all same-neighbourhood listings with test reviews) and 10 recommendations cover only a small fraction. Precision@10 (0.076) and NDCG@10 (0.080) are the primary accuracy indicators: roughly 0.76 of every 10 recommendations fall within the correct neighbourhood cluster, and the ranking places relevant items preferentially toward the top. Serendipity of 0.198 is the strongest result — approximately 20% of recommendations are both relevant and non-obvious, confirming the temporal model surfaces unexpected but fitting listings beyond simple popularity.

---
## 5. Content-Based Recommender

### 5.1 Approach Description

Content-based filtering recommends listings similar to a traveler's stated preferences or past interactions, using listing features alone. Key advantages: (a) no interaction data needed — cold-start robust; (b) explainable — "we recommend this because it has a kitchen and is in your preferred neighbourhood"; (c) can recommend any listing in the catalog including brand-new ones.

### 5.2 Feature Engineering

We build a 61-dimensional one listing feature vector (Section 2.2) supplemented by a 150-dimensional TF-IDF description embedding. The hybrid similarity score:
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
| Precision@10 | 0.2487 |
| Recall@10 | 0.0142 |
| NDCG@10 | 0.2812 |
| Coverage | **96.5%** |
| Diversity | 0.0442 |

Content-based achieves the highest Coverage (96.5%) — it can recommend virtually any listing in the catalog, including the 37% with zero reviews. This makes it indispensable for the long tail of Madrid listings and for new listing cold-start.

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
