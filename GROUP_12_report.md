# GROUP 12 — End-to-End Apartment Recommendation System
## Full Report — Inside Airbnb Madrid Dataset

**Team:** Jimena Navarro - Sofia Serrantes - Berni Andrade - Paula Evangelista - Tessa Correig -  Daniel Teixidor
**Course:** Recommender Systems 
**Submission:** April 19

---

## 1. Domain Analysis & Data Description

### 1.1 Domain Selection and Justification

We selected the **short-term apartment rental domain** using the Inside Airbnb Madrid dataset. This domain is an ideal testbed for recommender systems for several compounding reasons.

Madrid's catalog lists 25,000 active properties across 128 neighbourhoods, with enormous variation in price (€8–€2,000+/night), room type (entire apartments, private rooms, hotel rooms), host profile (superhosts vs. casual hosts), and neighbourhood character, from the tourist-heavy Centro barrios to quieter residential areas like Arapiles or Guindalera. A traveler searching for accommodation faces a genuine information overload problem, exactly the problem recommender systems are designed to solve. The catalog is large enough that exhaustive manual browsing is impractical, yet structured enough that algorithmic signals are meaningful.

The dataset provides distinct signals for all four required paradigms. Listing-level review scores serve as item quality ratings for the non-personalized baseline. Temporal review patterns enable collaborative filtering without explicit user identities. Rich listing features (location, price, host quality, 10+ amenity flags, description text) enable content-based filtering. And review date metadata enables context-aware modeling through season and day-of-week inference. The domain also has clear business stakes — Airbnb's own recommendation engine is estimated to drive 30–40% of bookings — making the business case section credible and grounded in real commercial practice.

**The recommendation problem, formally stated:** Given a traveler's stated preferences or inferred interaction history, rank the set of Madrid apartments so that the most relevant listings appear in the top-K positions, where relevance is defined by a combination of listing quality, content match, and contextual fit.

### 1.2 Dataset Description

**Source:** Inside Airbnb — Madrid  
**URL:** http://insideairbnb.com/get-the-data.html  
**Files used:** `listings.csv.gz` (detailed listings) and `reviews.csv`

| File | Description | Records |
|------|-------------|---------|
| `listings.csv.gz` | Full listing details (79 columns) | 25,000 listings |
| `reviews.csv` | Review dates per listing | 1,275,992 reviews |

**Critical data characteristic:** The Inside Airbnb `reviews.csv` contains only `listing_id` and `date` — no reviewer ID. This is a realistic data limitation present in many production systems where user anonymity is protected or where the data provider strips identifiers before public release. This constraint directly shapes our Collaborative Filtering approach (Section 4) and is a central design challenge throughout the project. We do not treat it as a flaw to be glossed over; we treat it as a constraint to be engineered around systematically.

**Key statistics after preprocessing:**

| Metric | Value |
|--------|-------|
| Active listings (price ≤ €2,000) | 24,958 |
| Total reviews | 1,275,992 |
| Review date range | July 2010 – September 2025 |
| Unique neighbourhoods (barrios) | 128 |
| Median price per night | €110 |
| Mean review score | 4.63 / 5.0 |
| Listings scoring ≥ 4.5 | 83% |
| Superhost listings | 21.6% |
| Entire home/apt | 66.7% |
| Private room | 32.4% |
| Listings with zero reviews | 20.6% |

**Train/test split:** Reviews before July 1, 2024 (862,549 reviews, 67.6%) constitute the training set; reviews from July 2024 onward (413,443 reviews, 32.4%) constitute the evaluation set. We use a **temporal hold-out** rather than a random split for two reasons: first, it mirrors production conditions where a model trained today will be evaluated on future behavior; second, random splitting would leak future review patterns into the training set, artificially inflating evaluation metrics.

### 1.3 Exploratory Data Analysis

**Rating distribution:** Ratings are heavily left-skewed — 83% of listings score ≥ 4.5/5. This ceiling effect compresses quality differences between listings. A listing at 4.6 vs. 4.9 may represent a meaningful real-world quality gap but appears small numerically. Additionally, 20.6% of listings lack any rating at all. These two facts together — compression at the top and large gaps at the bottom — make raw mean ratings a poor signal, motivating our Bayesian regularisation approach (Section 3).

**Price distribution:** Prices are right-skewed (median €110, mean €145, standard deviation €187). Entire homes have a median of €125/night; private rooms €55/night. We cap at €2,000 after inspecting 42 extreme outliers that appear to be commercial properties or data errors. Log-transforming price before feature engineering normalises this distribution for downstream similarity calculations.

**Neighbourhood concentration:** The top 5 neighbourhoods (Embajadores, Universidad, Palacio, Sol, Justicia) account for 37% of all listings. Review activity is even more concentrated — these central barrios receive proportionally more reviews than their catalog share, creating a popularity bias that will systematically disadvantage good listings in quieter residential areas. Listings in the top 5 barrios average 89 reviews, versus 31 for all other barrios. This finding directly motivates our diversity penalty (Section 9).

**Review timeline:** Review volume grew steadily from 2010 to a pre-COVID peak in 2019, collapsed during 2020–2021 lockdowns (dropping to 18% of 2019 volume in 2020), and recovered strongly from 2022 onward, reaching new highs in 2023–2024. COVID-era reviews are down-weighted in our temporal CF model (0.7× weight applied to 2020–2021 reviews) to avoid distributional shift from pandemic-period booking patterns, which do not reflect normal traveler preferences.

**Host analysis:** Superhosts (21.6% of hosts) average 4.834/5 vs. 4.573/5 for regular hosts — a 0.26-point gap. Median price is identical at €110 for both groups — superhosts do not charge a premium, making their higher rating a genuine quality signal rather than a pricing artefact. This gap is at least partly structural: superhosts are more likely to have more reviews (stabilising their average) and attract guests who select specifically for quality (a selection effect). This creates a superhost bias we address in Section 9 with a guaranteed non-superhost inclusion policy.

**Amenity prevalence:** WiFi (92.3%), kitchen (89.3%), and hot water (74.9%) are near-universal and therefore poor discriminators. More selective amenities — gym (8.1%), parking (14.7%), pool (2.3%) — are meaningful differentiators for targeted filtering. This guides our choice of which amenity flags to include in the feature vector.

**Seasonal patterns:** Review volume by month shows a clear bimodal pattern with peaks in July–August and again in October–December. Summer peak volume is 2.3× winter trough. This temporal structure motivates our Context-Aware approach and validates the use of seasonal review patterns as a collaborative filtering signal.

---

## 2. Data Preprocessing & Feature Engineering

### 2.1 Preprocessing Pipeline

**Price cleaning:** Strip `$` and `,` characters from the raw string field; cast to float; cap at €2,000 (removes 42 extreme outliers); impute 312 missing values with neighbourhood median, then global median for the 3 listings in neighbourhoods with no priced comparators.

**Rating imputation:** 20.6% of listings lack `review_scores_rating`. We impute using neighbourhood mean, then global mean (4.63) for listings in neighbourhoods with no rated listings. An indicator flag `rating_imputed` is added to the feature vector so downstream models can discount imputed ratings.

**Boolean cleaning:** Convert `host_is_superhost` and `instant_bookable` from `t/f` string format to integer flags (1/0).

**Host features:** Parse `host_response_rate` and `host_acceptance_rate` from percentage strings to floats in [0,1]. Compute `host_years` as the difference in years between `host_since` and reference date 2025-01-01. Missing response/acceptance rates (14.3% of listings) are imputed with the neighbourhood median.

**Amenity parsing:** The `amenities` field is a JSON-like list stored as a string. We apply regex matching against the lowercased string to create 10 binary amenity flags: WiFi, kitchen, AC, heating, washer, TV, parking, elevator, gym, pets allowed. We validate extraction accuracy by manually checking 200 randomly sampled listings against their raw amenity strings; precision and recall for each flag exceed 97%.

**Temporal context labelling:** Each review receives: season (winter/spring/summer/autumn from month), is_peak (True for July, August, December), day_of_week (0–6), is_weekend, trip_purpose_proxy (business if weekday AND accommodates ≤ 2; leisure otherwise), and group_size_proxy from the accommodates field.

We acknowledge that the trip_purpose_proxy is a heuristic with meaningful noise. A solo traveler visiting Madrid on a Tuesday for leisure purposes will be misclassified as a business traveler. We estimate this error rate at roughly 20–25% based on Madrid tourism statistics showing approximately 30% of solo weekday visitors are leisure travelers. This does not invalidate the proxy — the signal still improves ranking on average — but context predictions should be understood as probabilistic rather than deterministic. A production system would replace this proxy with explicit user-stated trip purpose at the point of search.

### 2.2 Feature Engineering

**Listing feature vector (61 dimensions total):**

| Feature Group | Features | Dims |
|---------------|----------|------|
| Location | Neighbourhood OHE (top 30 + Other) | 31 |
| Price | log_price, price_tier (0/1/2), log_price_per_person, log_accommodates | 4 |
| Room | room_type OHE (4 types), log_bedrooms | 5 (was misreported as 7) |
| Host quality | response_rate, acceptance_rate, host_years, is_superhost | 4 |
| Review scores | rating, cleanliness, location, value, checkin, communication | 6 |
| Amenities | 10 binary flags | 10 |
| Booking policy | is_instant_book, log_min_nights | 2 |

We apply log transforms to price, accommodates, bedrooms, and min_nights to handle right-skewed distributions. All 61 features are L2-normalised before cosine similarity computation, ensuring that magnitude differences do not dominate similarity calculations.

**TF-IDF description features (150 dimensions):**  
We concatenate `description`, `neighborhood_overview`, and the raw amenity text string; clean to ASCII and common Spanish characters; then fit a TF-IDF vectoriser with bigrams, min_df=10, max_df=0.85, and max_features=150. The resulting 150-dimensional matrix captures qualitative listing character that numeric features cannot: "luminoso", "tranquilo", "close metro", "cozy apartment", "fully equipped kitchen". Top terms are verified by manual inspection to be substantively meaningful rather than stopwords.

**Hybrid weighting rationale:** The content-based similarity score combines structured features and TF-IDF as: `combined_sim = 0.65 × feature_cosine + 0.35 × description_cosine`. This 65/35 split was selected via grid search on a 20% holdout of the training set, testing splits from 50/50 to 80/20 in increments of 5 percentage points. The 65/35 split maximised Precision@10 on the holdout (0.228), outperforming 60/40 (0.221) and 70/30 (0.224). The result is intuitive: structured features are better-calibrated signals than free-text descriptions, which vary in length and quality across hosts.

---

## 3. Non-Personalized Recommender

### 3.1 Approach Description

The non-personalized recommender serves all users identically, ranking listings by aggregate quality without reference to individual preferences. It functions as both a practical fallback and the performance baseline against which all personalized approaches are measured.

**Raw Average:** Sort by mean `review_scores_rating`. This fails for listings with few reviews — a listing with one 5-star review ranks above an established listing averaging 4.9 with 500 reviews. We include this variant to quantify the failure mode.

**Bayesian Average (primary):**

```
BR(i) = [v(i) / (v(i) + m)] × R(i) + [m / (v(i) + m)] × C
```

where `v(i)` = review count, `R(i)` = mean rating, `m` = regularisation threshold, and `C` = 4.63 (global mean). Listings with fewer than m reviews shrink toward the global mean — the "wisdom of crowds" correction preventing low-volume outliers from dominating rankings.

**Hyperparameter sensitivity for m:** We treat m as a hyperparameter and test m ∈ {10, 25, 50, 100, 200} on the validation set. Precision@10 is stable between m=25 and m=100 (ranging 0.082–0.089), with degradation at the extremes: m=10 is too lenient (poorly-rated high-volume listings rank too highly), m=200 is too conservative (good listings with 50–199 reviews are suppressed). We select m=50 as the midpoint of the stable plateau, matching the convention used in the IMDb weighted rating system.

**Value-Weighted variant:** `value_score = bayesian_avg / log(1 + price)`. Rewards high-quality listings at accessible prices — a distinct and commercially relevant objective for budget-sensitive travelers.

### 3.2 Results

Top Bayesian Average listings span central barrios — Cortes, Embajadores, Sol, Justicia, Palacio, Simancas, and San Isidro — with 300–600 reviews and Bayesian scores above 4.93. The Value-Weighted variant shifts recommendations entirely toward ultra-budget shared rooms at €15–€18/night in outer barrios such as Vista Alegre, Quintana, and Pradolongo, surfacing high-scoring but very cheap listings that never appear in the Bayesian top-10.

### 3.3 Evaluation

| Metric | Bayesian Average |
|--------|-----------------|
| RMSE | 0.1459 |
| MAE | 0.0755 |
| Precision@10 | 0.0574 |
| Recall@10 | 0.0006 |
| NDCG@10 | 0.0607 |
| Coverage | 0.04% |
| Diversity | 0.002 |
| Serendipity | 0.000 |

RMSE and MAE are computed here because the non-personalized model produces explicit rating estimates, the only model in our pipeline for which these metrics are meaningful. Ranking metrics (Precision@10, Recall@10, NDCG@10) are near zero by construction: the system always returns the same global top-10 for every user, so the specific listing a traveller actually visited almost never appears in that list. Coverage of 0.04%, just 10 listings aprox out of 24,958 ever recommended, is the defining structural weakness and sets our performance floor. Every personalized approach must surpass Precision@10 = 0.0574 to justify added complexity.

---

## 4. Collaborative Filtering Recommender

### 4.1 Approach and Data Adaptation

**The data challenge:** The Inside Airbnb `reviews.csv` contains only `listing_id` and `date` — no `reviewer_id`. Standard user-item collaborative filtering is therefore impossible: we cannot build a user×item interaction matrix because users are unidentifiable.

**Our adaptation — temporal co-occurrence CF:** Rather than a user×item matrix, we build a **listing×month matrix** where each cell M[i,t] is the normalised review count for listing i in month t. The intuition is that listings with similar temporal review patterns attract similar types of travelers. If listing A and listing B both peak in July–August, plateau in spring and autumn, and dip in winter, they likely serve the same traveler cohort — for example, summer tourists seeking central city experience.

**Precise characterisation of what this signal proves:** This approach produces a valid item-item similarity model based on shared seasonal traveler patterns. It does not prove that the same individuals stayed at both listings — we have no user identity to make that claim. Two listings could share temporal patterns because they are in similar neighbourhoods, serve the same trip type, or are popular with the same nationality of traveler. We frame recommendations accordingly: rather than "users who booked X also booked Y" (which overstates what we can prove), we state "listings with similar seasonal traveler patterns to X." The signal is real and informative; we characterise it accurately.

### 4.2 Implementation

**Matrix construction:** We filter to reviews from January 2019 onward (66 months through June 2024), yielding a raw matrix of 13,142 listings × 66 months. We retain only listings with ≥10 reviews in this window, giving an active matrix of 8,722 listings × 66 months with 65.6% sparsity. Each listing's temporal vector is L2-normalised so that similarity reflects pattern shape rather than total review volume.

**SVD decomposition:** We decompose the normalised matrix using `scipy.sparse.linalg.svds` with k=50. The top-5 singular values are [61.07, 31.95, 23.67, 19.46, 15.83], with 13 factors needed to explain 80% of variance. Latent factor 1 (largest singular value) separates central tourist-heavy listings (Embajadores, private rooms) from quieter residential ones (Sol, entire apartments). Factors 2–10 capture seasonal patterns, weekend-heavy short-stay listings, and year-round business-district apartments.

**Ablation over k:** The top-5 singular values of the decomposition are [61.07, 31.95, 23.67, 19.46, 15.83], with a clear elbow after the first factor. Cumulative explained variance reaches 80% at k=13 factors; further factors capture increasingly noisy seasonal micro-patterns. We select k=50 as the elbow point that retains all primary seasonal archetypes (tourist-summer, business-year-round, weekend-short-stay) while remaining computationally tractable. Values above k=50 risk overfitting to COVID-era distributional noise in the 2020–2021 months.

**Item-item cosine similarity:** Cosine similarity is computed between SVD listing embeddings (U×Σ). For each query listing we retrieve the top-N most similar listings, excluding the query listing itself and applying budget or availability filters.

### 4.3 Evaluation

| Metric | CF (SVD k=50) |
|--------|--------------|
| Precision@10 | 0.0758 |
| Recall@10 | 0.0012 |
| NDCG@10 | 0.0802 |
| Coverage | 12.2% |
| Diversity | 0.1352 |
| Serendipity | 0.120 |

RMSE and MAE are not reported because the CF model produces ranked lists, not rating predictions. CF achieves higher Precision@10 than the non-personalized baseline (0.0758 vs 0.0574), and Coverage improves substantially to 12.2% — the model surfaces listings from across the catalog rather than always recommending the same central top-10. The Gini coefficient (0.909) remains high, indicating continued concentration, though lower than non-personalized (1.000). Diversity (0.1352) reflects temporal co-occurrence surfacing listing pairs that cross neighbourhood boundaries.

---

## 5. Content-Based Recommender

### 5.1 Approach Description

Content-based filtering recommends listings similar to a traveler's stated preferences or past interactions, using listing features alone. Three key advantages apply here: it requires no interaction data (fully functional for new users and new listings); it is explainable (we can tell a traveler exactly which features drove a recommendation); and it can recommend any listing in the catalog regardless of review history, giving the long tail of zero-review listings a path to visibility.

### 5.2 Feature Engineering and Similarity

We construct the 61-dimensional structured feature vector described in Section 2.2, supplemented by the 150-dimensional TF-IDF description embedding. The hybrid similarity score:

```
combined_sim = 0.65 × feature_cosine_similarity + 0.35 × description_cosine_similarity
```

The weight split was empirically validated via grid search on a holdout set (see Section 2.2). Both components are L2-normalised before combination.

### 5.3 Interactive Query Interface for Cold-Start Users

New users (no booking history) specify preferences via a structured query that generates a synthetic query vector in the same 61-dimensional feature space as the listing catalog:

```python
recs = build_query_vector(
    budget_max=120,
    neighbourhood_pref=['Embajadores', 'Sol', 'Universidad'],
    room_type_pref='Entire home/apt',
    group_size=2,
    must_haves=['wifi', 'kitchen'],
    prefer_superhost=True
)
```

Hard filters (budget ≤ max, amenities ∈ must_haves, accommodates ≥ group_size) narrow the candidate set. Within filtered candidates, soft scoring ranks by: Bayesian average + 0.1 × is_superhost + 0.05 × is_instant_book + 0.08 × value_bonus. Results include explanation tags (⭐ Superhost · 🍳 Kitchen · 📶 WiFi · ⚡ Instant book · 💰 Good value).

### 5.4 Evaluation

| Metric | Content-Based (Hybrid) |
|--------|----------------------|
| Precision@10 | 0.4096 |
| Recall@10 | 0.0127 |
| NDCG@10 | 0.4429 |
| Coverage | 16.98% |
| Diversity | 0.070 |
| Serendipity | 0.972 |

Content-based achieves the strongest ranking performance of all four models — Precision@10 = 0.4096, NDCG@10 = 0.4429 — because feature similarity provides a direct signal of listing compatibility regardless of review volume. Serendipity (0.972) is the highest of all models, reflecting that content similarity surfaces relevant listings that would never appear on a popularity ranking. Coverage (16.98%) is the highest of all approaches. RMSE and MAE are not applicable: ranked similarity outputs are not rating predictions.

---

## 6. Context-Aware Recommender

### 6.1 Context Dimensions and Statistical Validation

We model three context dimensions derived entirely from available metadata:

| Dimension | Values | Derivation |
|-----------|--------|------------|
| `season` | winter, spring, summer, autumn | Review date month |
| `trip_purpose` | leisure, business | Weekday + accommodates ≤ 2 → business; else leisure |
| `is_peak` | True / False | July, August, December |

Before building the CARS model, we validate statistically that context effects are real and not noise artifacts.

**Price by season (one-way ANOVA):** F-statistic = 42.78, p < 0.0001. Seasonal price variation is highly significant. Summer prices are 28% above the annual median; winter prices are 11% below.

**Neighbourhood preference by trip purpose (Chi-squared):** χ² = 880.84, p < 0.0001. Business-proxy reviews are more concentrated in Salamanca, Chamartín, and Recoletos. Leisure-proxy reviews are more concentrated in Embajadores, Sol, and Lavapiés.

**AC preference by season:** 34% higher AC mention rate in summer listings that receive summer-peak reviews versus the annual average. Winter reviews show 29% higher heating-mention rate in top-performing listings.

These statistical confirmations establish that context-aware modelling is empirically justified by the data — not merely theoretically motivated.

**Key context effects:**
- Summer travelers pay a 28% price premium; AC preference 34% above annual average
- Business travelers concentrate in Salamanca, Chamartín, Recoletos (proximity to business districts)
- Leisure travelers dominate Embajadores, Sol, Lavapiés (nightlife, restaurants, museums)
- Winter travelers show elevated preference for listings explicitly mentioning heating
- Peak-season listings show 41% higher review velocity than off-peak

### 6.2 Implementation: Pre-Filtering CARS

We implement a **pre-filtering contextual approach**, restricting the training review set to context-matching reviews before computing listing quality scores. The five-step pipeline:

1. Filter training reviews to the matching context (season + trip_purpose combination)
2. Count reviews per listing within that context slice
3. Compute a context-specific Bayesian Average using context review count as signal strength (same m=50 regularisation, applied to context-filtered counts)
4. Apply user hard filters (budget, neighbourhood, amenities, group size)
5. Rank by context score + soft bonuses (superhost, instant-book, value)

**Fallback path:** If fewer than 20 listings match the context filter (occurring in approximately 3.2% of evaluation requests, typically for rare context combinations such as business + winter + peak), we fall back to global Bayesian ranking. This prevents empty recommendation lists for edge cases.

**Pre-filtering vs. post-filtering:** We implement a pre-filtering contextual approach, which allows the Bayesian averaging to reflect context-specific review distributions rather than applying a context mask to scores computed from the full distribution.

### 6.3 Evaluation

| Metric | Context-Aware (CARS) |
|--------|---------------------|
| Precision@10 | 0.0680 |
| Recall@10 | 0.0007 |
| NDCG@10 | 0.0760 |
| Coverage | 0.09% |
| Diversity | 0.001 |
| Serendipity | 0.014 |

Context-Aware performs above the non-personalized baseline on NDCG (0.0760 vs 0.0607), confirming the seasonal signal adds ranking value. Coverage is very low (0.09%) because pre-filtering restricts the candidate pool to the listings active within a specific context combination. The Gini coefficient (1.000) indicates maximum concentration — the same small set of popular context-matched listings appears across all queries, which is the key limitation of this approach at our data scale.

---

## 7. Comparative Evaluation

### 7.1 Full Standardised Comparison Table

| Approach | RMSE | MAE | P@10 | R@10 | NDCG | Coverage | Diversity | Serendipity | Context |
|----------|------|-----|------|------|------|----------|-----------|-------------|---------|
| Non-Personalized | 0.1459 | 0.0755 | 0.0574 | 0.0006 | 0.0607 | 0.04% | 0.002 | 0.000 | None |
| CF (SVD k=50) | — | — | 0.0741 | 0.0009 | 0.0741 | 12.6% | **0.139** | 0.120 | Temporal |
| **Content-Based** | — | — | **0.4096** | **0.0127** | **0.4429** | **16.98%** | 0.070 | **0.972** | Listing features |
| Context-Aware | — | — | 0.0680 | 0.0007 | 0.0760 | 0.09% | 0.001 | 0.014 | Season + purpose |


**RMSE/MAE:** These metrics are reported only for Non-Personalized because it is the only model producing explicit rating-scale estimates (the Bayesian Average is a point estimate on the 1–5 rating scale, and RMSE measures prediction error against actual observed ratings). The CF, Content-Based, and Context-Aware models produce ranked recommendation lists, not rating predictions. Computing RMSE on cosine similarity scores or ranked positions would require an arbitrary and unjustifiable scale mapping, producing a metric that cannot be interpreted meaningfully. For retrieval tasks, NDCG is the appropriate accuracy measure and is reported for all models.

### 7.2 Key Findings

**Ranking quality:** Content-Based leads all models with Precision@10 = 0.4096 and NDCG@10 = 0.4429 — roughly 7× the non-personalized baseline (0.0574). The feature + TF-IDF hybrid captures listing compatibility signals that neither pure popularity nor temporal patterns can match. Context-Aware shows a modest improvement over the non-personalized baseline on NDCG (0.0760 vs 0.0607) but does not reach Content-Based levels under the unified NB05 evaluation protocol.

**Coverage:** Content-Based achieves the highest coverage (16.98%), followed by CF (12.6%). Non-Personalized (0.04%) and Context-Aware (0.09%) both effectively recommend the same tiny set of listings to all users, which is a critical limitation for platform revenue.

**Diversity:** CF achieves the highest diversity (0.139) among the four approaches, reflecting temporal co-occurrence surfacing listing pairs that cross neighbourhood boundaries. All models show high Gini concentration (0.853–1.000), indicating that further diversity injection at serving time remains important.

**Serendipity:** Content-Based achieves the highest serendipity (0.972), because feature similarity can surface niche listings highly compatible with a query but invisible to popularity-based approaches.

**No single model dominates across all dimensions.** Content-Based leads on precision, coverage, and serendipity; CF leads on diversity; Context-Aware adds a seasonal signal on top of the baseline. Production systems should cascade these approaches as proposed in Section 8.

### 7.3 Simulated A/B Test

To estimate real-world impact, we simulate an A/B test on the evaluation set. The Control arm uses Content-Based recommendations; the Treatment arm uses Context-Aware. Engagement is defined as a relevant listing appearing in the top-10 for a query.

Results: Control (Content-Based) engagement rate = 0.404, 95% CI [0.384, 0.423]; Treatment (Context-Aware) rate = 0.060, 95% CI [0.051, 0.070]. Relative lift = −85.0%. Two-proportion z-test: z = −28.746, p = 1.000 — **fail to reject H₀**.

The A/B simulation does not support replacing Content-Based with Context-Aware as the primary model; Context-Aware's narrower candidate pool (0.09% coverage) produces a much lower hit rate in this synthetic test. This is consistent with the offline metrics: Content-Based dominates on ranking quality. The appropriate role for Context-Aware is as a **re-ranker on top of Content-Based candidates**, not as a standalone recommender — which is exactly the cascade architecture described in Section 8.

### 7.4 Evaluation Methodology

All models are trained exclusively on pre-July 2024 data and evaluated on post-July 2024 reviews. Relevant items are defined as listings that received test-period reviews in the matching context (for CARS) or same neighbourhood (for CF/CB). We acknowledge this proxy introduces measurement error: a test-period review indicates the listing was booked, not necessarily that the traveler would have preferred it from a recommendation list. This is a limitation inherent to offline evaluation without click or booking conversion data, and motivates the live A/B test recommended in Section 10.

---

## 8. Business Case & Deployment Design

### 8.1 Interactive Recommendation Interface

Our models power a traveler preference query interface operating effectively for both cold-start and returning users:

```
┌────────────────────────────────────────────────┐
│  Find your perfect Madrid apartment            │
├────────────────────────────────────────────────┤
│  Budget per night      [ €50 ──── €150 ]       │
│  Neighbourhood         [ Embajadores ▼ ]       │
│  Travel purpose        [Leisure/Business] │
│  Group size            [ 2 people ]          │
│  Travel dates          [ Mar 15 → Mar 18 ]     │
│  Must-haves            [WiFi]  [Kitchen]   │
│  Superhost preferred   [✓]                     │
└────────────────────────────────────────────────┘
               [ Find My Apartment ]
```

Results include explanation tags (" Superhost ·  Kitchen · * 4.92 · €89/night · Popular this season"). Explainability builds traveler trust and reduces decision fatigue in a high-stakes booking context.

### 8.2 Production Architecture

```
[Traveler query / history]
           │
           ▼
[Candidate generation]  ─── Content-Based: budget, neighbourhood,
           │                 amenities, group size → ~200 candidates
           ▼
[Context-Aware ranker]  ─── Season + trip purpose pre-filtering → top-20
           │
           ▼
[Diversity injection]   ─── No single neighbourhood > 40% of top-10;
           │                 guarantee ≥ 1 non-superhost option
           ▼
[Response: top-10 with explanation tags]
```

**Technology stack:** Python (scikit-learn, scipy, pandas for model training); FastAPI (REST serving); Redis (candidate cache, 6h TTL); PostgreSQL (listing features + review index); Apache Airflow (nightly retraining pipeline); Streamlit (internal prototype).

**Latency:** < 100ms P99 end-to-end. Candidate generation is pre-computed and cached; context re-ranking is applied at query time over the 200-candidate set. Context scores are pre-computed nightly — only a lightweight Bayesian lookup and sort occurs at serving time. Prototype benchmarks: p50 = 23ms, p95 = 67ms, p99 = 94ms — within target.

### 8.3 ROI Analysis

We base the ROI estimate on the offline precision improvement of Content-Based over the non-personalized baseline (P@10: 0.4096 vs 0.0574 — roughly 7× ranking quality gain). We apply a conservative 15% realisation rate to translate offline ranking lift into booking conversion uplift, consistent with industry benchmarks for recommendation systems in travel platforms.

**Conservative scenario (15% realisation of offline ranking improvement):**

| | Current (Non-Personalized) | Content-Based |
|-|---------------------------|---------------|
| Assumed conversion uplift | — | +5.0% |
| Monthly MAU | 50,000 | 50,000 |
| Conversion rate | 7.5% | 7.9% |
| Monthly bookings | 3,750 | 3,975 |
| Avg booking value | €220 | €220 |
| Platform commission (12%) | €99,000 | €105,060 |
| **Monthly uplift** | — | **+€6,060** |
| **Annual uplift** | — | **+€72,720** |

**Optimistic scenario (25% realisation):**

| | Current | Content-Based |
|-|---------|---------------|
| Conversion rate | 7.5% | 8.4% |
| Monthly bookings | 3,750 | 4,200 |
| Monthly revenue | €99,000 | €110,880 |
| **Annual uplift** | — | **+€142,560** |

**Implementation cost:** €70,000 (one-time development) + €3,000/month (infrastructure and maintenance).

**Break-even:**
- Conservative: ~15 months (€70,000 / (€6,060 − €3,000))
- Optimistic: ~8 months (€70,000 / (€11,880 − €3,000))
- Year-3+ net benefit: €36,720/year (conservative) to €106,560/year (optimistic)

The A/B simulation (Section 7.3) showed that Context-Aware as a standalone model does not outperform Content-Based; its role is as a re-ranker within the cascade (Phase 3 of the rollout). The ROI figures above reflect the Content-Based system as the primary driver, with Context-Aware providing additional precision at no extra cost once the feature store is live.

---

## 9. Cold-Start & Bias Mitigation Strategy

### 9.1 Cold-Start

**New listing cold-start (20.6% of listings have zero reviews):** A four-stage cascade progressively incorporates richer signals as listings accumulate evidence:

- Stage 0 (0 reviews): Query-based content filtering only. A 30-day new-listing promotion boost (1.15× multiplier on final ranking score) ensures new listings appear in some top-10 lists, preventing permanent invisibility for new hosts.
- Stage 1 (1–9 reviews): Content-based similarity remains primary; Bayesian Average (heavily regularised to the global mean at this review count) incorporated as a secondary signal at 30% weight.
- Stage 2 (10–49 reviews): Content + CF hybrid (60/40 weighted combination). Sufficient temporal signal exists for a noisy CF embedding.
- Stage 3 (50+ reviews): Full Context-Aware pipeline. Sufficient review history to compute context-specific Bayesian Averages with meaningful statistical precision.

**New user cold-start (traveler's first search):** Handled entirely through explicit preference elicitation via the query interface (Section 5.3). No booking history is required. This converts the cold-start problem into a content-filtering problem, which our model handles at 16.98% catalog coverage (highest of all four approaches).

### 9.2 Bias Identification and Mitigation

**Popularity bias:** The top 5 neighbourhoods (20% of barrios) appear in 61% of non-personalized recommendations. Even context-aware models inherit this skew because popular areas generate disproportionate reviews. Mitigation: neighbourhood diversity penalty — no single barrio exceeds 40% of any top-10 list, guaranteeing at least 6 distinct barrios appear across each user's recommendations.

**Superhost bias:** Superhosts receive structurally higher ratings and dominate recommendations partly due to the selection mechanisms that create superhosts in the first place, not only because of genuine quality. A business traveler seeking a no-frills Chamartín apartment may be better served by a 4.6-rated non-superhost with instant booking than a 4.9-rated superhost with a 3-night minimum. Mitigation: guarantee ≥ 1 non-superhost in every top-10 list; evaluate within price tiers to prevent premium listings from crowding budget options.

**Temporal bias:** COVID-era reviews (2020–2021) show inflated scores due to reduced competition, survivor bias, and possible traveler gratitude during an exceptional period. Mitigation: 0.7× weight applied to 2020–2021 reviews in both the temporal CF matrix and Bayesian Average computation.

**Price tier bias:** Luxury listings in Salamanca and Recoletos aggregate to higher mean ratings (4.81) than budget listings in Carabanchel (4.43), partly because high-paying guests have higher expectations more frequently met by premium properties. Comparing across price tiers distorts quality rankings. Mitigation: evaluate and present recommendations within price tiers (budget: <€80/night; mid: €80–€200; premium: >€200).

**Recency bias:** A listing that was excellent in 2018 but has since declined in quality will still carry historical excellence in its Bayesian Average. Mitigation: compute a recency-weighted Bayesian Average using exponential decay (half-life = 18 months) as a supplementary flag for listings whose recent review trend diverges significantly from their historical average.

---

## 10. Conclusions

This project delivers a complete end-to-end recommendation pipeline for Madrid Airbnb listings, implementing all four required algorithmic paradigms on real Inside Airbnb data at production scale.

**Primary result:** Content-Based filtering achieves the strongest ranking performance — Precision@10 = 0.4096, NDCG@10 = 0.4429 — roughly 7× the non-personalized baseline (P@10 = 0.0574). The simulated A/B test compared Context-Aware against Content-Based and found no significant improvement from Context-Aware as a standalone model (lift = −85%, p = 1.000), confirming that Content-Based is the primary driver and Context-Aware is best deployed as a re-ranker on top of Content-Based candidates. Translated conservatively to revenue — applying a 15% realisation rate to the offline ranking improvement — this represents €72,720 annual uplift for a 50,000-MAU platform at 12% commission, with break-even in approximately 15 months.

**Key methodological contribution:** We demonstrate a practical collaborative filtering approach for datasets without explicit user-item interactions — temporal co-occurrence signatures using listing×month review matrices (8,722 listings × 66 months, 65.6% sparsity, k=50 SVD). We extract meaningful item-item similarity from review timing patterns while carefully characterising what the signal can and cannot prove, avoiding overstatement of the method's capabilities.

**Architecture contribution:** The two-stage cascade (content-based candidate generation at 16.98% coverage → context-aware re-ranking for seasonal precision) directly addresses the coverage-accuracy tradeoff in real-world recommendation systems. Each model contributes its comparative advantage: content-based for ranking quality and serendipity, context-aware for seasonal relevance, CF for diversity and long-tail discovery.

**Limitations:** Three limitations constrain this project. First, the absence of reviewer IDs prevents user-level personalization in CF; a production system should invest in explicit post-stay rating collection with verified reviewer IDs. Second, the trip_purpose_proxy introduces approximately 20–25% classification noise; direct user elicitation or clickstream signals would substantially improve context accuracy. Third, offline evaluation using review occurrence as a proxy for relevance introduces measurement error; a live A/B test with booking conversion as the primary outcome metric is the necessary next step to provide definitive evidence of system value.

---

## 11. Individual Contributions

| Member | Primary Role | Sections | Hours |
|--------|-------------|----------|-------|
| Jimena Navarro | Domain analysis, EDA, data preprocessing | §1, §2 | ~12h |
| Sofia Serrantes | Non-personalized recommender, CF support | §3, §4 (partial) | ~11h |
| Berni Andrade | Content-based recommender, feature engineering, query interface | §2 (feature eng.), §5 | ~12h |
| Paula Evangelista | Context-aware model, evaluation framework (NB05) | §6, §7 | ~13h |
| Tessa Correig | Business case, bias analysis, deployment design, Streamlit prototype | §8, §9 | ~10h |
| Daniel Teixidor | CF implementation, integration, cross-notebook testing | §4, §10–§12 | ~10h |
| All members | Report writing, presentation rehearsal | §10–§12 | ~7h |

**Total: ~75 hours** across the team.

---

## 12. AI Usage Disclosure

- **Writing assistance:** Claude (Anthropic) was used to improve grammar, clarity, and structure in Sections 1, 8, and 9. All analytical conclusions, numerical results, business logic, and interpretations are entirely our own.
- **Code assistance:** GitHub Copilot was used for boilerplate autocompletion in pandas and scipy idioms (standard DataFrame merge patterns, scipy.sparse matrix construction). All algorithmic logic — the Bayesian Average formulation, SVD construction and k-selection, TF-IDF weighting scheme, CARS pre-filtering pipeline, hybrid weighting grid search, diversity injection logic — was designed and implemented by team members.
- **No AI tool generated metric values, evaluation results, statistical tests, or business estimates.** All numbers in this document are computed from model runs on the actual Inside Airbnb Madrid dataset, with code available in the submitted notebooks for full reproducibility.
- **Verification:** All AI-assisted passages were reviewed and edited by at least two team members before inclusion. The team takes full responsibility for the accuracy of all content in this report.
