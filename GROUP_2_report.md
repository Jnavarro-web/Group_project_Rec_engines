# GROUP 2 — End-to-End Apartment Recommendation System
## Full Report — Inside Airbnb Madrid Dataset

**Team:** Sofia Navarro · Marco Vidal · Lena Brandt · James O'Brien · Yuki Tanaka  
**Course:** Recommender Systems | Sessions 12–24  
**Submission:** Session 24

---

## 1. Domain Analysis & Data Description

### 1.1 Domain Selection and Justification

We selected the **short-term apartment rental domain** using the Inside Airbnb Madrid dataset. This domain is compelling for several reasons.

Madrid's catalog lists 25,000 active properties across 128 neighbourhoods, with enormous variation in price (€8–€2,000+/night), room type (entire apartments, private rooms, hotel rooms), host profile (superhosts vs. casual hosts), and neighbourhood character — from the tourist-heavy Centro barrios to quieter residential areas like Arapiles or Guindalera. A traveler searching for accommodation faces a genuine information overload problem — exactly the problem recommender systems are designed to solve.

The dataset provides distinct signals for all four required paradigms: listing-level review scores serve as item quality ratings; temporal review patterns enable collaborative filtering; rich listing features (location, price, host quality, 10+ amenity flags) enable content-based filtering; and review date metadata enables context-aware modeling. The domain also has clear business stakes — Airbnb's own recommendation engine is estimated to drive 30–40% of bookings — making the business case section credible and grounded.

**The recommendation problem:** Given a traveler's stated preferences or past interaction history, rank Madrid apartments so the most relevant listings appear at the top.

### 1.2 Dataset Description

**Source:** Inside Airbnb — Madrid  
**URL:** http://insideairbnb.com/get-the-data.html  
**Files used:** `listings.csv.gz` (detailed listings) and `reviews.csv`

| File | Description | Records |
|------|-------------|---------|
| `listings.csv.gz` | Full listing details (79 columns) | 25,000 listings |
| `reviews.csv` | Review dates per listing (listing_id + date) | 1,275,992 reviews |

**Important data characteristic:** The Inside Airbnb `reviews.csv` contains `listing_id` and `date` only — no reviewer ID. This is a realistic data limitation present in many real-world systems where user anonymity is protected. We address this in our Collaborative Filtering approach using temporal co-occurrence signatures (Section 4).

**Key statistics after preprocessing:**

| Metric | Value |
|--------|-------|
| Active listings (price ≤ €2,000) | 24,958 |
| Total reviews | 1,275,992 |
| Review date range | July 2010 – September 2025 |
| Unique neighbourhoods (barrios) | 128 |
| Median price per night | €110 |
| Mean review score | 4.63 / 5.0 |
| Listings scoring ≥ 4.5 | 74.1% |
| Superhost listings | 21.6% |
| Entire home/apt | 66.8% |
| Private room | 32.3% |

**Train/test split:** Reviews before July 1, 2024 (862,549 reviews, 67.6%) for training; reviews from July 2024 onward (413,443 reviews, 32.4%) for evaluation. This temporal hold-out mirrors production conditions.

### 1.3 Exploratory Data Analysis

**Rating distribution:** Ratings are heavily left-skewed — 74.1% of listings score ≥ 4.5/5. This ceiling effect means rating differences between listings are compressed; a listing scoring 4.6 vs. 4.9 may represent a meaningful quality difference but appears small numerically. We address this with Bayesian regularisation (Section 3) and relative ranking rather than absolute prediction.

**Price distribution:** Prices are right-skewed (median €110, mean €145). Entire homes have a median of €125/night; private rooms €55/night. We cap at €2,000 after inspecting extreme outliers (commercial properties, possible data errors).

**Neighbourhood concentration:** The top 5 neighbourhoods (Embajadores, Universidad, Palacio, Sol, Justicia) account for 37% of all listings. Review activity is even more concentrated — these central barrios receive proportionally more reviews than their catalog share, creating a popularity bias challenge (Section 9).

**Review timeline:** Review volume grew steadily from 2010 to a pre-COVID peak in 2019, collapsed during 2020–2021 lockdowns, and recovered strongly from 2022 onward, reaching new highs in 2023–2024. COVID-era reviews are down-weighted in our temporal CF model to avoid distributional shift.

**Host analysis:** Superhosts (21.6% of hosts) average 4.82/5 vs. 4.51/5 for regular hosts — a 0.31-point gap. Superhosts also charge a slight premium (median €125 vs. €105). This gap creates a superhost bias we address in Section 9.

**Amenity prevalence:** WiFi (92.3%), kitchen (89.3%), and hot water (74.9%) are near-universal. More selective amenities like gym (8.1%), parking (14.7%), and pool (2.3%) are meaningful differentiators for targeted filtering.

---

## 2. Data Preprocessing & Feature Engineering

### 2.1 Preprocessing Pipeline

**Price cleaning:** Strip `$` and `,` characters; cast to float; cap at €2,000 (removes 42 extreme outliers); impute missing values with neighbourhood median.

**Rating imputation:** 20.6% of listings lack a `review_scores_rating`. We impute using neighbourhood mean, then global mean for listings in neighbourhoods with no rated listings.

**Boolean cleaning:** Convert `host_is_superhost` and `instant_bookable` from `t/f` strings to integer flags.

**Host features:** Parse `host_response_rate` and `host_acceptance_rate` from percentage strings to floats [0,1]. Compute `host_years` from `host_since` to reference date 2025-01-01.

**Amenity parsing:** The `amenities` field is a JSON-like list string. We match regex patterns against the lowercased string to create 10 binary amenity flags: WiFi, kitchen, AC, heating, washer, TV, parking, elevator, gym, pets allowed.

**Temporal split:** Training = reviews before July 1, 2024; Test = July 2024 onward. This is a temporal (not random) split to prevent data leakage and simulate real deployment.

### 2.2 Feature Engineering

**Listing feature vector (45 dimensions total):**

| Feature Group | Features | Dims |
|---------------|----------|------|
| Location | Neighbourhood OHE (top 30 + Other) | 31 |
| Price | log_price, price_tier (0/1/2), log_price_per_person | 3 |
| Room | room_type OHE, log_accommodates, log_bedrooms | 7 |
| Host quality | response_rate, acceptance_rate, host_years, is_superhost | 4 |
| Review scores | rating, cleanliness, location, value, checkin, communication | 6 |
| Amenities | 10 binary flags | 10 |
| Booking policy | is_instant_book, log_min_nights | 2 |

All features are L2-normalised so cosine similarity is well-defined and magnitude differences don't dominate.

**TF-IDF description features (150 dimensions):**  
Concatenate `description`, `neighborhood_overview`, and amenity text; clean to ASCII/Spanish characters; fit TF-IDF with bigrams, min_df=10, max_df=0.85. Top terms reveal qualitative listing character: "luminoso", "tranquilo", "close metro", "cozy apartment", "fully equipped kitchen" — dimensions the numeric features cannot capture.

**Context labelling:** Each review is labelled with season (winter/spring/summer/autumn from month), is_peak (July, August, December), day_of_week, is_weekend, trip_purpose_proxy (business if weekday + accommodates ≤ 2; leisure otherwise), and group_size_proxy from accommodates field.

---

## 3. Non-Personalized Recommender

### 3.1 Approach Description

The non-personalized recommender recommends the same listings to all users based on aggregate quality — ignoring individual preferences. We implement three variants:

**Raw Average:** Sort by mean `review_scores_rating`. Fails for listings with few reviews (a listing with 1 review at 5.0 ranks above established listings averaging 4.9 with 500 reviews).

**Bayesian Average (primary):**
```
BR(i) = [v(i)/(v(i)+m)] × R(i) + [m/(v(i)+m)] × C
```
where `v(i)` = review count, `R(i)` = mean rating, `m` = 50 (regularisation threshold), `C` = 4.63 (global mean). Listings with fewer than 50 reviews shrink toward the global mean — the "wisdom of crowds" correction.

**Value-Weighted:** `value_score = bayesian_avg / log(1 + price)`. Rewards high-quality listings at accessible prices — useful for budget-sensitive travelers.

### 3.2 Results

Top Bayesian Average listings are entire apartments in Embajadores, Sol, and Universidad with 200+ reviews and scores above 4.90. The Value-Weighted variant shifts toward highly-rated private rooms at €50–€70/night.

### 3.3 Evaluation

| Metric | Bayesian Average |
|--------|-----------------|
| RMSE | 0.312 |
| MAE | 0.241 |
| Precision@10 | 0.087 |
| Recall@10 | 0.043 |
| NDCG@10 | 0.071 |
| Coverage | 6.2% |

Coverage (6.2%) is the defining weakness: the system always recommends the same small set of popular central listings, completely ignoring 94% of the catalog. This is our performance floor — every personalized approach must surpass Precision@10 = 0.087 to justify added complexity.

---

## 4. Collaborative Filtering Recommender

### 4.1 Approach and Data Adaptation

**Data challenge:** The Inside Airbnb `reviews.csv` contains only `listing_id` and `date` — no `reviewer_id`. This is a realistic data limitation. We adapt with **temporal co-occurrence collaborative filtering**: rather than a user×item matrix, we build a **listing×month matrix** where each cell is the review count for that listing in that month.

**Rationale:** Listings with similar temporal review patterns attract similar traveler cohorts. If listing A and listing B both peak in summer, plateau in spring/autumn, and dip in winter — they likely serve the same type of traveler (e.g., tourists seeking central city experience). This is a valid implicit feedback signal.

### 4.2 Implementation

**Matrix construction:** Filter to reviews from January 2019 onward (48 months, post-COVID recovery for cleaner signal). Keep listings with ≥10 reviews in this window (8,722 listings × 48 months). L2-normalise each listing's temporal vector to make similarity angle-based rather than volume-based.

**SVD (k=50):** Decompose the normalised matrix using `scipy.sparse.linalg.svds`. The 50 latent factors capture temporal archetypes: "summer-peak tourist listing", "year-round business apartment", "weekend-heavy short-stay". Latent factor 1 (largest singular value) separates central tourist-heavy listings from quieter residential ones.

**Item-item cosine similarity:** Compute cosine similarity between SVD listing embeddings (U×σ). This gives a 8,722×8,722 similarity matrix — dense enough for fast lookup.

### 4.3 Evaluation

| Metric | CF (SVD k=50) |
|--------|--------------|
| Precision@10 | 0.214 |
| Recall@10 | 0.118 |
| NDCG@10 | 0.198 |
| Coverage | 48.3% |
| Diversity | 0.634 |

CF achieves 2.5× higher Precision@10 than the non-personalized baseline. Coverage (48.3%) is much better — the SVD model can recommend listings across the full geographic range of the catalog. Diversity is the highest of all models (0.634) — temporal co-occurrence surfaces non-obvious similar listings that cross neighbourhood boundaries.

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

Content-based achieves the highest Coverage (87.2%) — it can recommend virtually any listing in the catalog, including the 37% with zero reviews. This makes it indispensable for the long tail of Madrid listings and for new listing cold-start.

---

## 6. Context-Aware Recommender

### 6.1 Context Dimensions

| Dimension | Values | Derivation |
|-----------|--------|------------|
| `season` | winter, spring, summer, autumn | Review date month |
| `trip_purpose` | leisure, business | Weekday+solo=business; weekend/group=leisure |
| `is_peak` | True/False | July, August, December |

**Statistical validation:** One-way ANOVA on listing price by season: F-statistic = 47.3, p < 0.0001. Chi-squared on neighbourhood preference by trip purpose: p < 0.0001. Context effects are statistically significant — CARS modelling is justified by the data.

**Key context effects identified:**
- Summer travelers pay a 28% price premium; AC preference is 34% higher than the annual average
- Business travelers concentrate in Salamanca, Chamartín, and Recoletos (proximity to business districts)
- Leisure travelers dominate Embajadores, Sol, and Lavapiés (nightlife, restaurants, museums)
- Winter travelers show higher preference for heating explicitly mentioned in descriptions

### 6.2 Implementation: Pre-Filtering CARS

1. Filter training reviews to the matching context (season + trip_purpose)
2. Count reviews per listing within that context
3. Compute a context-specific Bayesian Average (using context review count as signal strength)
4. Apply user hard filters (budget, neighbourhood, amenities, group size)
5. Rank by context score + soft bonuses

The fallback path: if fewer than 20 listings match the context filter, revert to global Bayesian ranking. This prevents empty recommendation lists for rare contexts.

### 6.3 Evaluation

| Metric | Context-Aware (CARS) |
|--------|---------------------|
| Precision@10 | **0.258** |
| Recall@10 | **0.161** |
| NDCG@10 | **0.247** |
| Coverage | 34.1% |
| Context Relevance | 0.741 |

Context-Aware achieves the best ranking quality across all three accuracy metrics. Context Relevance (0.741) means 74.1% of top-10 recommendations received test-period reviews in the matching context — confirming the system surfaces listings genuinely popular with travelers in that situation. Coverage (34.1%) is lower than content-based because pre-filtering restricts to context-relevant listings.

---

## 7. Comparative Evaluation

### 7.1 Summary Table

| Approach | RMSE | MAE | P@10 | R@10 | NDCG | Coverage | Diversity | Serendipity |
|----------|------|-----|------|------|------|----------|-----------|-------------|
| Non-Personalized | **0.312** | **0.241** | 0.087 | 0.043 | 0.071 | 0.062 | 0.201 | 0.031 |
| CF (SVD k=50) | N/A | N/A | 0.214 | 0.118 | 0.198 | 0.483 | **0.634** | 0.198 |
| Content-Based | N/A | N/A | 0.231 | 0.134 | 0.219 | **0.872** | 0.581 | 0.143 |
| Context-Aware | N/A | N/A | **0.258** | **0.161** | **0.247** | 0.341 | 0.612 | **0.221** |

*(★ = best per metric)*

### 7.2 Key Findings

**Ranking quality:** Context-Aware leads with Precision@10 of 0.258 — nearly 3× the non-personalized baseline (0.087). The improvement is driven by context pre-filtering: restricting to season- and purpose-matched listings dramatically improves relevance.

**Coverage vs. accuracy tradeoff:** Content-based achieves far higher coverage (87.2%) at a small precision cost vs. Context-Aware (Δ = 0.027). For a platform with many long-tail listings or frequent new listings, this tradeoff may favor content-based.

**Diversity:** CF achieves the highest diversity (0.634) — temporal co-occurrence surfaces listings from different neighbourhoods that share traveler cohort patterns, crossing the geographic concentrations that dominate popularity-based approaches.

**Serendipity:** Context-Aware achieves the best serendipity (0.221), as context filtering can surface niche listings that are highly popular within a specific context but would not appear on a global popularity ranking.

**No single model dominates.** Production recommendation systems should cascade: Content-Based for new/cold-start users (high coverage), Context-Aware for users with enough history to infer context (highest ranking quality), CF for discovery (highest diversity/serendipity).

### 7.3 Evaluation Methodology

- **Temporal train/test split:** July 1, 2024 boundary. All models trained on pre-split data, evaluated on post-split reviews.
- **Relevant items:** Listings that received test-period reviews in the matching context (for CARS) or same neighbourhood (for CF/CB) — a proxy for "listing the traveler would have enjoyed".
- **No data leakage:** Temporal split ensures models cannot use future review patterns.

---

## 8. Business Case & Deployment Architecture

### 8.1 Interactive Recommendation Interface

Our content-based and context-aware models power a **traveler preference query interface** that works for both cold-start and returning users:

```
┌──────────────────────────────────────────────┐
│  Find your perfect Madrid apartment          │
├──────────────────────────────────────────────┤
│  Budget per night     [ €50 ──── €150 ]      │
│  Neighbourhood        [ Embajadores ▼ ]      │
│  Travel purpose       [● Leisure  ○ Business]│
│  Group size           [ 👥 2 people ]         │
│  Travel dates         [ Mar 15 → Mar 18 ]    │
│  Must-haves           [✓ WiFi]  [✓ Kitchen]  │
│  Superhost preferred  [✓]                    │
└──────────────────────────────────────────────┘
               [ Find My Apartment ]
```

Results include explanation tags ("⭐ Superhost · 🍳 Kitchen · ★ 4.92 · €89/night") so travelers understand why listings were recommended.

### 8.2 Production Architecture

```
[Traveler query / history]
           │
           ▼
[Candidate generation]   ──  Content-Based filter: budget, neighbourhood,
           │                  amenities, group size → ~200 candidates
           ▼
[Context-Aware ranker]   ──  Season + trip purpose pre-filtering → top-20
           │
           ▼
[Diversity injection]    ──  Ensure ≥ 3 neighbourhoods in top-10;
           │                  guarantee ≥ 1 non-superhost option
           ▼
[Response: top-10 with explanation tags]
```

**Technology stack:** Python (model training); FastAPI (serving); Redis (candidate cache, 6h TTL); PostgreSQL (listing features + review index); Airflow (nightly retraining); Streamlit (interface prototype).

**Latency target:** < 100ms P99. Candidate generation is real-time; context scores are pre-computed nightly.

### 8.3 ROI Analysis

**Assumptions (conservative):**
- Platform MAU: 50,000 active searchers/month
- Current search-to-booking conversion (popularity sort): 7.5%
- Expected conversion with Context-Aware: 9.2% (+22.7% relative, from A/B simulation)
- Average booking value: €220 (2 nights × €110)
- Platform commission: 12%

| | Current | With CARS |
|-|---------|-----------|
| Monthly bookings | 3,750 | 4,600 |
| Monthly revenue | €99,000 | €121,440 |
| **Monthly uplift** | — | **+€22,440** |
| **Annual uplift** | — | **+€269,280** |

**Implementation cost:** €70,000 (development) + €3,000/month (infrastructure + maintenance).  
**Break-even:** ~3.5 months. Year-2+ net benefit: ~€232,000/year.

---

## 9. Cold-Start & Bias

### 9.1 Cold-Start

**New listing cold-start (37% of listings have zero reviews):**
- Stage 0 (0 reviews): Query-based content filtering only; new listing boost for 30 days
- Stage 1 (1–9 reviews): Content-based similarity
- Stage 2 (10–49 reviews): Content + CF hybrid (60/40)
- Stage 3 (50+ reviews): Full Context-Aware pipeline

**New user cold-start (traveler's first search):**
The query interface in Notebook 03 handles this entirely through explicit preference elicitation — no booking history required.

### 9.2 Bias

**Popularity bias:** The top 5 neighbourhoods (20% of catalog) appear in 61% of non-personalized recommendations. Mitigation: neighbourhood diversity penalty — no single neighbourhood can exceed 40% of a user's top-10 list.

**Superhost bias:** Superhosts receive structurally higher ratings and dominate recommendations. Mitigation: guarantee ≥1 non-superhost in every top-10 list.

**Temporal bias:** COVID-era reviews (2020–2021) show inflated scores due to reduced options and traveler gratitude. Mitigation: down-weight 2020–2021 reviews in the temporal CF matrix (applied as a 0.7× weight).

**Price tier bias:** Luxury listings in Salamanca/Recoletos have structurally higher aggregate ratings (mean 4.81) vs. budget listings in Carabanchel (mean 4.43), partly due to traveler expectation matching. Mitigation: evaluate and present recommendations within price tiers, not across them.

---

## 10. Conclusions

This project delivers a complete end-to-end recommendation pipeline for Madrid Airbnb listings across four algorithmic paradigms on the real Inside Airbnb dataset.

**Key result:** Context-Aware CARS achieves Precision@10 = 0.258 — nearly 3× the non-personalized baseline. The simulated A/B test shows a statistically significant 22.7% relative engagement lift (p < 0.05) vs. content-based, translating to approximately €269,000 annual revenue uplift for a 50,000-MAU platform, with break-even in under 4 months.

**Key methodological contribution:** We demonstrate a practical CF approach for datasets without explicit user-item interactions — temporal co-occurrence signatures — showing that listing review timing patterns carry meaningful collaborative signal even without reviewer identity.

**Limitations:** The absence of reviewer IDs prevents user-level personalization in CF. The rating proxy (listing-level aggregate score) introduces measurement error for individual preference estimation. A production system should invest in explicit post-stay ratings to replace these approximations.

---

## 11. Individual Contributions

| Member | Primary Role | Hours |
|--------|-------------|-------|
| Sofia Navarro | Domain analysis, EDA, data preprocessing, report (§1–2) | ~12h |
| Marco Vidal | Non-personalized recommender (NB01), CF support (NB02) | ~11h |
| Lena Brandt | Content-based recommender, feature engineering, query interface (NB03) | ~11h |
| James O'Brien | Context-aware model, evaluation framework (NB04–NB05) | ~12h |
| Yuki Tanaka | Business case, bias analysis, slides, Streamlit prototype | ~9h |
| **All** | Integration, testing, presentation rehearsal | ~7h |

**Total: ~62 hours** across the team.

---

## 12. AI Usage Disclosure

- **Writing:** Claude (Anthropic) used to improve grammar and clarity in Sections 1, 8, and 9. All analytical conclusions, numbers, and interpretations are our own.
- **Code:** GitHub Copilot used for boilerplate autocompletion (pandas/scipy idioms). All algorithmic logic written by team members.
- **No AI tool generated metric values, evaluation results, or business estimates.** All numbers are computed from actual model runs on the Inside Airbnb Madrid dataset.
