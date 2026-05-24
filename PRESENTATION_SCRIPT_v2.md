# Presentation Script — MissVARPath (v2, 24-slide deck)
**Total runtime target: 10 minutes (≈ 5 + 5).**
24 slides, ~25 s/slide average. Speaking rate: ~150 wpm.
Slide cues marked `[slide N]`. Stage directions in *italics*.
Pause beats marked `…` short pause / `………` bigger pause.

> **What's new vs. v1:** three dedicated per-task leaderboard slides
> (9 = 2-class, 10 = 3-class with VUS, 11 = 5-class with VUS) inserted
> after the canonical headline slide. All downstream slide numbers
> shifted by +3. The handoff between speakers moves from slide 10 →
> slide 13 (after Stress tests) to keep the narrative arc clean.

---

## Speaker A — Setup, baselines, leaderboards, stress tests  (slides 1–13, ≈ 5 minutes)

### Slide 1 — Title  (10 s)

*[slide 1: title, names, course code, university]*

"Good afternoon. We're presenting **MissVARPath**, our project on
missense variant pathogenicity prediction with a focus on Variants of
Uncertain Significance — VUS. I'll cover the setup, data, methods, and
the canonical results, then **[name]** will take you through the VUS
deployment, the ablation study, and our final ablation-optimal model."

---

### Slide 2 — Three questions  (45 s)

*[slide 2: the three numbered questions]*

"In clinical genomics, every missense variant goes through a battery of
in-silico predictors — **PolyPhen-2, SIFT, REVEL, CADD, AlphaMissense,
DITTO**. Each disagrees with the others, the predictors have
**correlated training data**, and most were themselves trained with at
least partial overlap on **ClinVar** — the same database we use to
label our targets.

So the three questions our project asks:
**One**, can a meta-classifier do better than any single predictor by
combining them?
**Two**, how much of any improvement is data-circularity — predictors
trained on ClinVar, evaluated on ClinVar?
**Three**, can we predict pathogenicity from raw data alone — without
any pre-existing predictor's score?"

---

### Slide 3 — Data  (25 s)

*[slide 3: big numbers + 4 task definitions]*

"Our dataset is **21,872 missense variants** from ClinVar, annotated
through OpenCRAVAT's pipeline. Four task definitions: the original
4-class label, a 2-class collapse, and two extensions where VUS is
added as its own class — 3-class and 5-class. The raw schema is 777
columns; after preprocessing, **208 numerical features**."

---

### Slide 4 — Label balance  (20 s)

*[slide 4: label distribution bars]*

"The corpus is perfectly balanced — **5,468 rows per ClinVar class**,
and 10,936 per side in the 2-class collapse. Any class-imbalance
effect is gone by construction."

---

### Slide 5 — Missingness  (20 s)

*[slide 5: missingness histogram]*

"The missingness landscape across the raw 777 columns. Everything
above **50% missing** drops out in preprocessing — that's 134
columns. Combined with 98 identifier and text columns, we land at
208 numerical features."

---

### Slide 6 — Predictor scores separate classes  (25 s)

*[slide 6: VEP boxplots by class]*

"Quick sanity check before benchmarking — do the predictor scores
actually separate classes? **DITTO, MetaRNN, REVEL, AlphaMissense**
all separate cleanly from Benign through Pathogenic. **gnomAD AF**
runs the other way — high frequency means benign, the ACMG BS1 rule
encoded in the data. The signal is there."

---

### Slide 7 — Methods  (30 s)

*[slide 7: model suite + 5 regimes]*

"We train a **10-model suite** — nine classical and linear methods
plus HistGradientBoosting. Stratified 80/20 holdout plus 5-fold CV,
`RANDOM_STATE = 42` throughout.

Five evaluation regimes: canonical kfold, an augmented variant with
k-mer and BLAST locus-neighbour columns, gene-stratified CV, a
no-VEP ablation that drops every predictor and conservation score,
and a raw-only ablation that keeps only population frequency,
position, and sequence composition."

---

### Slide 8 — Headline numbers (4-class focus)  (35 s)

*[slide 8: 0.795 / 0.987 big numbers + 4-class bar chart]*

"Headline numbers. With `HistGradientBoosting` as the consistent top
model: macro-F1 of **0.795** on the 4-class task, **0.987** on the
2-class task.

The 4-class leaderboard on the right shows the band structure:
HistGB on top, LinearSVC and DecisionTree at 0.75, the linear family
in the low 0.70s, centroid and KNN families around 0.55 to 0.66."

---

### Slide 9 — 2-class leaderboard  (20 s)  *(NEW)*

*[slide 9: 0.987 + 2-class bar chart]*

"Zooming in on 2-class: the leaderboard is **extremely tight**.
HistGradientBoosting and LinearSVC are tied to three decimal places
at 0.987. SGD and DecisionTree at 0.985 and 0.982. **Seven of the
ten models clear 0.96.** The 2-class task is essentially solved by
this suite."

---

### Slide 10 — 3-class with VUS leaderboard  (25 s)  *(NEW)*

*[slide 10: 0.946 + 3-class bar chart]*

"For 3-class — Benign-side, Pathogenic-side, **VUS as the third
class** — HistGB reaches **0.946**. Note the gap to second place:
DecisionTree at 0.904, then LinearSVC at 0.876. The booster's
margin over the rest of the suite is much wider here — about four
percentage points — because the 3-class task has more class-pair
boundaries to disentangle. We'll come back to why VUS is so
recognisable in a moment."

---

### Slide 11 — 5-class with VUS leaderboard  (20 s)  *(NEW)*

*[slide 11: 0.798 + 5-class bar chart]*

"And on 5-class — all four ClinVar labels plus VUS — HistGB lands
at **0.798**. That's actually *slightly higher* than the no-VUS
4-class baseline of 0.795. The leaderboard structure mirrors
canonical 4-class: HistGB on top, DecisionTree and LinearSVC in the
0.72 band, linear models around 0.66."

---

### Slide 12 — Where the error lives  (25 s)

*[slide 12: 4-class confusion matrix]*

"This is the canonical 4-class confusion matrix, held-out, row-normalized.
**Benign at 90%, Likely-benign at 96%** diagonal. Then look at the
bottom-right: **Likely-pathogenic at 68%, Pathogenic at 64%** — with
about 30% of each bleeding into the other. The residual difficulty
in the 4-class task is one specific boundary: distinguishing
Pathogenic from Likely-pathogenic."

---

### Slide 13 — Stress tests + handoff  (35 s)

*[slide 13: anchor table across the 5 regimes]*

"To bound how much of the headline is data-circularity, we ran
four stricter regimes.

**Gene-stratified CV** — no gene in both train and test — costs
**2.7 points** on 4-class, essentially nothing on 2-class.
**No-VEP** — drop every learned predictor — costs only 2.2 points.
**Raw-only** — predict from data, not from predictors — costs
7.4 points but the booster is still well above chance.

The 2-class task is essentially insensitive to every stress.
That's the operationally-deployable headline.

………

**[name]**, take it away."

*[hand-off to Speaker B]*

---
---

## Speaker B — VUS deployment, ablation, optimal model, future  (slides 14–24, ≈ 5 minutes)

### Slide 14 — Real VUS deployment  (40 s)

*[slide 14: per-gene VUS prediction chart]*

"Thanks. So far we've evaluated on labelled ClinVar variants. The
whole point of an in-silico classifier is to score **unclassified**
variants — VUS — that nobody has labelled yet.

We curated a **5,468-row VUS pro-set** from ClinVar — 3-star
expert-reviewed plus a 2-star sample — annotated through the same
OpenCRAVAT pipeline.

This is the most clinically interesting result we have. Per-gene
prediction breakdown for the top-20 genes in the pro-set: the model
recovers clinically sensible priors that it could only have learned
indirectly from gene-correlated features. **GCK at 100%** pathogenic,
**PAH at 99%**, **MYH7 at 90%**, **LDLR at 84%** — every one of
these matches the published literature."

---

### Slide 15 — The four numbers  (30 s)

*[slide 15: 2x2 grid — 62/38, 80%, 87%, 95.4%]*

"Four numbers from that deployment.

**62/38** — pathogenic-side / benign-side split on the 2-class head.
**80%** — eighty percent of predictions have max probability above
0.9; the model is confident on most.
**87%** — on the 4-class head, 87% of predictions hedge into
Likely-star categories. Only 13% are definitive calls.
**95.4%** — when we collapse 4-class to pathogenic-side / benign-side
and compare with the 2-class head, the two — trained
independently — agree on 95.4 percent of variants."

---

### Slide 16 — VUS as a class  (30 s)

*[slide 16: comparison + per-class F1 bar chart]*

"Can the model learn to **recognise** VUS — not just score them.
We added strict-VUS as a labelled class. Two new task definitions —
3-class and 5-class.

We saw the leaderboard numbers earlier. The striking result is
5-class macro-F1: **0.798**, *higher* than the no-VUS 4-class
baseline of 0.795. Adding VUS doesn't hurt — MCC improves
materially too.

Per-class F1 chart: **VUS reaches 0.901** — between Benign and
Likely-benign in difficulty. **One of the cleanest classes to
recognise.**"

---

### Slide 17 — 5-class confusion matrix  (20 s)

*[slide 17: 5-class confusion matrix]*

"The 5-class confusion matrix. VUS row, bottom: **0.94 diagonal**.
When the model misses on a VUS variant — six percent of them —
it hedges into the **Likely-star** categories. Only **1 percent**
escalate to a definitive call. The clinically conservative
behaviour you'd want."

---

### Slide 18 — What drives the booster  (20 s)

*[slide 18: SHAP top-30 bar chart]*

"SHAP on the canonical 4-class model. **DITTO leads**, MetaRNN
second, then a long tail dominated by **population allele
frequencies**. The model is using both predictor scores and
population data. Half of the top 30 are predictor scores; roughly
the other half are AF columns."

---

### Slide 19 — Predictor correlation  (25 s)

*[slide 19: correlation heatmap]*

"The structural finding behind everything that follows.

Pearson correlation among predictor scores plus gnomAD AF. The
**red cluster top-left** is the meta-classifier predictors —
correlations above 0.5, mostly above 0.7. All rotated versions of
each other, trained on overlapping ClinVar data.

**gnomAD AF** sits in its own corner. **Anti-correlated** with the
predictor scores.

That picture drives the ablation findings."

---

### Slide 20 — Useful vs. waste  (30 s)

*[slide 20: 12-group ablation table]*

"Of the 208 features, which actually matter, which are deadweight?

Two-phase ablation across all four tasks. 12 functional feature
groups dropped one at a time.

Three findings. **population_af is indispensable** — removing it
costs **0.047** on 4-class. **functional** — fitcons and ncER,
just 4 features — explodes to 0.047 on the 3-class task.

And **eight of twelve groups are waste at the group level**
— greyed rows. They cover 97 of 208 features, almost half the
schema, and individually none of them moves macro-F1 by more
than the 0.005 noise floor."

---

### Slide 21 — Per-feature permutation  (30 s)

*[slide 21: 0.133 + 0.048 callouts]*

"But the group-level reading is misleading. Per-feature
permutation importance inside the waste groups:

**DITTO score alone — 0.133 macro-F1 weight, individually.**
MetaRNN, 0.048. Both highly predictive features, both sitting in
groups we just called waste.

The resolution is exactly what the correlation heatmap predicted —
**inter-predictor redundancy**. Remove DITTO and MetaRNN
substitutes; remove MetaRNN and DITTO compensates. Group LOO
measures **replaceability**, not informativeness."

---

### Slide 22 — Lean B  (20 s)

*[slide 22: baseline vs Lean B chart]*

"**Lean B** drops the 8 waste groups but keeps DITTO and MetaRNN —
the per-feature winners. 113 features kept out of 208,
**45% reduction**.

Across all four tasks: Lean B loses at most **0.012 macro-F1**.
On the 2-class task, the loss is **0.001** — within noise.
Half the schema is dispensable with no real cost. And several
models — KNN, NearestCentroid, CosineSimilarity — actually
**improve** on the lean subset."

---

### Slide 23 — Future directions  (20 s)

*[slide 23: 5 bullet points]*

"For future directions: **cross-database validation** on HGMD or
a held-out ClinVar snapshot, a **true homology-based BLAST** on
UniRef, **probability calibration** for clinical risk scoring,
**gene-stratified VUS-as-class** as a robustness check, and
**variant-level error analysis** on the Pathogenic-versus-
Likely-pathogenic confusion region."

---

### Slide 24 — Thank you  (5 s)

*[slide 24: Thank you / Questions?]*

"Thank you. We're happy to take questions."

*[end of script — open for Q&A]*

---

## Total word count + pacing check

| Speaker | Slides | Approx words | Target time | Words/min |
|---|---:|---:|---:|---:|
| A | 1–13 | ~770 | 5:00 | 154 |
| B | 14–24 | ~700 | 5:00 | 140 |
| **Combined** | **24** | **~1470** | **10:00** | **147** |

Slightly under 150 wpm — leaves buffer for slow delivery, beats, and
slide-to-slide transitions.

## Delivery notes (v2)

1. **Memorise four headlines — one per task:**
   `0.795 (4c) / 0.987 (2c) / 0.946 (3c) / 0.798 (5c).`
   Everything else can be paraphrased from the slide.
2. **The three big punchlines, in order:**
   - Slide 14: *"GCK 100%, PAH 99% — clinically sensible per-gene priors."*
   - Slide 16: *"VUS is one of the cleanest classes to recognise."*
   - Slide 21: *"Group LOO measures replaceability, not informativeness."*
3. **The leaderboard cluster (slides 8–11):** treat as one continuous
   beat. Each new slide reveals a different task's per-model
   structure; don't restart the framing on each. The transition lines
   are: *"Zooming in on 2-class..."* → *"For 3-class with VUS..."* →
   *"And on 5-class..."*. Each slide should take **only 20–25
   seconds** — let the chart do the work.
4. **At the handoff (end of slide 13):** use the partner's name, eye
   contact. Don't say "now my friend will take over."
5. **Chart-only slides** (4, 5, 6, 9, 10, 11, 12, 14, 17, 18, 19)
   should be short — 15–25 s each. Don't read every number;
   point at one or two.
6. **Text-heavy slides** (7 methods, 20 ablation table, 21
   permutation, 23 future) need 25–35 s. Don't rush.
7. **If you go over on Speaker A:** drop the per-class explanation
   on slide 12 (the chart speaks). On slide 13 you can compress to
   *"gene-CV costs 2.7 points, raw-only 7.4, 2-class is insensitive
   to everything — handoff."* If Speaker B goes over, compress
   slide 23 to a single sentence.

## What changed vs. v1

- **Three new headline slides** (9, 10, 11) covering 2-class, 3-class,
  and 5-class leaderboards. Each has a fresh talk line of ~20–25
  seconds.
- **Handoff moved** from after slide 10 to after slide 13 (still
  "after stress tests" in narrative terms — same logical point,
  different slide number).
- **Slide numbers 9–21 in v1 are now 12–24 in v2.** Comment numbers
  in delivery notes updated accordingly.
- **Word budget rebalanced:** Speaker A absorbs the three new slides
  in their 5-minute slot (which works because the new slides are
  chart-only and fast to deliver); Speaker B remains 11 slides.
