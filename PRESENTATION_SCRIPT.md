# Presentation Script — MissVARPath
**Total runtime target: 10 minutes (5 + 5).**
21 slides, ~28 s/slide average. Speaking rate assumed: ~150 wpm.
Slide cues marked `[slide N]`. Stage directions in *italics*.
Pause beats marked `…` — short pause; `………` — bigger pause.

---

## Speaker A — Setup, baselines, stress tests  (slides 1–10, ≈ 5 minutes)

### Slide 1 — Title  (10 s)

*[slide 1: title, names, course code, university]*

"Good afternoon. We're presenting **MissVARPath**, our project on
missense variant pathogenicity prediction with a focus on Variants of
Uncertain Significance — VUS. I'll cover the setup, data, methods, and
the canonical results, then **[name]** will take you through the VUS
deployment, the ablation study, and our final ablation-optimal model."

---

### Slide 2 — Three questions  (50 s)

*[slide 2: the three numbered questions]*

"In clinical genomics, every missense variant goes through a battery of
in-silico predictors — **PolyPhen-2, SIFT, REVEL, CADD, AlphaMissense,
DITTO**. Each predictor disagrees with the others, the predictors have
**correlated training data**, and most of them were themselves trained
with at least partial overlap on **ClinVar** — the same database we use
to label our targets.

So the three questions our project asks are:
**One**, can a meta-classifier do better than any single predictor by
combining them?
**Two**, how much of any improvement is data-circularity — predictors
trained on ClinVar, evaluated on ClinVar?
**Three**, can we predict pathogenicity from raw data alone — without
using any pre-existing predictor's score?"

---

### Slide 3 — Data  (30 s)

*[slide 3: big numbers + 4 task definitions]*

"Our dataset is **21,872 missense variants** from ClinVar, annotated
through OpenCRAVAT's 158-annotator pipeline.

We work with **four task definitions**: the original 4-class label space,
a 2-class collapse, and then two extensions we'll get to later where VUS
is added as its own class — 3-class and 5-class.

The raw schema is **777 columns**. After preprocessing we land at **208
numerical features**."

---

### Slide 4 — Label balance  (20 s)

*[slide 4: label distribution bars]*

"A quick point on the data: the corpus is **perfectly balanced** —
**5,468 rows per ClinVar class** in the 4-class space, and 10,936 in
each side of the 2-class collapse.

Any class-imbalance effect is gone by construction. Errors we report
reflect feature signal — not prevalence."

---

### Slide 5 — Missingness  (25 s)

*[slide 5: missingness histogram]*

"This is the missingness landscape across the raw 777 columns. Most
columns are densely populated — population-frequency columns sit at
zero percent missing. The long tail on the right is what gets dropped
in preprocessing: any column with **more than 50% missing**.

134 columns drop out that way, plus 98 identifier and free-text columns.
208 numerical features survive."

---

### Slide 6 — Predictor scores separate classes  (30 s)

*[slide 6: VEP boxplots by class]*

"Before we benchmark, a sanity check — do the predictor scores actually
separate the ClinVar classes?

These are boxplots of the headline predictor scores stratified by
4-class label. **DITTO, MetaRNN, REVEL, AlphaMissense** all separate
cleanly from Benign through Pathogenic. **gnomAD AF** runs the other
way — high frequency means benign. That's the ACMG BS1 rule encoded
directly in the data.

The signal is there. Now we need a model that can combine it."

---

### Slide 7 — Methods  (40 s)

*[slide 7: model suite + 5 regimes]*

"We train an **11-model suite** — nine classical and linear methods
(KNN, NearestCentroid, our custom CosineSimilarity classifier,
DecisionTree, LDA, QDA, LinearSVC, RidgeClassifier, SGDClassifier),
plus two ensembles (AdaBoost and HistGradientBoosting).

Every model runs under the **same protocol** — stratified 80/20 holdout
plus 5-fold CV — with `RANDOM_STATE = 42` for full reproducibility.

We evaluate the tuned suite under **five regimes**: canonical, an
augmented variant that adds k-mer and BLAST locus-neighbour columns,
gene-stratified CV where no gene crosses train and test, a no-VEP
ablation that drops every predictor and conservation score, and a
raw-only ablation that keeps only population frequency, position, and
sequence composition."

---

### Slide 8 — Headline numbers  (50 s)

*[slide 8: 0.795 / 0.987 + leaderboard bar chart]*

"Our headline numbers, with `HistGradientBoosting` as the consistent
top model: macro-F1 of **0.795** on the 4-class task, and **0.987** on
the 2-class task.

Below the booster, LinearSVC and DecisionTree trail by about 4
percentage points on the 4-class task. The linear family — SGD, LDA,
Ridge — sits in the low 0.70s. The centroid and KNN families are
around 0.55 to 0.66.

What's interesting isn't the average — it's the per-class structure."

---

### Slide 9 — Where the error lives  (25 s)

*[slide 9: 4-class confusion matrix]*

"This is the held-out 4-class confusion matrix, row-normalized.

**Benign at 90%, Likely-benign at 96%** — both diagonals very high.
Then look at the bottom-right corner: **Likely-pathogenic at 68%,
Pathogenic at 64%** — with about 30% of each one bleeding into the
other. The residual difficulty in the 4-class task is one specific
boundary: distinguishing Pathogenic from Likely-pathogenic."

---

### Slide 10 — Stress tests + handoff  (30 s)

*[slide 10: anchor table across the 5 regimes]*

"To bound how much of the headline is data-circularity, we ran the suite
under four stricter regimes.

**Gene-stratified CV** — no gene in both train and test — costs
**2.7 points** on 4-class, essentially nothing on 2-class.
**No-VEP** — drop every learned predictor — costs only 2.2 points.
**Raw-only** — predict from data, not from predictors — costs 7.4
points but the booster is still well above chance.

The 2-class task is **essentially insensitive** to every stress. That's
the operationally-deployable headline.

………

**[name]**, take it away."

*[hand-off to Speaker B]*

---
---

## Speaker B — VUS, ablation, optimal model, future  (slides 11–21, ≈ 5 minutes)

### Slide 11 — Real VUS deployment  (45 s)

*[slide 11: per-gene VUS prediction chart]*

"Thanks. So far we've evaluated on labelled ClinVar variants. But the
whole point of an in-silico classifier is to score **unclassified**
variants — VUS — that nobody has labelled yet.

We curated a **5,468-row VUS pro-set** from ClinVar — every 3-star
expert-reviewed VUS plus a random sample of 2-star VUS — annotated
through the same OpenCRAVAT pipeline.

This chart is the most clinically interesting result we have. It shows
the predicted-class breakdown for the top-20 most-frequent genes in the
pro-set. The model recovers **clinically sensible per-gene priors**
that it could only have learned indirectly from gene-correlated
features — **GCK at 100%** pathogenic-side, **PAH at 99%**, **MYH7 at
90%**, **LDLR at 84%**. Every one of these matches the published
clinical literature."

---

### Slide 12 — The four numbers  (30 s)

*[slide 12: 2x2 grid — 62/38, 80%, 87%, 95.4%]*

"Four numbers from that deployment.

**62/38** — the model splits the uncertain pool 62 percent
pathogenic-side, 38 percent benign-side on the 2-class head.
**80%** — eighty percent of predictions have max probability above
0.9; the model is confident on most.
**87%** — on the 4-class head, 87% of predictions hedge into the
Likely-star categories. Only 13% get escalated to a definitive call.
**95.4%** — when we collapse the 4-class predictions to
pathogenic-side / benign-side, the two heads — trained
independently — agree on 95.4 percent of the variants."

---

### Slide 13 — VUS as a class  (35 s)

*[slide 13: comparison + per-class F1 bar chart]*

"The next question we asked: can the model learn to **recognise** VUS
as a class — not just score them.

We added the strict-VUS subset of the pro-set as a labelled class. Two
new task definitions: 3-class — Benign-side, Pathogenic-side, VUS —
and 5-class with VUS as a fifth label.

The 3-class macro-F1 lands at **0.946**, costing about 0.04 against the
2-class baseline. But the striking result is 5-class: **0.798**, which
is actually **higher than the no-VUS 4-class baseline of 0.795**.
Adding VUS as a fifth class doesn't hurt — it slightly improves the
metric, and MCC improves materially.

Per-class F1 in the chart: VUS reaches **0.901** — between Benign and
Likely-benign in difficulty. **One of the cleanest classes to
recognise.**"

---

### Slide 14 — 5-class confusion matrix  (25 s)

*[slide 14: 5-class confusion matrix]*

"This is the 5-class held-out confusion matrix. The VUS row, bottom
left, reaches **0.94 diagonal** — about as clean as Benign and
Likely-benign.

When the model does miss on a VUS variant — about 6 percent of them —
it hedges almost entirely into the **Likely-star** categories. Only
**1 percent** get escalated to a definitive Pathogenic or Benign call.
That's the clinically conservative behaviour you'd want."

---

### Slide 15 — What drives the booster  (25 s)

*[slide 15: SHAP top-30 bar chart]*

"To understand what the model is actually using, we ran SHAP on the
canonical 4-class HistGradientBoosting.

**DITTO leads**, MetaRNN second, then a long tail dominated by
**population allele frequencies** — AllOfUs and gnomAD ancestry
breakdowns. The model is using both predictor scores and population
frequency data. Half of the top 30 features are predictor scores;
roughly the other half are AF columns."

---

### Slide 16 — Predictor correlation  (30 s)

*[slide 16: correlation heatmap]*

"Now here's the structural finding behind everything that follows.

This is the Pearson correlation among the headline predictor scores
plus gnomAD allele frequency.

The **top-left red cluster** is the meta-classifier predictors —
correlations above 0.5, mostly above 0.7. They're all rotated versions
of each other, because they were all trained on overlapping ClinVar
data and they encode broadly the same biological signal.

**gnomAD AF** — second column from the right — sits in its own corner.
**Anti-correlated** with the predictor scores.

That structural picture is going to drive the ablation findings."

---

### Slide 17 — Useful vs. waste  (30 s)

*[slide 17: 12-group ablation table]*

"With the model settled, we asked: of the 208 features, **which
actually matter, and which are deadweight**?

We ran a two-phase ablation across all four tasks, full 10-model suite,
12 functional feature groups dropped one at a time.

Three findings on this table. **population_af is indispensable** —
removing its 47 columns costs **0.047 macro-F1** on the 4-class task.
**functional** — fitcons and ncER, just 4 features — is invisible on
the canonical tasks but explodes to 0.047 on the 3-class task.

And **eight of twelve groups are waste at the group level** — the
greyed rows. They cover 97 of 208 features — about 47% of the schema —
and individually none of them moves macro-F1 by more than the 0.005
noise floor on any task."

---

### Slide 18 — Per-feature permutation  (35 s)

*[slide 18: 0.133 + 0.048 callouts]*

"But that group-level reading is misleading. We also ran per-feature
**permutation importance** inside the waste groups.

**DITTO score alone — 0.133 macro-F1 weight, individually.**
MetaRNN, 0.048. Both highly predictive features, both sitting in
groups we'd just called waste.

The resolution is straightforward — and it's exactly what the
correlation heatmap predicted. **Inter-predictor redundancy.** Remove
DITTO and MetaRNN substitutes. Remove MetaRNN and DITTO compensates.
Group leave-one-out measures **replaceability**, not informativeness.

The only way to make the redundancy buffer collapse is to remove
several correlated predictors together — which is what Lean B does."

---

### Slide 19 — Lean B  (25 s)

*[slide 19: baseline vs Lean B chart]*

"**Lean B** drops the 8 waste groups but keeps DITTO and MetaRNN — the
two per-feature winners. 113 features kept out of 208, **45% reduction**.

Across all four tasks: Lean B loses at most **0.012 macro-F1**.
On the 2-class task, the loss is **0.001** — within run-to-run noise.

So half the schema is dispensable for the headline model, with no real
cost. And several models — KNN, NearestCentroid, CosineSimilarity —
actually **improve** on the lean subset because they were sensitive to
the noise in the dropped columns."

---

### Slide 20 — Future directions  (20 s)

*[slide 20: 6 bullet points]*

"For future directions, the most important step is **cross-database
validation** on HGMD or a held-out ClinVar snapshot. After that: a true
homology-based BLAST feature on UniRef, finishing the AdaBoost tuning
grid, probability calibration so the outputs are usable as risk scores,
gene-stratified VUS-as-class as a robustness check, and variant-level
error analysis on the Pathogenic-versus-Likely-pathogenic confusion
region."

---

### Slide 21 — Thank you  (5 s)

*[slide 21: Thank you / Questions?]*

"Thank you. We're happy to take questions."

*[end of script — open for Q&A]*

---

## Total word count + pacing check

| Speaker | Slides | Approx words | Target time | Words/min |
|---|---:|---:|---:|---:|
| A | 1–10 | ~750 | 5:00 | 150 |
| B | 11–21 | ~680 | 5:00 | 136 |
| **Combined** | **21** | **~1430** | **10:00** | **143** |

Roughly 30 words per slide on average. Buffer for slow delivery and
slide-to-slide transitions built in.

## Delivery notes

1. **Memorise the four headlines:** `0.795 / 0.987 / 0.946 / 0.798`.
   Everything else can be paraphrased from the slide.
2. **The three big punchlines, in order:**
   - Slide 11: "*GCK 100%, PAH 99% — clinically sensible per-gene priors.*"
   - Slide 13: "*VUS is one of the cleanest classes to recognise.*"
   - Slide 18: "*Group LOO measures replaceability, not informativeness.*"
3. **At the handoff (slide 10)**, do not say "now my friend will take
   over" — use the name. Make eye contact with each other; signals
   confidence in joint ownership.
4. **Chart-only slides** (4, 5, 6, 9, 11, 14, 15, 16) can be short —
   the chart does the work. 15–20 seconds is enough. Don't read every
   number on screen; pick one or two to point at.
5. **Text-heavy slides** (7 methods, 17 ablation table, 18 permutation,
   20 future) need 30–40 seconds. Don't rush through them.
6. **If you go over on Speaker A**, drop the per-class breakdown on
   slide 9 (the chart speaks for itself). If Speaker B goes over,
   compress the bullet list on slide 20.
