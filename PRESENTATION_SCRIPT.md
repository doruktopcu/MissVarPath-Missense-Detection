# Presentation Script — MissVARPath / MVP
**Total runtime target: 10 minutes (5 + 5).**
Speaking rate assumed: ~150 wpm.
Slide cues marked `[slide N]`. Stage directions in *italics*.
Pause beats marked `…` — short pause; `………` — bigger pause.

---

## Speaker A — Setup, methods, and baselines  (5 minutes)

### Slide 1 — Title (15 s)

*[slide 1: title, names, course code, date]*

"Good afternoon. We're presenting **MissVARPath**, our project on missense
variant pathogenicity prediction with a focus on Variants of Uncertain
Significance — VUS. I'll cover the setup, data, methods, and canonical
results, then **[name]** will take you through the VUS deployment, the
ablation study, and our final ablation-optimal model."

---

### Slide 2 — The problem  (50 s)

*[slide 2: clinical workflow → classifier → output, with VUS callout]*

"In clinical genomics, every missense variant goes through a battery of
in-silico predictors — **PolyPhen-2, SIFT, REVEL, CADD, AlphaMissense,
DITTO** — and the results are combined with **population frequency** data
and ACMG/AMP rules to produce a clinical classification.

The catch is that **each predictor disagrees with the others**, the
predictors have **correlated training data**, and most of them were
themselves trained with at least partial overlap on **ClinVar** — which
is the same database we use to label our targets.

So the three questions our project asks are:
**One**, can a meta-classifier do better than any single predictor by
combining them?
**Two**, how much of any improvement is data-circularity, since the
predictors saw ClinVar during their own training?
**Three**, can we predict pathogenicity from raw data alone — population
frequencies, position, sequence composition — *without* using any
pre-existing predictor's score?"

---

### Slide 3 — Data  (45 s)

*[slide 3: dataset card — 21,872 rows, 5,468 per class, 777 → 208 cols]*

"Our dataset is **21,872 missense variants** from ClinVar, annotated
through OpenCRAVAT's 158-annotator pipeline. We use **two tasks**: the
original 4-class label — Benign, Likely-benign, Likely-pathogenic,
Pathogenic — and a 2-class collapse where the Likely-star labels merge
into the definitive ones.

A key design decision: the dataset is **perfectly balanced**, 5,468 rows
per class. Any class-imbalance effect is gone by construction, so
errors we see reflect feature signal — not prevalence.

The raw schema is **777 columns**. After preprocessing — dropping
identifier columns, label-leakage columns from the clinvar annotator,
free-text columns, and anything more than 50% missing — we land at
**208 numerical features**."

---

### Slide 4 — Methods  (50 s)

*[slide 4: model suite block + 5 evaluation regimes block]*

"We train an **11-model suite** spanning classical and ensemble methods:
KNN, NearestCentroid, a custom CosineSimilarity classifier, DecisionTree,
LDA, QDA, LinearSVC, RidgeClassifier, SGDClassifier, AdaBoost, and
HistGradientBoosting.

Every model runs under the **same protocol**: a stratified 80/20
held-out split plus 5-fold cross-validation on the training portion,
with `RANDOM_STATE = 42` for full reproducibility.

We tuned every non-AdaBoost model by per-grid 5-fold macro-F1 — AdaBoost
hit our compute budget at combo 23 of 27 and was excluded from tuning.

Then we evaluate the tuned suite under **five regimes**:
canonical k-fold; an augmented variant adding **k-mer and BLAST
locus-neighbour features**; **gene-stratified** cross-validation where
no gene appears in both train and test folds; a **no-VEP scenario** that
drops every learned predictor score; and a **raw-only** ablation that
keeps only population frequencies, position, and sequence composition."

---

### Slide 5 — Headline numbers  (60 s)

*[slide 5: leaderboard table or single bar chart — 4-class + 2-class tuned]*

"Our headline numbers, with `HistGradientBoosting` as the consistent top
model: macro-F1 of **0.7950** on the 4-class task, and **0.9872** on the
2-class task — accuracy and MCC are essentially identical on the binary
problem.

Below the booster, the linear models — LinearSVC and SGDClassifier —
trail by 4 to 7 percentage points on the 4-class task. The centroid and
KNN families sit at the bottom around 0.55 macro-F1 on 4-class.

The interesting structure shows up in **per-class** error: the 2-class
task is near-perfect because Benign-versus-Pathogenic is a clean split,
but on the 4-class task the residual difficulty is concentrated at two
boundaries — **Benign vs. Likely-benign**, and **Pathogenic vs.
Likely-pathogenic**. The per-class F1 for those middle classes sits
around 0.65, while definitive Benign hits 0.93."

---

### Slide 6 — Stress tests  (60 s)

*[slide 6: anchor table — macro-F1 across the 5 regimes]*

"To bound how much of that headline is data-circularity, we ran the
suite under four stricter regimes.

**Gene-stratified** CV — where no gene appears in both train and test —
costs us **2.7 percentage points** on 4-class — from 0.795 to 0.766 —
and **essentially zero** on 2-class. So there's a real
data-circularity contribution on the fine-grained task, but the binary
problem generalises cleanly across genes.

The **no-VEP scenario** drops every learned predictor and conservation
score — 130 columns out of the schema. HistGB still reaches **0.773**
on 4-class and **0.976** on 2-class. The graceful degradation here is
the headline finding for clinical deployability — even without any
meta-classifier output, the population-and-sequence-composition feature
set carries enough signal.

The **raw-only** ablation — keeping only population frequencies,
position, and k-mer composition — drops us to **0.721** on 4-class and
**0.934** on 2-class. The linear models collapse hardest in this
regime, losing 12 to 18 points; the booster is by far the most robust.

So the picture is: there's some data-circularity, the linear models are
piggy-backing on pre-trained predictor scores, but the booster — and
the binary task — survive even the strictest ablations.

………

That's the canonical setup and the stress tests. **[name]** will now
walk you through what we did beyond the benchmark."

*[hand-off to Speaker B]*

---
---

## Speaker B — VUS deployment, ablation, and final model  (5 minutes)

### Slide 7 — Real VUS deployment  (75 s)

*[slide 7: per-gene VUS prediction stacked bar chart]*

"Thanks. So far we've evaluated on labelled ClinVar variants. But the
whole point of an in-silico classifier is to score **uncertain**
variants — VUS — that nobody has classified yet.

We curated a **5,468-row VUS pro-set** from ClinVar — every 3-star
expert-reviewed VUS plus a random sample of 2-star VUS — and annotated
it through the same OpenCRAVAT pipeline. We applied the tuned
classifier:

On the 2-class head, the model splits the uncertain pool **62%
pathogenic-side, 38% benign-side**, with 80% of predictions above 0.9
probability — so the model has high confidence on most of them.

On the 4-class head, **87% of predictions hedge into the Likely-star
categories** — only 13% are escalated to a definitive Pathogenic or
Benign call. That's clinically conservative behaviour.

Cross-task consistency: when we collapse the 4-class predictions back
to pathogenic-side / benign-side, the two heads — trained
independently — agree on **95.4%** of the 5,468 variants.

The per-gene structure shown here is the key validation: the model
recovers clinically sensible per-gene priors that it could only have
learned *indirectly* from gene-correlated features. GCK at 100%
pathogenic-side, PAH at 99%, MYH7 at 90%, LDLR at 84% — every one of
these is consistent with the published clinical literature on those
gene families."

---

### Slide 8 — VUS-as-class study  (75 s)

*[slide 8: bar chart of 4 study macro-F1s + per-class F1 in 5-class]*

"That's *scoring* VUS with the canonical classifier. The next question
we asked is whether **VUS itself can be a labelled class** — can the
model learn to *recognise* uncertain variants as a class of their own.

We built two new task definitions. **3-class**: Benign-side,
Pathogenic-side, VUS. **5-class**: the original four ClinVar labels
plus VUS as a fifth.

The headline result is two-part. **One**: 3-class macro-F1 lands at
**0.946** — adding VUS as a third class to the binary task costs about
0.04 against the 2-class baseline. **Two** — and this is the striking
part: 5-class macro-F1 is **0.798**, **higher than the no-VUS 4-class
baseline of 0.795**. Adding VUS as a fifth class doesn't hurt the
overall metric — it slightly improves it, and MCC improves materially
too.

The reason is in the per-class breakdown: VUS reaches **F1 = 0.901** in
the 5-class study — sitting between Benign at 0.93 and Likely-benign at
0.89 in difficulty. **VUS is one of the cleanest classes to recognise**,
much easier than the Pathogenic-versus-Likely-pathogenic boundary that
has been the residual difficulty since the start.

And when the model does misclassify a VUS row — about 6% of them — it
hedges into a Likely-star category 4.7% of the time, and only escalates
to a definitive call in 1.6% of cases. Conservative behaviour, again."

---

### Slide 9 — Feature ablation: useful vs. waste  (75 s)

*[slide 9: the 12-group headline delta table, with 8 waste rows highlighted]*

"With the model architecture settled, we asked: of the **208 features**,
**which actually matter, and which are deadweight**?

We ran a two-phase ablation across all four tasks times the full
10-model suite. **Phase A** is group leave-one-out: 12 functional
feature groups — population AFs, conservation, DITTO, MetaRNN, REVEL,
CADD, AlphaMissense, BayesDel, chasmplus, the other-VEP basket,
positional features, and a small functional-genomics group — dropped
one at a time.

Three findings:

**One — `population_af` is indispensable.** Removing its 47 columns
costs **0.047 macro-F1** on the 4-class task — by far the biggest
single effect. This is the model encoding the ACMG BS1/BA1 rule — common
in healthy populations means probably benign — and no other feature
group can substitute.

**Two — eight of twelve groups are waste at the group level.** Dropping
DITTO, MetaRNN, BayesDel, REVEL, CADD, AlphaMissense, chasmplus's 68
columns, conservation's 17 columns — none of them moves macro-F1 by
more than the 0.005 noise floor on any task.

**Three** — and this is the methodologically important finding — *that
doesn't mean those features are uninformative*. **Phase B**, per-feature
permutation importance, shows that **DITTO alone carries 0.13 macro-F1
weight individually**, and MetaRNN carries 0.05. The reason removing
DITTO as a group is cheap is **inter-predictor redundancy** — every
learned predictor was trained on overlapping ClinVar data, so when one
is removed the others substitute. Group LOO measures replaceability,
not informativeness."

---

### Slide 10 — Lean B + Future directions  (45 s)

*[slide 10: Lean B macro-F1 vs baseline, 4 tasks bar chart + 3-tier feature classification]*

"That redundancy insight produces our ablation-optimal model — **Lean B**.

Lean B drops the 8 waste groups *except* keeps the two per-feature
winners — `ditto_score` and `metarnn_score`. The result: **113 features
out of 208**, a 45% reduction in the schema. And the headline cost is
at most **0.012 macro-F1** on any task — 0.005 on 4-class, 0.001 on
2-class. Several models — KNN, NearestCentroid, CosineSimilarity —
actually *improve* on the lean subset because they were sensitive to
the noise in the dropped columns.

So **half the schema is dispensable** for the headline model, with no
real cost.

………

For future directions: the most important step is **cross-database
validation** on HGMD or a held-out ClinVar snapshot; a true homology-based
BLAST feature on UniRef; finishing the AdaBoost tuning grid; and
probability calibration so the model's outputs are usable as risk scores
in a clinical pipeline.

Thank you. We're happy to take questions."

*[end of script — open for Q&A]*

---

## Total word count + pacing check

| Speaker | Approximate words | Target time | Words/min |
|---|---:|---:|---:|
| A (slides 1–6) | ~720 | 5:00 | 144 |
| B (slides 7–10) | ~660 | 5:00 | 132 |
| **Combined**   | **~1380** | **10:00** | **138** |

Slightly under 150 wpm — leaves buffer for slow delivery and beats.

## Delivery notes

1. **Numbers are the spine.** Memorise the four headlines: `0.795 / 0.987 /
   0.946 / 0.798`. Everything else can be paraphrased from the slide.
2. **The two big punchlines:**
   - Slide 8: "*VUS is one of the cleanest classes to recognise.*"
   - Slide 9: "*Group LOO measures replaceability, not informativeness.*"
3. **The clinically resonant moment** is the per-gene VUS chart (slide 7) —
   GCK 100% / PAH 99% / MYH7 90%. Land that with conviction.
4. **At the handoff**, do not say "now my friend will take over" —
   use the name. Make eye contact with each other; signals confidence
   in the joint work.
5. If you go over on Speaker A, cut the per-class explanation in slide 5;
   the slide will say it for you. If Speaker B goes over, drop the
   "graceful degradation" line on slide 6 and skip directly to the
   anchor table.
