"""Build the 21-slide presentation deck — minimal black-and-white theme.

Pulls headline numbers from the on-disk CSVs (leaderboards, grand_summary,
ablation tables) so the deck stays in sync with the report. Charts are
native editable PowerPoint charts where possible; the EDA, SHAP, and
confusion-matrix figures are embedded as PNGs from outputs/.

Run: python -m scripts.build_presentation
Output: MissVarPath_Presentation.pptx
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

# --- Palette: pure monochrome ---
BLACK    = RGBColor(0x00, 0x00, 0x00)
DARK     = RGBColor(0x20, 0x20, 0x20)
GRAPHITE = RGBColor(0x40, 0x40, 0x40)
GREY     = RGBColor(0x70, 0x70, 0x70)
LIGHT    = RGBColor(0xBF, 0xBF, 0xBF)
PALE     = RGBColor(0xE8, 0xE8, 0xE8)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)

# Typography
TITLE_FONT = "Georgia"
BODY_FONT = "Calibri"

# Slide dimensions (16:9 widescreen, 13.33" × 7.5")
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.6)

TOTAL = 21


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _set_run(run, text, *, font=BODY_FONT, size=18, bold=False, italic=False, color=BLACK):
    run.text = text
    f = run.font
    f.name = font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color


def add_textbox(slide, x, y, w, h, *, text="", font=BODY_FONT, size=18, bold=False,
                italic=False, color=BLACK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
                line_spacing=None, margin=Inches(0)):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = margin
    tf.margin_top = tf.margin_bottom = margin
    p = tf.paragraphs[0]
    p.alignment = align
    if line_spacing is not None:
        p.line_spacing = line_spacing
    _set_run(p.add_run(), text, font=font, size=size, bold=bold, italic=italic, color=color)
    return box, tf


def add_multiline(slide, x, y, w, h, lines, *, line_spacing=1.2, anchor=MSO_ANCHOR.TOP,
                  align=PP_ALIGN.LEFT, margin=Inches(0)):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = margin
    tf.margin_top = tf.margin_bottom = margin
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = ln.get("align", align)
        p.line_spacing = ln.get("line_spacing", line_spacing)
        if ln.get("space_after"):
            p.space_after = Pt(ln["space_after"])
        run = p.add_run()
        _set_run(run, ln["text"],
                 font=ln.get("font", BODY_FONT),
                 size=ln.get("size", 18),
                 bold=ln.get("bold", False),
                 italic=ln.get("italic", False),
                 color=ln.get("color", BLACK))
    return box, tf


def add_slide_title(slide, text):
    return add_textbox(slide, MARGIN, Inches(0.4), Inches(12.1), Inches(1.0),
                       text=text, font=TITLE_FONT, size=32, bold=True, color=BLACK,
                       line_spacing=1.05)


def add_slide_number(slide, n, total=TOTAL):
    add_textbox(slide, Inches(12.4), Inches(7.1), Inches(0.8), Inches(0.3),
                text=f"{n} / {total}", font=BODY_FONT, size=11, color=GREY,
                align=PP_ALIGN.RIGHT)


def add_caption(slide, x, y, w, h, text, *, italic=True, color=GREY, size=13):
    return add_textbox(slide, x, y, w, h, text=text, font=BODY_FONT,
                       size=size, italic=italic, color=color, line_spacing=1.3)


def fit_image(slide, path, x, y, max_w, max_h):
    """Insert an image fitting within (max_w, max_h), preserving aspect ratio,
    centered horizontally within the box."""
    from PIL import Image
    im = Image.open(path)
    iw, ih = im.size
    # Scale to fit
    sx = max_w / Inches(iw / 96).emu
    sy = max_h / Inches(ih / 96).emu
    # Use the image's natural pixel→inch via 96 dpi, scale to bounds
    target_w = max_w
    target_h = int(max_w * ih / iw)
    if target_h > max_h:
        target_h = max_h
        target_w = int(max_h * iw / ih)
    # Center within the bounding box
    x_centered = x + (max_w - target_w) // 2
    return slide.shapes.add_picture(str(path), x_centered, y, width=target_w, height=target_h)


def style_chart_minimal(chart, *, show_value=False, value_format="0.000",
                        label_size=11, axis_size=12):
    """Mono palette + larger fonts."""
    chart.has_title = False
    chart.has_legend = False
    plot = chart.plots[0]
    for i, series in enumerate(plot.series):
        fill = series.format.fill
        fill.solid()
        fill.fore_color.rgb = DARK if i == 0 else GREY
        line = series.format.line
        line.color.rgb = DARK
        line.width = Pt(0.75)
    plot.has_data_labels = show_value
    if show_value:
        dl = plot.data_labels
        dl.font.name = BODY_FONT
        dl.font.size = Pt(label_size)
        dl.font.color.rgb = DARK
        dl.number_format = value_format
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
    cat = chart.category_axis
    val = chart.value_axis
    for ax in (cat, val):
        ax.tick_labels.font.name = BODY_FONT
        ax.tick_labels.font.size = Pt(axis_size)
        ax.tick_labels.font.color.rgb = GRAPHITE
    val.format.line.color.rgb = LIGHT
    cat.format.line.color.rgb = LIGHT
    try:
        cat.major_gridlines.format.line.fill.background()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

REPORTS = Path("outputs/reports")


def lb(setting): return pd.read_csv(REPORTS / setting / "leaderboard.csv")


# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------

def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank = prs.slide_layouts[6]

    # =========================================================================
    # Slide 1 — Title
    # =========================================================================
    s = prs.slides.add_slide(blank)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(0.0),
                             Inches(0.03), SLIDE_H)
    bar.fill.solid(); bar.fill.fore_color.rgb = BLACK
    bar.line.fill.background()

    add_textbox(s, Inches(1.2), Inches(2.3), Inches(11), Inches(0.4),
                text="CMP682 — Final Project",
                font=BODY_FONT, size=18, color=GREY)
    add_textbox(s, Inches(1.2), Inches(2.7), Inches(11), Inches(1.3),
                text="MissVARPath",
                font=TITLE_FONT, size=72, bold=True, color=BLACK)
    add_textbox(s, Inches(1.2), Inches(4.0), Inches(11), Inches(0.7),
                text="Missense Variant Pathogenicity Prediction",
                font=TITLE_FONT, size=28, italic=True, color=DARK)
    add_textbox(s, Inches(1.2), Inches(5.7), Inches(11), Inches(0.5),
                text="Doruk Topcu  ·  Alihan Sağöz",
                font=BODY_FONT, size=20, color=BLACK)
    add_textbox(s, Inches(1.2), Inches(6.2), Inches(11), Inches(0.4),
                text="Hacettepe University · 2026",
                font=BODY_FONT, size=14, color=GREY)

    # =========================================================================
    # Slide 2 — Three questions
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "Three questions")

    intro = ("Every missense variant goes through a battery of in-silico predictors — "
             "PolyPhen-2, SIFT, REVEL, CADD, AlphaMissense, DITTO. Each disagrees with "
             "the others; most were trained on overlapping ClinVar data — the same database "
             "we use to label our targets.")
    add_textbox(s, MARGIN, Inches(1.4), Inches(12.1), Inches(1.2),
                text=intro, font=BODY_FONT, size=17, color=GRAPHITE,
                line_spacing=1.35)

    questions = [
        ("1.", "Can a meta-classifier do better than any single predictor by combining them?"),
        ("2.", "How much of any improvement is data-circularity — predictors trained on ClinVar, evaluated on ClinVar?"),
        ("3.", "Can we predict pathogenicity from raw data alone — without using any pre-existing predictor's score?"),
    ]
    y = Inches(3.0)
    for num, q in questions:
        add_textbox(s, MARGIN, y, Inches(0.9), Inches(0.9),
                    text=num, font=TITLE_FONT, size=48, bold=True, color=BLACK)
        add_textbox(s, Inches(1.6), y + Inches(0.2), Inches(11.3), Inches(0.9),
                    text=q, font=BODY_FONT, size=19, color=DARK, line_spacing=1.3)
        y = y + Inches(1.3)

    add_slide_number(s, 2)

    # =========================================================================
    # Slide 3 — Data overview (big numbers)
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "Data")

    add_multiline(s, MARGIN, Inches(1.3), Inches(6.5), Inches(5.8), [
        {"text": "21,872", "size": 68, "bold": True, "font": TITLE_FONT, "space_after": 4},
        {"text": "ClinVar / OpenCRAVAT missense variants",
         "size": 15, "color": GREY, "space_after": 22},

        {"text": "5,468  ×  4", "size": 42, "bold": True, "font": TITLE_FONT, "space_after": 4},
        {"text": "perfectly balanced 4-class — Benign / Likely-benign / Likely-pathogenic / Pathogenic",
         "size": 15, "color": GREY, "space_after": 22},

        {"text": "777  →  208", "size": 42, "bold": True, "font": TITLE_FONT, "space_after": 4},
        {"text": "raw schema  →  numerical features after preprocessing",
         "size": 15, "color": GREY, "space_after": 22},

        {"text": "5,428", "size": 42, "bold": True, "font": TITLE_FONT, "space_after": 4},
        {"text": "strict-VUS rows in the curated pro-set",
         "size": 15, "color": GREY},
    ], line_spacing=1.15)

    # Right column — task definitions
    right_x = Inches(7.5)
    add_caption(s, right_x, Inches(1.3), Inches(5.4), Inches(0.4),
                "Four task definitions used in the project", size=14)

    add_multiline(s, right_x, Inches(1.8), Inches(5.4), Inches(5.2), [
        {"text": "4-class", "size": 22, "bold": True, "font": TITLE_FONT, "space_after": 4},
        {"text": "original ClinVar label space",
         "size": 14, "color": GREY, "space_after": 22},

        {"text": "2-class", "size": 22, "bold": True, "font": TITLE_FONT, "space_after": 4},
        {"text": "Likely-* collapsed into definitive labels → 10,936 / 10,936",
         "size": 14, "color": GREY, "space_after": 22},

        {"text": "3-class (with VUS)", "size": 22, "bold": True, "font": TITLE_FONT, "space_after": 4},
        {"text": "Benign-side / Pathogenic-side / VUS",
         "size": 14, "color": GREY, "space_after": 22},

        {"text": "5-class (with VUS)", "size": 22, "bold": True, "font": TITLE_FONT, "space_after": 4},
        {"text": "all four ClinVar labels + VUS — essentially balanced",
         "size": 14, "color": GREY},
    ], line_spacing=1.15)

    add_slide_number(s, 3)

    # =========================================================================
    # Slide 4 — Label distribution (EDA chart)
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "Label balance — by design")
    add_caption(s, MARGIN, Inches(1.2), Inches(12.1), Inches(0.5),
                "5,468 rows per ClinVar class in the 4-class space; 10,936 / 10,936 in the 2-class collapse. "
                "Any class-imbalance effect is gone by construction.")
    fit_image(s, "outputs/eda/label_distribution.png",
              x=Inches(1.0), y=Inches(2.0), max_w=Inches(11.3), max_h=Inches(4.7))
    add_slide_number(s, 4)

    # =========================================================================
    # Slide 5 — Missingness landscape
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "Missingness across the 777-column raw schema")
    add_caption(s, MARGIN, Inches(1.2), Inches(12.1), Inches(0.5),
                "Per-column missing fraction. Everything above the 50% threshold is dropped during preprocessing — "
                "134 columns lost that way. 208 numerical features survive.")
    fit_image(s, "outputs/eda/missingness_histogram.png",
              x=Inches(0.8), y=Inches(2.0), max_w=Inches(11.7), max_h=Inches(4.6))
    add_slide_number(s, 5)

    # =========================================================================
    # Slide 6 — Key VEP scores by class
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "Do the predictor scores actually separate classes?")
    add_caption(s, MARGIN, Inches(1.2), Inches(12.1), Inches(0.6),
                "Boxplots of headline VEP scores stratified by the 4-class label. DITTO, MetaRNN, REVEL, AlphaMissense "
                "all separate cleanly; gnomAD AF runs the other way — high frequency means benign (the ACMG BS1 rule).")
    fit_image(s, "outputs/eda/key_vep_scores_by_class.png",
              x=Inches(0.5), y=Inches(2.1), max_w=Inches(12.3), max_h=Inches(4.6))
    add_slide_number(s, 6)

    # =========================================================================
    # Slide 7 — Methods
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "Methods")

    add_caption(s, MARGIN, Inches(1.2), Inches(6), Inches(0.4),
                "Model suite — 11 estimators", size=15)

    classical = ["KNN", "NearestCentroid", "CosineSimilarity (custom)",
                 "DecisionTree", "LDA", "QDA", "LinearSVC", "RidgeClassifier", "SGDClassifier"]
    ensemble = ["AdaBoost", "HistGradientBoosting"]

    add_multiline(s, MARGIN, Inches(1.7), Inches(6), Inches(4.5),
                  [{"text": "Classical / linear", "size": 18, "bold": True,
                    "color": DARK, "space_after": 6}] +
                  [{"text": m, "size": 15, "color": GRAPHITE, "space_after": 2}
                   for m in classical] +
                  [{"text": " ", "size": 15, "space_after": 8}] +
                  [{"text": "Ensembles", "size": 18, "bold": True,
                    "color": DARK, "space_after": 6}] +
                  [{"text": m, "size": 15, "color": GRAPHITE, "space_after": 2}
                   for m in ensemble],
                  line_spacing=1.15)

    add_caption(s, Inches(7.2), Inches(1.2), Inches(6), Inches(0.4),
                "Evaluation regimes — 5 cuts", size=15)

    regimes = [
        ("Canonical", "stratified 80/20 holdout + 5-fold CV"),
        ("Augmented", "+ 196 k-mer + 10 BLAST locus-neighbour features"),
        ("Gene-stratified", "no gene appears in both train and test folds"),
        ("No-VEP / VUS", "drop every learned-predictor + conservation score"),
        ("Raw-only", "keep only population AF, position, sequence composition"),
    ]
    y = Inches(1.8)
    for name, desc in regimes:
        add_textbox(s, Inches(7.2), y, Inches(2.4), Inches(0.4),
                    text=name, font=BODY_FONT, size=16, bold=True, color=DARK)
        add_textbox(s, Inches(9.7), y, Inches(3.2), Inches(0.9),
                    text=desc, font=BODY_FONT, size=14, color=GRAPHITE,
                    line_spacing=1.25)
        y = y + Inches(0.7)

    add_caption(s, MARGIN, Inches(6.55), Inches(12.1), Inches(0.5),
                "RANDOM_STATE = 42 throughout. Per-model grid-search tuning by 5-fold CV macro-F1; "
                "AdaBoost grid stopped at combo 23/27 (compute budget).", size=13)

    add_slide_number(s, 7)

    # =========================================================================
    # Slide 8 — Headline numbers + leaderboard
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "Headline numbers")

    callout_y = Inches(1.3)
    add_textbox(s, MARGIN, callout_y, Inches(5.5), Inches(0.3),
                text="4-class macro-F1", font=BODY_FONT, size=15,
                italic=True, color=GREY)
    add_textbox(s, MARGIN, callout_y + Inches(0.4), Inches(5.5), Inches(1.7),
                text="0.795", font=TITLE_FONT, size=80, bold=True, color=BLACK)
    add_textbox(s, MARGIN, callout_y + Inches(2.0), Inches(5.5), Inches(0.4),
                text="HistGradientBoosting · 21,872 variants · 208 features",
                font=BODY_FONT, size=13, color=GREY)

    add_textbox(s, Inches(7.0), callout_y, Inches(5.5), Inches(0.3),
                text="2-class macro-F1", font=BODY_FONT, size=15,
                italic=True, color=GREY)
    add_textbox(s, Inches(7.0), callout_y + Inches(0.4), Inches(5.5), Inches(1.7),
                text="0.987", font=TITLE_FONT, size=80, bold=True, color=BLACK)
    add_textbox(s, Inches(7.0), callout_y + Inches(2.0), Inches(5.5), Inches(0.4),
                text="HistGradientBoosting · 10,936 / 10,936 collapse",
                font=BODY_FONT, size=13, color=GREY)

    # Leaderboard bar chart
    df4 = lb("4class_final_no_adaboost")
    df4 = df4.sort_values("holdout_macro_f1", ascending=False).head(7)
    chart_data = CategoryChartData()
    chart_data.categories = df4["model"].tolist()[::-1]
    chart_data.add_series("4-class macro-F1", df4["holdout_macro_f1"].tolist()[::-1])
    chart = s.shapes.add_chart(
        XL_CHART_TYPE.BAR_CLUSTERED,
        MARGIN, Inches(3.8), Inches(12.1), Inches(3.1),
        chart_data,
    ).chart
    style_chart_minimal(chart, show_value=True, value_format="0.000",
                        label_size=12, axis_size=12)
    chart.value_axis.minimum_scale = 0.5
    chart.value_axis.maximum_scale = 0.85

    add_slide_number(s, 8)

    # =========================================================================
    # Slide 9 — 4-class confusion matrix
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "Where the error lives — 4-class confusion matrix")
    add_caption(s, MARGIN, Inches(1.2), Inches(12.1), Inches(0.5),
                "Held-out, row-normalized. Definitive Benign / Pathogenic call near-perfect (>0.93 / 0.79). "
                "Residual difficulty is the Pathogenic ↔ Likely-pathogenic boundary.")
    fit_image(s, "final_report/figures/histgb_4class_cm.png",
              x=Inches(2.5), y=Inches(2.0), max_w=Inches(8.3), max_h=Inches(5.0))
    add_slide_number(s, 9)

    # =========================================================================
    # Slide 10 — Stress tests (was slide 6)
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "Stress tests — bounding the data-circularity")

    anchor_rows = [
        ("Regime",                                   "4-class F1", "2-class F1"),
        ("Canonical tuned (headline)",               "0.795",      "0.987"),
        ("Augmented (k-mer + BLAST)",                "0.795",      "0.990"),
        ("Gene-stratified CV",                       "0.766",      "0.987"),
        ("No-VEP, k-fold",                           "0.773",      "0.976"),
        ("No-VEP, gene-stratified",                  "0.761",      "—"),
        ("Raw-only (no predictor, no conservation)", "0.721",      "0.934"),
        ("DITTO removed",                            "0.791",      "—"),
    ]
    rows, cols = len(anchor_rows), 3
    tbl = s.shapes.add_table(rows, cols,
                              MARGIN, Inches(1.4),
                              Inches(12.1), Inches(0.5 * rows)).table
    tbl.columns[0].width = Inches(7.5)
    tbl.columns[1].width = Inches(2.3)
    tbl.columns[2].width = Inches(2.3)
    for r, row in enumerate(anchor_rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            for run in p.runs:
                f = run.font
                f.name = BODY_FONT
                f.size = Pt(15)
                f.bold = (r == 0) or (r == 1)
                f.color.rgb = BLACK if r > 0 else GREY
            cell.margin_left = Inches(0.15)
            cell.margin_right = Inches(0.15)
            cell.margin_top = Inches(0.06)
            cell.margin_bottom = Inches(0.06)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PALE if r == 0 else WHITE

    add_caption(s, MARGIN, Inches(6.4), Inches(12.1), Inches(0.7),
                "Graceful degradation across regimes: HistGB loses 0.027 to gene-stratified CV, 0.022 to no-VEP, "
                "0.074 to raw-only. The 2-class task is essentially insensitive to all stresses.",
                color=DARK, size=14)
    add_slide_number(s, 10)

    # =========================================================================
    # Slide 11 — Real VUS deployment (per-gene chart)
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "Real VUS deployment — per-gene predictions")
    add_caption(s, MARGIN, Inches(1.2), Inches(12.1), Inches(0.5),
                "Top-20 most-frequent genes in the 5,468-variant pro-set, stacked by 4-class prediction. "
                "GCK 100%, PAH 99%, MYH7 90%, LDLR 84% pathogenic-side.")
    fit_image(s, "outputs/figures/vus_per_gene_predictions.png",
              x=Inches(0.6), y=Inches(2.0), max_w=Inches(12.1), max_h=Inches(4.8))
    add_slide_number(s, 11)

    # =========================================================================
    # Slide 12 — VUS deployment stats
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "VUS deployment — the four numbers")

    add_caption(s, MARGIN, Inches(1.2), Inches(12.1), Inches(0.4),
                "Applying the tuned HistGradientBoosting to 5,468 strict-VUS variants annotated through the same pipeline.")

    callouts = [
        ("62% / 38%",
         "pathogenic-side / benign-side split on 2-class head"),
        ("80%",
         "of predictions have max probability > 0.9"),
        ("87%",
         "of 4-class predictions hedge into Likely-* — only 13% definitive"),
        ("95.4%",
         "cross-task agreement (4-class collapsed vs. 2-class head)"),
    ]
    # 2x2 grid layout
    positions = [
        (Inches(0.8),  Inches(2.0)),
        (Inches(7.0),  Inches(2.0)),
        (Inches(0.8),  Inches(4.7)),
        (Inches(7.0),  Inches(4.7)),
    ]
    for (num, desc), (x, y) in zip(callouts, positions):
        add_textbox(s, x, y, Inches(5.7), Inches(1.1),
                    text=num, font=TITLE_FONT, size=64, bold=True, color=BLACK)
        add_textbox(s, x, y + Inches(1.2), Inches(5.7), Inches(0.9),
                    text=desc, font=BODY_FONT, size=16, color=GRAPHITE,
                    line_spacing=1.3)

    add_slide_number(s, 12)

    # =========================================================================
    # Slide 13 — VUS-as-class
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "VUS as a class")

    add_caption(s, MARGIN, Inches(1.2), Inches(6), Inches(0.4),
                "With VUS added as a labelled class", size=15)

    pairs = [
        ("2-class no-VUS",      "0.987"),
        ("3-class with VUS",    "0.946"),
        ("4-class no-VUS",      "0.795"),
        ("5-class with VUS",    "0.798"),
    ]
    y = Inches(1.8)
    for name, val in pairs:
        add_textbox(s, MARGIN, y, Inches(4.5), Inches(0.5),
                    text=name, font=BODY_FONT, size=18, color=GRAPHITE)
        add_textbox(s, Inches(5.0), y, Inches(1.5), Inches(0.5),
                    text=val, font=TITLE_FONT, size=24, bold=True, color=BLACK,
                    align=PP_ALIGN.RIGHT)
        y = y + Inches(0.65)

    add_caption(s, MARGIN, Inches(4.6), Inches(6), Inches(1.4),
                "Adding VUS as a 5th class does NOT hurt overall macro-F1 — it slightly improves it. "
                "MCC improves by +0.022.", color=DARK, size=15)

    # Per-class F1 bar chart
    with open(REPORTS / "5class_vus" / "histgradientboosting_metrics.json") as fh:
        hgb5 = json.load(fh)
    per_class = hgb5["holdout"]["f1_per_class"]
    classes = ["Benign", "Likely benign", "Likely pathogenic", "Pathogenic", "VUS"]
    f1s = [round(per_class[c], 3) for c in classes]

    chart_data = CategoryChartData()
    chart_data.categories = classes
    chart_data.add_series("Per-class F1 (5-class HistGB)", f1s)
    chart = s.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(7.0), Inches(1.3), Inches(6.0), Inches(4.7),
        chart_data,
    ).chart
    style_chart_minimal(chart, show_value=True, value_format="0.000",
                        label_size=12, axis_size=12)
    chart.value_axis.minimum_scale = 0.5
    chart.value_axis.maximum_scale = 1.0

    add_caption(s, Inches(7.0), Inches(6.1), Inches(6.0), Inches(0.7),
                "VUS reaches F1 = 0.901 — between Benign and Likely-benign in difficulty. "
                "One of the cleanest classes to recognise.",
                color=DARK, size=13)

    add_slide_number(s, 13)

    # =========================================================================
    # Slide 14 — 5-class confusion matrix (with VUS as a class)
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "5-class confusion matrix — where VUS sits")
    add_caption(s, MARGIN, Inches(1.2), Inches(12.1), Inches(0.5),
                "Held-out, row-normalized. VUS row reaches 0.94 diagonal — cleaner than Pathogenic (0.64) "
                "or Likely-pathogenic (0.61). VUS misclassifications hedge into Likely-*, not into definitive calls.")
    fit_image(s, "outputs/reports/5class_vus/histgradientboosting_confusion_matrix_normalized.png",
              x=Inches(2.5), y=Inches(2.0), max_w=Inches(8.3), max_h=Inches(5.0))
    add_slide_number(s, 14)

    # =========================================================================
    # Slide 15 — SHAP: what drives the model
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "What drives the booster — SHAP top-30")
    add_caption(s, MARGIN, Inches(1.2), Inches(12.1), Inches(0.5),
                "Canonical 4-class HistGradientBoosting. DITTO leads, MetaRNN second, "
                "then population allele frequencies — the model is using both predictor scores and frequency data.")
    fit_image(s, "final_report/figures/shap_bar_4class.png",
              x=Inches(2.5), y=Inches(2.0), max_w=Inches(8.3), max_h=Inches(5.0))
    add_slide_number(s, 15)

    # =========================================================================
    # Slide 16 — Predictor correlation heatmap
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "Predictor correlation — the redundancy structure")
    add_caption(s, MARGIN, Inches(1.5), Inches(12.1), Inches(0.7),
                "Pearson correlation among key VEP scores + gnomAD AF. Predictor scores form a tight cluster; "
                "gnomAD AF sits in its own corner — the structural reason predictors substitute for each other, "
                "but population AF does not.")
    fit_image(s, "outputs/eda/key_vep_score_correlation.png",
              x=Inches(1.5), y=Inches(2.5), max_w=Inches(10.3), max_h=Inches(4.5))
    add_slide_number(s, 16)

    # =========================================================================
    # Slide 17 — Feature ablation table
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "Useful vs. waste — feature ablation")

    head = pd.read_csv(REPORTS / "feature_ablation" / "headline_histgb.csv", index_col=0)
    head = head[["4class", "2class", "5class", "3class"]]
    head = head.reindex(head.abs().max(axis=1).sort_values(ascending=False).index)

    ablation_rows = [("Group", "4-class", "2-class", "5-class", "3-class", "n feat")]
    n_feat_map = {"population_af": 47, "functional": 4, "other_vep": 56, "cadd": 4,
                  "ditto": 1, "metarnn": 2, "bayesdel": 4, "position": 2,
                  "revel": 2, "chasmplus": 68, "alphamissense": 1, "conservation": 17}
    for grp, row in head.iterrows():
        ablation_rows.append((
            grp,
            f"{row['4class']:+.4f}",
            f"{row['2class']:+.4f}",
            f"{row['5class']:+.4f}",
            f"{row['3class']:+.4f}",
            str(n_feat_map.get(grp, "-")),
        ))

    waste_groups = {"ditto", "metarnn", "bayesdel", "position", "revel",
                    "chasmplus", "alphamissense", "conservation"}

    rows, cols = len(ablation_rows), 6
    tbl = s.shapes.add_table(rows, cols, MARGIN, Inches(1.3),
                              Inches(12.1), Inches(0.36 * rows)).table
    tbl.columns[0].width = Inches(3.4)
    for c in range(1, 5):
        tbl.columns[c].width = Inches(1.7)
    tbl.columns[5].width = Inches(1.9)

    for r, row in enumerate(ablation_rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            for run in p.runs:
                f = run.font
                f.name = BODY_FONT
                f.size = Pt(13)
                f.bold = (r == 0)
                if r > 0 and ablation_rows[r][0] in waste_groups:
                    f.color.rgb = GREY
                else:
                    f.color.rgb = BLACK
            cell.margin_left = Inches(0.1)
            cell.margin_right = Inches(0.1)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PALE if r == 0 else WHITE

    add_caption(s, MARGIN, Inches(6.4), Inches(12.1), Inches(0.6),
                "Greyed rows: HistGB |Δ| < 0.005 across every task. "
                "8 of 12 groups (97 of 208 features) are waste at the group level.",
                color=GREY, size=13)

    add_slide_number(s, 17)

    # =========================================================================
    # Slide 18 — Per-feature permutation: the redundancy insight
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "Per-feature permutation — the redundancy story")

    add_caption(s, MARGIN, Inches(1.5), Inches(12.1), Inches(0.7),
                "Removing a 'waste' group is cheap — but individual features inside those groups are still highly "
                "predictive. Group LOO measures replaceability, not informativeness.", color=DARK, size=15)

    # Two columns of big numbers
    add_textbox(s, MARGIN, Inches(2.6), Inches(6), Inches(0.5),
                text="ditto_score", font=TITLE_FONT, size=28, bold=True, color=BLACK)
    add_textbox(s, MARGIN, Inches(3.2), Inches(6), Inches(1.3),
                text="0.133", font=TITLE_FONT, size=72, bold=True, color=BLACK)
    add_textbox(s, MARGIN, Inches(4.6), Inches(6), Inches(0.5),
                text="individual macro-F1 weight (4-class)",
                font=BODY_FONT, size=14, color=GREY)

    add_textbox(s, Inches(7), Inches(2.6), Inches(6), Inches(0.5),
                text="metarnn_score", font=TITLE_FONT, size=28, bold=True, color=BLACK)
    add_textbox(s, Inches(7), Inches(3.2), Inches(6), Inches(1.3),
                text="0.048", font=TITLE_FONT, size=72, bold=True, color=BLACK)
    add_textbox(s, Inches(7), Inches(4.6), Inches(6), Inches(0.5),
                text="individual macro-F1 weight (4-class)",
                font=BODY_FONT, size=14, color=GREY)

    add_caption(s, MARGIN, Inches(5.8), Inches(12.1), Inches(1.4),
                "Inter-predictor redundancy. Remove DITTO and MetaRNN compensates; remove MetaRNN and DITTO compensates. "
                "Only by removing several correlated meta-classifiers together does the redundancy buffer collapse — "
                "which is what Lean B exploits.", color=DARK, size=15)

    add_slide_number(s, 18)

    # =========================================================================
    # Slide 19 — Lean B vs baseline
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "Lean B — 45% smaller, 0.012 cost")

    add_caption(s, MARGIN, Inches(1.3), Inches(12.1), Inches(0.5),
                "Lean B drops 95 of 208 features, keeping ditto_score and metarnn_score (the per-feature winners).",
                size=15)

    tasks_lean = ["2-class", "3-class", "4-class", "5-class"]
    baseline = [0.9872, 0.9459, 0.7950, 0.7984]
    lean_b   = [0.9863, 0.9386, 0.7900, 0.7867]
    chart_data = CategoryChartData()
    chart_data.categories = tasks_lean
    chart_data.add_series("Baseline (208 features)", baseline)
    chart_data.add_series("Lean B (113 features)",   lean_b)
    chart = s.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        MARGIN, Inches(2.0), Inches(12.1), Inches(4.4),
        chart_data,
    ).chart
    chart.has_title = False
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.name = BODY_FONT
    chart.legend.font.size = Pt(14)
    chart.legend.font.color.rgb = GRAPHITE
    plot = chart.plots[0]
    for i, series in enumerate(plot.series):
        fill = series.format.fill; fill.solid()
        fill.fore_color.rgb = BLACK if i == 0 else LIGHT
        line = series.format.line
        line.color.rgb = DARK
        line.width = Pt(0.75)
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.font.name = BODY_FONT
    dl.font.size = Pt(11)
    dl.font.color.rgb = GRAPHITE
    dl.number_format = "0.000"
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    cat = chart.category_axis; val = chart.value_axis
    for ax in (cat, val):
        ax.tick_labels.font.name = BODY_FONT
        ax.tick_labels.font.size = Pt(13)
        ax.tick_labels.font.color.rgb = GRAPHITE
    val.minimum_scale = 0.5
    val.maximum_scale = 1.05
    val.format.line.color.rgb = LIGHT
    cat.format.line.color.rgb = LIGHT

    add_caption(s, MARGIN, Inches(6.5), Inches(12.1), Inches(0.5),
                "At most 0.012 macro-F1 cost on any task. Several models (KNN, NearestCentroid, CosineSimilarity) "
                "actually improve on the lean subset.",
                color=DARK, size=14)
    add_slide_number(s, 19)

    # =========================================================================
    # Slide 20 — Future directions
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_slide_title(s, "Future directions")

    directions = [
        ("Cross-database validation",
         "replicate Lean B numbers on HGMD or a held-out ClinVar snapshot"),
        ("True homology BLAST",
         "blastp on UniRef50 with the amino-acid substitution applied"),
        ("Finish the AdaBoost tuning grid",
         "the partial 23/27 grid was halted for compute budget"),
        ("Probability calibration",
         "Platt or isotonic regression so predict_proba is a usable risk score"),
        ("Gene-stratified VUS-as-class",
         "test whether VUS-detection survives without same-gene leakage"),
        ("Variant-level error analysis",
         "manual review of the Pathogenic ↔ Likely-pathogenic confusion region"),
    ]
    y = Inches(1.5)
    for name, desc in directions:
        add_textbox(s, MARGIN, y, Inches(0.4), Inches(0.5),
                    text="·", font=TITLE_FONT, size=24, bold=True, color=BLACK)
        add_textbox(s, Inches(1.0), y, Inches(11.5), Inches(0.5),
                    text=name, font=BODY_FONT, size=20, bold=True, color=BLACK)
        add_textbox(s, Inches(1.3), y + Inches(0.45), Inches(11.5), Inches(0.5),
                    text=desc, font=BODY_FONT, size=14, color=GRAPHITE,
                    line_spacing=1.25)
        y = y + Inches(0.85)

    add_slide_number(s, 20)

    # =========================================================================
    # Slide 21 — Thank you / Q&A
    # =========================================================================
    s = prs.slides.add_slide(blank)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(0.0),
                             Inches(0.03), SLIDE_H)
    bar.fill.solid(); bar.fill.fore_color.rgb = BLACK
    bar.line.fill.background()

    add_textbox(s, Inches(1.2), Inches(2.5), Inches(11), Inches(1.5),
                text="Thank you", font=TITLE_FONT, size=88, bold=True, color=BLACK)
    add_textbox(s, Inches(1.2), Inches(4.1), Inches(11), Inches(0.6),
                text="Questions?", font=TITLE_FONT, size=32, italic=True, color=DARK)
    add_textbox(s, Inches(1.2), Inches(5.8), Inches(11), Inches(0.4),
                text="Doruk Topcu  ·  Alihan Sağöz",
                font=BODY_FONT, size=18, color=BLACK)
    add_textbox(s, Inches(1.2), Inches(6.3), Inches(11), Inches(0.4),
                text="github.com/doruktopcu/MissVarPath-Missense-Detection",
                font=BODY_FONT, size=14, italic=True, color=GREY)
    add_slide_number(s, 21)

    # =========================================================================
    # Write — fall back to a versioned filename if the canonical one is locked
    # (typically: user has it open in PowerPoint).
    # =========================================================================
    out = Path("MissVarPath_Presentation.pptx")
    if out.exists():
        try:
            with out.open("ab"):
                pass
        except PermissionError:
            out = Path("MissVarPath_Presentation_v2.pptx")
            print(f"[build] canonical pptx is locked; writing to {out} instead.")
    prs.save(out)
    print(f"Wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
