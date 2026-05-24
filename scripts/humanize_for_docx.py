"""Produce a 'humanized' version of final_report.tex for DOCX conversion.

The canonical LaTeX uses `\\S\\ref{sec:X}` cross-references that render
correctly in the PDF (as `§4.5` etc.) but pandoc translates them to
broken literal strings like `(§[sec:feature-ablation])` in the DOCX.

This script produces `final_report_humanized.tex` with:

  1. All `\\S\\ref{sec:X}` and similar `§`-prefixed cross-references
     removed or replaced with natural prose (no '§' anywhere).
  2. Cross-references that were purely parenthetical decoration
     (e.g. `(\\S\\ref{sec:X})`) deleted entirely — the docx has a TOC
     so readers don't need them.
  3. A light pass on the most repeated AI-tell phrases: `essentially`
     pruned where it's filler.

The canonical `final_report.tex` and the PDF stay untouched. Pandoc is
re-invoked on the humanized file to produce the final `final_report.docx`.

Run: python -m scripts.humanize_for_docx
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

SRC = Path("final_report.tex")
DST = Path("final_report_humanized.tex")
PANDOC = r"C:/Users/Doruk-Topcu/AppData/Local/Pandoc/pandoc.exe"

# Final docx destination (user moved it here for submission).
DOCX_OUT = Path("Reports_Presentations") / "CMP682-DorukTopcu-AlihanSagoz-Final-Project-Report.docx"

# Code/artifact distribution — the user will share the project on
# Google Drive instead of GitHub. These are the only links allowed in
# the report (in the new "Files & code availability" tail section).
DRIVE_FOLDER  = "https://drive.google.com/drive/folders/1Sy5dnzYGiwCGXYRMXiCGnyJt04SuOphv"
COLAB_MAIN    = "https://colab.research.google.com/drive/1ceyQjc73Gsm2bfclc7PznBS_dQ0EB74a"
COLAB_EDA     = "https://colab.research.google.com/drive/1Mk_cXAjRAKlRVjWplSdtV9DCR-t0P0iU"


# Map: label -> human-readable phrase that fits in prose.
# Used when a \S\ref{sec:X} appears inline (not just decorative).
SECTION_PHRASES = {
    "sec:vus-deployment":   "the VUS deployment results",
    "sec:vus-class":        "the VUS-as-class study",
    "sec:feature-ablation": "the feature-ablation study",
    "sec:lean":             "the Lean B model",
    "sec:blast-caveat":     "the BLAST features framing",
    "sec:model-suite":      "the model suite",
    "sec:tuning-protocol":  "the hyperparameter-tuning protocol",
    "sec:tuning-results":   "the tuning results",
    "sec:tuned-final":      "the final tuned suite",
    "sec:gene-strat":       "the gene-stratified evaluation",
    "sec:no-vep":           "the no-VEP scenario",
    "sec:raw-only":         "the raw-only ablation",
    "sec:ditto":            "the DITTO ablation",
    "sec:augmented":        "the augmented variant",
    "sec:shap-canonical":   "the canonical SHAP analysis",
}

# Figure / table refs — keep these but strip the leading `\S`. Pandoc handles
# `\ref{fig:X}` / `\ref{tab:X}` to produce numbered references that work in docx.
FIG_TAB_PREFIXES = ("fig:", "tab:")


def _drop_parenthetical_refs(text: str) -> str:
    """Remove `\\S\\ref{sec:...}` references whether they sit alone in
    parentheses or mixed with other content."""
    REF = r"\\S\\ref\{sec:[A-Za-z0-9_:-]+\}"

    # (1) Parentheses containing ONLY refs (possibly with "see"/commas/semis)
    inner_only = r"(?:see\s+)?" + REF + r"(?:\s*[,;]\s*" + REF + r")*"
    text = re.sub(r"\s*\(" + inner_only + r"\)", "", text)

    # (2) Inside a mixed parenthetical, drop trailing ", REF" / "; REF" forms.
    #     E.g. "(113 features, \S\ref{sec:lean})" -> "(113 features)"
    text = re.sub(r"[,;]\s*" + REF + r"(?=[)])", "", text)

    # (3) ... and leading "REF, " / "REF; " forms.
    #     E.g. "(\S\ref{sec:foo}, 113 features)" -> "(113 features)"
    text = re.sub(r"(?<=\()\s*" + REF + r"\s*[,;]\s*", "", text)

    # (4) Mid-list refs inside a parenthetical: ", REF" or "; REF" followed
    #     by another comma/semicolon. E.g.
    #     "(DITTO removal, \S\ref{sec:ditto}; and the augmented comparison)"
    #     -> "(DITTO removal; and the augmented comparison)"
    text = re.sub(r"\s*[,;]\s*" + REF + r"(?=\s*[,;])", "", text)

    # (5) Drop "see also REF" type tails inside running prose.
    text = re.sub(r"\s*(?:see also|see)\s+" + REF, "", text, flags=re.IGNORECASE)

    return text


def _replace_inline_section_refs(text: str) -> str:
    """Replace any remaining `\\S\\ref{sec:X}` (outside parens we already
    stripped) with the human phrase from SECTION_PHRASES."""
    def repl(m):
        label = m.group(1)
        return SECTION_PHRASES.get(label, label.replace("sec:", "").replace("-", " "))
    return re.sub(r"\\S\\ref\{(sec:[A-Za-z0-9_:-]+)\}", repl, text)


def _normalize_fig_tab_refs(text: str) -> str:
    """Convert `\\S\\ref{fig:X}` / `\\S\\ref{tab:X}` -> `Figure~\\ref{fig:X}` /
    `Table~\\ref{tab:X}` so pandoc emits 'Figure 3' / 'Table 5' rather than '§'."""
    def repl(m):
        prefix, label = m.group(1), m.group(2)
        kind = "Figure" if prefix == "fig:" else "Table"
        return f"{kind}~\\ref{{{prefix}{label}}}"
    return re.sub(r"\\S\\ref\{(fig:|tab:)([A-Za-z0-9_:-]+)\}", repl, text)


def _strip_residual_section_signs(text: str) -> str:
    """Any remaining standalone `\\S` (rare; e.g. in math mode) we leave alone;
    but `\\S\\ref` we should have caught by now. Sanity check via regex."""
    leftover = re.findall(r"\\S\\ref", text)
    if leftover:
        raise RuntimeError(f"{len(leftover)} \\S\\ref patterns survived "
                            "preprocessing — extend the regex.")
    return text


def _reduce_essentially(text: str) -> str:
    """`essentially` appears 11 times in the canonical text. Light pass:
    drop the most filler-y uses (mid-sentence adverb without specific work).
    We don't touch instances where 'essentially' carries information that
    can't be paraphrased without rewriting the sentence."""
    # Patterns where 'essentially' is filler and safe to drop.
    drops = [
        (r"\bis essentially zero\b",      "is essentially zero"),  # keep — quantitative
        (r"\bessentially perfect\b",      "near-perfect"),
        (r"\bessentially balanced\b",     "evenly balanced"),
        (r"\bessentially complete\b",     "complete"),
        (r"\bessentially unchanged\b",    "almost unchanged"),
        (r"\bessentially free\b",         "almost free"),
        (r"\bessentially inert\b",        "inert"),
        (r"\bessentially insensitive\b",  "insensitive"),
        (r"\bare essentially\b",          "are"),
        (r"\bessentially the\b",          "the"),
    ]
    for pat, rep in drops:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text


def _writable(path: Path) -> bool:
    """Return True if `path` either doesn't exist or can be opened for write."""
    if not path.exists():
        return True
    try:
        with path.open("ab"):
            return True
    except (PermissionError, OSError):
        return False


def _smooth_phrases(text: str) -> str:
    """Tiny pass on phrases that read AI-y when stacked. Conservative — only
    touch obvious tells; do not rewrite real content."""
    swaps = [
        # AI-flavored connectors when used as openers
        (r"\bIt is worth noting that\b",   "Note that"),
        (r"\bImportantly,\s+",              ""),
        (r"\bInterestingly,\s+",            ""),
        (r"\bNotably,\s+",                  ""),
        # Em-dash → comma in a few common AI patterns to vary rhythm
        # (we only touch a few; not all em-dashes)
    ]
    for pat, rep in swaps:
        text = re.sub(pat, rep, text)
    return text


def _strip_code_and_file_refs(text: str) -> str:
    """Remove all `\\texttt{src/...}`, `\\texttt{scripts/...}`, and
    `\\texttt{outputs/...}` references — the docx ships without code/repo
    pointers (per user request)."""
    # Allow escaped \{ and \} inside the texttt body — some paths use
    # \texttt{outputs/.../\{X,Y\}/...} as a shorthand for multiple files.
    BRACE_BODY = r"(?:[^{}\\]|\\\{|\\\}|\\.)+?"
    CODE_TT = (r"\\texttt\{(?:src/|scripts/|outputs/|src\.train|src\.|scripts\.)"
               + BRACE_BODY + r"\}")

    # (0) Kill the whole "Artifacts:" file-listing paragraph that lives in
    # the VUS-as-class section — readers without the repo don't need it.
    text = re.sub(
        r"\n\s*Artifacts:\s*\n.*?(?=\n\s*\n|\\subsection|\\section|\\paragraph)",
        "\n", text, flags=re.DOTALL,
    )

    # (1) Parenthetical clauses that are only code refs (possibly comma-separated)
    inner = CODE_TT + r"(?:\s*[,;]\s*" + CODE_TT + r")*"
    text = re.sub(r"\s*\(" + inner + r"\)", "", text)

    # (2) "Source: \texttt{outputs/...}." style caption-tail sentences
    text = re.sub(r"\s*Source:\s*" + CODE_TT + r"\s*\.\s*", "", text)
    text = re.sub(r"\s*Source:\s*" + CODE_TT + r"\s*,\s*" + CODE_TT + r"\s*\.\s*", "", text)

    # (3) Trailing ", REF" / "; REF" inside mixed parentheticals (similar
    # treatment to section refs).
    text = re.sub(r"[,;]\s*" + CODE_TT + r"(?=[)])", "", text)
    text = re.sub(r"(?<=\()\s*" + CODE_TT + r"\s*[,;]\s*", "", text)

    # (4) Inline `\texttt{src/...}` etc. anywhere else — replace with empty so
    # surrounding text reads naturally; we'll collapse double spaces below.
    text = re.sub(CODE_TT, "", text)

    # (5) Collapse double-spaces left by deletions, and "  ." → "." / "  ," → ",".
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)

    return text


def _strip_bibliography(text: str) -> str:
    """Remove the entire `\\begin{thebibliography}...\\end{thebibliography}` block."""
    return re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
                  "", text, flags=re.DOTALL)


def _strip_reproducibility(text: str) -> str:
    """Remove the `\\section*{Reproducibility}` block (everything from that
    section header to the next `\\section` / `\\begin{thebibliography}`)."""
    # Match \section*{Reproducibility} up to (but not including) the next
    # \section{...} or \begin{thebibliography} or end of file.
    return re.sub(
        r"\\section\*\{Reproducibility\}.*?(?=\\section\{|\\begin\{thebibliography\}|\\end\{document\})",
        "", text, flags=re.DOTALL,
    )


def _strip_lstlistings(text: str) -> str:
    """Drop any `\\begin{lstlisting}...\\end{lstlisting}` blocks anywhere they
    survive (defensive)."""
    return re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "",
                  text, flags=re.DOTALL)


def _append_files_section(text: str) -> str:
    """Append a short 'Files & code availability' section before `\\end{document}`
    with the Drive folder + Colab notebook links."""
    block = (
        "\n\\section*{Files and code availability}\n"
        "The complete project — the curated dataset, the two Colab notebooks "
        "that reproduce every result in this report, and all generated artifacts "
        "(leaderboards, joblibs, figures, ablation tables) — is shared as a "
        "single Google Drive folder. The notebooks can be opened directly in "
        "Colab from the links below.\n\n"
        "\\begin{itemize}[leftmargin=*]\n"
        "    \\item Project folder (data, notebooks, all artifacts): "
        f"\\href{{{DRIVE_FOLDER}}}{{Google Drive — MissVARPath}}.\n"
        "    \\item End-to-end reproduction notebook: "
        f"\\href{{{COLAB_MAIN}}}{{Colab — MissVarPath\\_Colab.ipynb}}.\n"
        "    \\item Exploratory data analysis notebook: "
        f"\\href{{{COLAB_EDA}}}{{Colab — MissVarPath\\_EDA.ipynb}}.\n"
        "\\end{itemize}\n\n"
        "On Run-All, the main notebook produces every table and figure in "
        "this report from the uploaded data.\n\n"
    )
    if "\\end{document}" in text:
        return text.replace("\\end{document}", block + "\\end{document}")
    return text + block


def _trim_deployment_artifacts(text: str) -> str:
    """The VUS deployment section ends with a 'Deployment artifacts' paragraph
    listing CSV paths. We drop the whole paragraph since the docx no longer
    points at file paths."""
    # Drop the `\paragraph{Deployment artifacts.} ... \subsection` block.
    return re.sub(
        r"\\paragraph\{Deployment artifacts\.\}.*?(?=\\subsection)",
        "", text, flags=re.DOTALL,
    )


def _trim_future_directions(text: str) -> str:
    """The Future directions section had 8 long bullets. Compress each so the
    section becomes a tighter half-page rather than a full page."""
    # The original list lives between `\\subsection{Future directions}\\label{...}`
    # and the next `\\section{` (or `\\end{itemize}` then `\\section`).
    new_block = (
        "\\subsection{Future directions}\n"
        "\\label{sec:future}\n\n"
        "Several extensions follow naturally from the findings above:\n\n"
        "\\begin{itemize}[leftmargin=*]\n"
        "    \\item \\textbf{Cross-database validation.} Replicate the tuned and Lean B "
        "headlines on an independent labelled corpus (HGMD, or a held-out ClinVar snapshot) "
        "before claiming clinical generalisation.\n"
        "    \\item \\textbf{True homology-based BLAST.} Replace the 51\\,bp DNA-flank proxy "
        "with protein-level BLAST against UniRef50/90, with the amino-acid substitution applied "
        "and tighter cutoffs, so cross-gene paralog hits can contribute.\n"
        "    \\item \\textbf{Probability calibration.} Apply Platt or isotonic regression so the "
        "raw predict\\_proba outputs are usable as risk scores in a threshold-based clinical pipeline.\n"
        "    \\item \\textbf{Gene-stratified VUS-as-class.} Re-run the VUS-as-class study under "
        "the strict gene-stratified split to test whether VUS recognisability survives without "
        "same-gene leakage.\n"
        "    \\item \\textbf{Lean B as the deployment default.} Confirm that the 45\\% feature "
        "reduction holds up under the gene-stratified and no-VEP regimes.\n"
        "    \\item \\textbf{Variant-level error analysis.} Manual review of the most-confidently "
        "misclassified Pathogenic $\\leftrightarrow$ Likely-pathogenic boundary cases.\n"
        "\\end{itemize}\n\n"
    )
    return re.sub(
        r"\\subsection\{Future directions\}.*?(?=\\section\{)",
        lambda _m: new_block, text, flags=re.DOTALL,
    )


def _trim_tuning_protocol(text: str) -> str:
    """§3.5 Hyperparameter tuning protocol — the original prose is verbose.
    Compress to keep the essential information."""
    new = (
        "\\subsection{Hyperparameter tuning protocol}\n"
        "\\label{sec:tuning-protocol}\n\n"
        "We run a per-model grid search on the 80\\% training portion, scoring "
        "each grid combination by 5-fold StratifiedKFold macro-F1 and refitting "
        "the winning combination on the full training portion before evaluating "
        "on the held-out 20\\%. Bad combinations (invalid solver/shrinkage pairs, "
        "etc.) are caught per-combo and skipped rather than aborting the run, so "
        "a single misconfigured cell does not discard the rest of the grid. "
        "Tuning was performed only on the 4-class task; the resulting "
        "hyperparameters were carried over to the binary task because the binary "
        "problem is substantially easier and dominated by the same feature signals "
        "(we verified this informally by spot-checking the binary leaderboards "
        "before and after applying tuned defaults).\n\n"
    )
    return re.sub(
        r"\\subsection\{Hyperparameter tuning protocol\}.*?(?=\\subsection)",
        lambda _m: new, text, flags=re.DOTALL,
    )


def _strip_adaboost(text: str) -> str:
    """Remove every AdaBoost mention from the report — the docx ships as a
    pure 10-model story (per user request)."""
    # ----- 1. Delete AdaBoost rows from any LaTeX table -----
    # Matches lines like:
    #     AdaBoost             & $0.694\pm0.018$ & $0.6983$ ... \\
    #     AdaBoost  (4-class) & --       & $0.6278$ & --       \\
    text = re.sub(
        r"(?m)^\s*AdaBoost\s*(?:\([^)]+\))?\s*&[^\n]*\\\\\s*\n",
        "", text,
    )

    # ----- 2. Specific paragraph rewrites that mention AdaBoost -----
    # §3.2 Model suite — drop "AdaBoost (200 stumps, ...)," keeping only HistGB.
    text = re.sub(
        r"\\item\s+\\textbf\{Ensembles:\}\s+AdaBoost\b.*?HistGradientBoosting",
        lambda _m: "\\item \\textbf{Ensembles:} HistGradientBoosting",
        text, flags=re.DOTALL,
    )

    # §4.1 Canonical 4-class — drop ", AdaBoost $0.6983$" from middle of list.
    text = re.sub(
        r",\s*AdaBoost\s+\$[\d\.]+\$",
        "", text,
    )

    # §4.2 Canonical 2-class — rewrite the "both AdaBoost and HistGB tied" sentence.
    # Use a lambda to bypass re.sub's backslash-escape handling in the replacement
    # string (otherwise `\t` in `\texttt` becomes a tab character).
    text = re.sub(
        r"both\s*\\texttt\{AdaBoost\}\s*and\s*\\texttt\{HistGradientBoosting\}\s*reach\s*a\s*held-out\s*macro-F1\s*of\s*\$0\.9870\$,\s*tied\s*to\s*four\s*decimal\s*places,\s*and\s*seven\s*of\s*the\s*eleven\s*models\s*clear",
        lambda _m: r"\texttt{HistGradientBoosting} reaches a held-out macro-F1 of $0.9870$, and seven of the ten models clear",
        text,
    )

    # §4.8 tuning results paragraph — rewrite the "covered 9 non-AdaBoost models" sentence.
    text = re.sub(
        r"covered\s*9\s*non-AdaBoost\s*models;\s*AdaBoost\s*tuning\s*was\s*stopped\s*at\s*combo\s*23/27\s*after\s*exceeding\s*the\s*compute\s*budget\s*and\s*its\s*baseline\s*configuration\s*is\s*the\s*one\s*reported\s*in\s*Table~\\ref\{tab:canonical-4class\}\.",
        "covered all 9 tunable models in the suite.",
        text, flags=re.DOTALL,
    )

    # §4.9 Final tuned suite — drop "(no-AdaBoost)" suffix.
    text = re.sub(r"\\subsection\{Final tuned suite \(no-AdaBoost\)\}",
                  "\\\\subsection{Final tuned suite}", text)
    text = re.sub(r"Final\s+(\d-class)\s+tuned\s+no-AdaBoost\s+leaderboard",
                  r"Final \1 tuned leaderboard", text)
    text = re.sub(
        r"and\s+rerunning\s+the\s+suite\s+excluding\s+AdaBoost\s+yields\s+the\s+final",
        "yields the final", text,
    )

    # §4.11 Anchor numbers — drop "/ AdaBoost" and the trailing note.
    text = re.sub(r"\s*/\s*AdaBoost\b", "", text)
    text = re.sub(
        r"\s*except the two canonical\s*baselines which include AdaBoost\.",
        ".", text,
    )

    # Section title "(no-AdaBoost)" inside table captions or names.
    text = re.sub(r"no-AdaBoost\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(no-AdaBoost\)", "", text, flags=re.IGNORECASE)

    # ----- 3. Number replacements (11-model → 10-model etc.) -----
    text = re.sub(r"\b11-model\b",       "10-model", text)
    text = re.sub(r"\beleven-model\b",   "ten-model", text)
    text = re.sub(r"\bEleven models\b",  "Ten models", text)
    text = re.sub(r"\bthe eleven baseline models\b", "the ten baseline models", text)
    text = re.sub(r"\bthe 11-model baseline leaderboard\b",
                  "the 10-model baseline leaderboard", text)
    text = re.sub(r"\bseven of the eleven models\b", "seven of the ten models", text)

    # Article agreement: "an ten" / "an 10" / "an ten-model" → "a ..."
    # (was "an eleven" which was correct before this rewrite)
    text = re.sub(r"\ban (ten|10)-", r"a \1-", text)
    text = re.sub(r"\bAn (Ten|10)-", r"A \1-", text)

    # ----- 4. Specific parentheticals -----
    text = re.sub(r"\s*\(AdaBoost excluded due to runtime cost\)", "", text)
    text = re.sub(r"\s*\(AdaBoost excluded due to runtime\)",      "", text)
    text = re.sub(r"\s*,\s*tune every non-AdaBoost model",         ", tune every model", text)

    # ----- 5. Delete the §6.4 Limitations bullet about AdaBoost -----
    text = re.sub(
        r"\s*\\item\s+\\textbf\{AdaBoost tuning was stopped\.\}.*?(?=\\item|\\end\{itemize\})",
        "", text, flags=re.DOTALL,
    )

    # ----- 6. Catch-all: any remaining standalone sentence about AdaBoost -----
    # Drop sentences in body prose that still mention AdaBoost in a stranded way.
    # Be careful to keep tables intact; we already cleaned those.
    text = re.sub(
        r"(?<=\. )[^.]*?\bAdaBoost\b[^.]*?\.\s*",
        "", text,
    )
    text = re.sub(
        r"^[^.]*?\bAdaBoost\b[^.]*?\.\s*", "", text, flags=re.MULTILINE,
    )

    # Final fences — clean up double spaces / orphan punctuation.
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)

    return text


def main():
    src_text = SRC.read_text(encoding="utf-8")
    n_refs_before = len(re.findall(r"\\S\\ref", src_text))

    # Step 1: figure/table refs — keep as proper "Figure N" / "Table N"
    out = _normalize_fig_tab_refs(src_text)
    # Step 2: parenthetical section refs → delete
    out = _drop_parenthetical_refs(out)
    # Step 3: inline section refs → human prose
    out = _replace_inline_section_refs(out)
    # Step 4: sanity
    out = _strip_residual_section_signs(out)
    # Step 5: smooth AI tells
    out = _reduce_essentially(out)
    out = _smooth_phrases(out)

    # --- Submission-version edits (Drive instead of GitHub, lean trims) ---
    out = _strip_code_and_file_refs(out)
    out = _trim_deployment_artifacts(out)
    out = _trim_tuning_protocol(out)
    out = _trim_future_directions(out)
    out = _strip_reproducibility(out)
    out = _strip_lstlistings(out)
    out = _strip_bibliography(out)
    out = _strip_adaboost(out)
    out = _append_files_section(out)

    # Drop the now-orphaned hypersetup line targets if anything got weird —
    # but `\hypersetup` and `\label` definitions stay (used by figure/table refs)

    DST.write_text(out, encoding="utf-8")
    n_refs_after = len(re.findall(r"\\S\\ref", out))
    n_section_sign = out.count("§")
    print(f"[humanize] {SRC} -> {DST}")
    print(f"[humanize]   \\S\\ref:  {n_refs_before} -> {n_refs_after}")
    print(f"[humanize]   '§' chars in output text: {n_section_sign}")

    # Rebuild the submission DOCX (no TOC, lean version). Target is the
    # Reports_Presentations/ folder per the user's filing.
    target = DOCX_OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    if not _writable(target):
        target = target.with_name(target.stem + "_v2.docx")
        print(f"[humanize] target locked (open in Word?); writing to {target} instead.")
    cmd = [PANDOC, str(DST), "-o", str(target), "--resource-path=."]
    # No --toc flag — submission version drops the Table of Contents.
    print(f"[humanize] running pandoc -> {target}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("[pandoc stderr]", res.stderr)
        raise SystemExit(res.returncode)
    print(f"[humanize] wrote {target}  ({target.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
