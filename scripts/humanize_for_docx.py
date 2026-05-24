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

    # Drop the now-orphaned hypersetup line targets if anything got weird —
    # but `\hypersetup` and `\label` definitions stay (used by figure/table refs)

    DST.write_text(out, encoding="utf-8")
    n_refs_after = len(re.findall(r"\\S\\ref", out))
    n_section_sign = out.count("§")
    print(f"[humanize] {SRC} -> {DST}")
    print(f"[humanize]   \\S\\ref:  {n_refs_before} -> {n_refs_after}")
    print(f"[humanize]   '§' chars in output text: {n_section_sign}")

    # Rebuild the DOCX from the humanized source. If `final_report.docx` is
    # locked (typically because the user has it open in Word), write to a
    # `_humanized.docx` companion file instead.
    target = Path("final_report.docx")
    if not _writable(target):
        target = Path("final_report_humanized.docx")
        print(f"[humanize] final_report.docx is locked (open in Word?); "
              f"writing to {target} instead.")
    cmd = [PANDOC, str(DST), "-o", str(target),
           "--resource-path=.", "--toc", "--toc-depth=3"]
    print(f"[humanize] running pandoc -> {target}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("[pandoc stderr]", res.stderr)
        raise SystemExit(res.returncode)
    print(f"[humanize] wrote {target}  ({target.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
