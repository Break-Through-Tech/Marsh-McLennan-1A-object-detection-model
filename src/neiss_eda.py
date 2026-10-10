"""NEISS exploratory data analysis for the Marsh object-detection project.

Produces a statistics report on the injury narratives and the product-code
labels, covering:

  1. Dataset shape and field-by-field completeness
  2. Narrative length
  3. Narrative structure and preprocessing (the formulaic parts we can strip
     or exploit, and what the Task #6 cleaning rules change)
  4. Label structure (product codes: cardinality, concentration, multi-label)
  5. Narrative-to-label linkage (is the object actually named in the text?)
  6. The consumer-vs-workplace domain gap
  7. Class imbalance: what a stratified split can survive
  8. Hidden confounders: hospital, season, duplicate narratives
  9. Balancing exposure across incident contexts (balancing the label)
 10. Proposed split and sampling recipe

The modelling task this supports is Narrative_1 (text in) -> Product_1 (single
product code out).

Usage
-----
    python src/neiss_eda.py                      # report to stdout + reports/
    python src/neiss_eda.py --data data/raw/neiss2025.xlsx
    python src/neiss_eda.py --no-cache           # force a re-read of the xlsx

The first run parses the .xlsx (~60s for a full year) and caches both sheets
as pickles under data/processed/.cache/ so later runs start in about a second.
"""

import argparse
import os
import re
import sys
import textwrap
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from preprocess import (ABBREVIATIONS, clean_whitespace, has_redaction,
                        normalize_abbreviations, split_diagnosis)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(REPO, "data", "raw", "neiss2025.xlsx")
CACHE_DIR = os.path.join(REPO, "data", "processed", ".cache")   # gitignored
REPORT_DIR = os.path.join(REPO, "reports")

# Product slots. NEISS codes "no product in this slot" as 0, not as blank.
PRODUCT_COLS = ["Product_1", "Product_2", "Product_3"]
NO_PRODUCT = "0"

# Fields that describe the survey design rather than the incident. These leak
# sampling structure into a model and are never present on a claims narrative,
# so they are excluded from any feature set.
DESIGN_COLS = ["Stratum", "PSU", "Weight"]

# Fields that are coded by the same NEISS abstractor who assigns the product
# code. They are predictive in-sample but will not necessarily exist on the
# workers'-compensation test set, so they are flagged separately.
COABSTRACTED_COLS = [
    "Body_Part", "Body_Part_2", "Diagnosis", "Diagnosis_2",
    "Other_Diagnosis", "Other_Diagnosis_2", "Disposition", "Location",
    "Fire_Involvement", "Alcohol", "Drug",
]


# --------------------------------------------------------------------------
# Small formatting helpers (kept dependency-free: no tabulate, no matplotlib)
# --------------------------------------------------------------------------

def md_table(df: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table."""
    d = df.reset_index() if index else df.copy()
    cols = [str(c) for c in d.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in row] for row in d.itertuples(index=False)]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def anchor(heading: str) -> str:
    """GitHub-style anchor slug for a markdown heading.

    GitHub lowercases, drops punctuation other than hyphens and underscores,
    and turns spaces into hyphens. Getting this wrong silently produces a
    table-of-contents link that scrolls nowhere, so it is worth doing properly
    rather than stripping a hand-picked set of characters.
    """
    slug = heading.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)   # drop . : ? ( ) , ' etc.
    return re.sub(r"\s+", "-", slug.strip())


def pct(x: float) -> str:
    return "{:.2f}%".format(100 * x)


def thousands(n) -> str:
    return "{:,}".format(int(n))


def counts_table(counts: pd.Series, key_name: str, total: Optional[int] = None) -> pd.DataFrame:
    """Turn a value_counts() result into a labelled, formatted table.

    Built by hand rather than via reset_index(), whose column naming differs
    between pandas versions.
    """
    d = pd.DataFrame({key_name: [str(k) for k in counts.index],
                      "cases": [thousands(v) for v in counts.values]})
    if total:
        d["share"] = [pct(v / total) for v in counts.values]
    return d


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_data(path: str, use_cache: bool = True):
    """Load the NEISS data sheet and its NEISS_FMT codebook sheet.

    Everything is read as str: product codes are identifiers, not quantities,
    and reading them as ints silently drops the distinction between the code
    0 ("no product") and a missing cell.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = os.path.splitext(os.path.basename(path))[0]
    data_cache = os.path.join(CACHE_DIR, tag + "_data.pkl")
    fmt_cache = os.path.join(CACHE_DIR, tag + "_fmt.pkl")

    if use_cache and os.path.exists(data_cache) and os.path.exists(fmt_cache):
        return pd.read_pickle(data_cache), pd.read_pickle(fmt_cache)

    if not os.path.exists(path):
        sys.exit(
            "Data file not found: {}\n"
            "Download the annual file from the CPSC NEISS query tool\n"
            "  https://www.cpsc.gov/cgibin/NEISSQuery/home.aspx\n"
            "  -> Archived Annual NEISS Data -> pick a year -> Excel\n"
            "and save it into data/raw/.".format(path)
        )

    xl = pd.ExcelFile(path)
    data_sheet = next((s for s in xl.sheet_names if s.upper().startswith("NEISS")
                       and "FMT" not in s.upper()), xl.sheet_names[0])
    fmt_sheet = next((s for s in xl.sheet_names if "FMT" in s.upper()), None)

    print("reading {} (sheet '{}') ...".format(os.path.basename(path), data_sheet),
          file=sys.stderr)
    df = xl.parse(data_sheet, dtype=str)
    fmt = xl.parse(fmt_sheet, dtype=str) if fmt_sheet else pd.DataFrame()

    df.to_pickle(data_cache)
    fmt.to_pickle(fmt_cache)
    return df, fmt


def product_labels(fmt: pd.DataFrame) -> Dict[str, str]:
    """Map product code -> human-readable product name from the NEISS_FMT sheet."""
    if fmt.empty:
        return {}
    f = fmt.copy()
    f.columns = ["fmt_name", "start", "end", "label"][:len(f.columns)]
    f = f[f["fmt_name"].astype(str).str.strip().str.upper() == "PROD"]
    codes = f["start"].astype(str).str.strip()
    # Labels arrive as "464 - KNIVES, NOT ELSEWHERE CLASSIFIED"; drop the prefix.
    names = f["label"].astype(str).str.replace(r"^\d+\s*-\s*", "", regex=True).str.strip()
    return dict(zip(codes, names))


# --------------------------------------------------------------------------
# Section 1 — dataset shape and completeness
# --------------------------------------------------------------------------

def section_overview(df: pd.DataFrame, path: str) -> str:
    rows = []
    for c in df.columns:
        s = df[c]
        if c in DESIGN_COLS:
            role = "survey design - we don't use this"
        elif c in COABSTRACTED_COLS:
            role = "co-abstracted - may be absent at test time"
        elif c in PRODUCT_COLS:
            role = "LABEL"
        elif c == "Narrative_1":
            role = "PRIMARY FEATURE"
        else:
            role = "identifier / demographic"
        rows.append({
            "field": c,
            "non-null": thousands(s.notna().sum()),
            "% populated": pct(s.notna().mean()),
            "distinct": thousands(s.nunique()),
            "role": role,
        })

    dates = pd.to_datetime(df["Treatment_Date"], errors="coerce")
    body = [
        "**Source file:** `{}`".format(os.path.relpath(path, REPO)),
        "",
        "- **Rows:** {}".format(thousands(len(df))),
        "- **Columns:** {}".format(len(df.columns)),
        "- **Treatment dates:** {} to {}".format(
            dates.min().date() if dates.notna().any() else "?",
            dates.max().date() if dates.notna().any() else "?"),
        "",
        "Note: `% populated` counts non-empty cells only. Several NEISS fields "
        "use a sentinel code rather than a blank for \"unknown\" (`Location = 0`, "
        "`Product_2 = 0`), so a field can be 100% populated and still carry "
        "almost no information. Section 4 handles that for the product slots.",
        "",
        md_table(pd.DataFrame(rows)),
    ]
    return "\n".join(body)


# --------------------------------------------------------------------------
# Section 2 — narrative length
# --------------------------------------------------------------------------

def section_length(nar: pd.Series) -> str:
    chars = nar.str.len()
    words = nar.str.split().str.len()

    qs = [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    stat = pd.DataFrame({
        "percentile": ["min", "p1", "p5", "p25", "median", "p75", "p90", "p95", "p99", "max"],
        "characters": [int(chars.quantile(q)) for q in qs],
        "words": [int(words.quantile(q)) for q in qs],
    })

    edges = list(range(0, 425, 25))
    hist = pd.cut(chars, bins=edges).value_counts().sort_index()
    hmax = hist.max()
    hrows = [{"char range": "{}-{}".format(iv.left, iv.right),
              "cases": thousands(n),
              "share": pct(n / len(nar)),
              "distribution": "#" * int(round(40 * n / hmax))}
             for iv, n in hist.items()]

    at_cap = (chars >= 399).sum()
    body = [
        md_table(stat),
        "",
        "- Mean: **{:.1f} characters / {:.1f} words**".format(chars.mean(), words.mean()),
        "- At or near the 400-char cap: **{} ({})** - truncation is not a "
        "practical concern".format(thousands(at_cap), pct(at_cap / len(nar))),
        "- Under 40 characters: **{} ({})**".format(
            thousands((chars < 40).sum()), pct((chars < 40).mean())),
        "",
        "**Why this matters:** at a median of 21 words, every narrative fits "
        "comfortably in GLiNER's input window, so no narrative needs to be "
        "split or truncated before extraction. Run time will be "
        "driven by the number of cases, not by document length, so batching "
        "narratives through the model is the lever that matters.",
        "",
        "#### Distribution",
        "",
        md_table(pd.DataFrame(hrows)),
    ]
    return "\n".join(body)


# --------------------------------------------------------------------------
# Section 3 — narrative structure
# --------------------------------------------------------------------------

DEMOG_RE = r"^\s*\d{1,3}\s*(?:YOF|YOM|YO F|YO M|MOF|MOM|YO|MO)\b"

MARKERS = [
    ("DX: (explicit diagnosis marker)", r"\bDX\s*:"),
    ("DX without a colon", r"\bDX\b(?!\s*:)"),
    ("C/O (complains of)", r"\bC\s*/\s*O\b"),
    ("S/P (status post)", r"\bS\s*/\s*P\b"),
    ("PT (patient)", r"\bPT\b"),
    ("FX (fracture)", r"\bFX\b"),
    ("LAC (laceration)", r"\bLAC\b"),
    ("*** (redacted PII)", r"\*\*\*"),
    ("FELL", r"\bFELL\b"),
    ("STRUCK BY", r"\bSTRUCK BY\b"),
]


def section_structure(nar: pd.Series) -> str:
    has_demog = nar.str.contains(DEMOG_RE, regex=True, na=False)
    demog_kind = (nar.str.extract(r"^\s*\d{1,3}\s*(YOF|YOM|MOF|MOM|YO|MO)\b")[0]
                  .value_counts())

    stripped = nar.str.replace(DEMOG_RE + r"[\s,.-]*", "", regex=True)
    openers = stripped.str.extract(r"^([A-Za-z/']+)")[0].str.upper().value_counts().head(15)

    mrows = [{"pattern": name,
              "cases": thousands(nar.str.contains(p, regex=True, na=False).sum()),
              "share": pct(nar.str.contains(p, regex=True, na=False).mean())}
             for name, p in MARKERS]

    alpha = nar.str.replace(r"[^A-Za-z]", "", regex=True)
    all_upper = (alpha.str.count(r"[A-Z]") == alpha.str.len()).mean()

    # How much text sits after the DX marker, i.e. describes the injury not the event.
    has_dx = nar.str.contains(r"\bDX\s*:", regex=True, na=False)
    dx_tail = nar[has_dx].str.extract(r"\bDX\s*:\s*(.*)$")[0].fillna("")

    body = [
        "These narratives are formulaic, not free prose. Most follow the shape:",
        "",
        "```",
        "  <age><sex>  <event description>  DX: <diagnosis>",
        "     47YOF      FELL OFF A LADDER   DX: WRIST FRACTURE",
        "  \\_______/  \\_________________/  \\__________________/",
        "   strip it     THE OBJECT IS HERE   injury, not object",
        "```",
        "",
        "#### Leading demographic token",
        "",
        "- Narratives opening with an age/sex token: **{} ({})**".format(
            thousands(has_demog.sum()), pct(has_demog.mean())),
        "",
        md_table(counts_table(demog_kind, "token", len(nar))),
        "",
        "#### First word after the demographic prefix",
        "",
        md_table(counts_table(openers, "word", len(nar))),
        "",
        "#### Clinical shorthand and structural markers",
        "",
        md_table(pd.DataFrame(mrows)),
        "",
        "#### Casing and noise",
        "",
        "- Fully uppercase narratives: **{}**. Most pretrained NER models are "
        "trained on mixed case and degrade badly on all-caps input, so "
        "case-normalization is worth testing as a preprocessing step.".format(pct(all_upper)),
        "- Median sentence count: **{:.0f}**".format(nar.str.count(r"[.!?]").median()),
        "- Text after `DX:` runs a median of **{:.0f} characters**. This segment "
        "describes the *injury*, not the object, and contains bait words like "
        "FALL and FOREIGN BODY. Consider splitting it off and feeding the model "
        "the event clause only.".format(dx_tail.str.len().median()),
        "- Unique narratives: **{} of {}**. The repeats are short generic cases "
        "(e.g. teenagers spraining ankles at basketball), not data "
        "corruption.".format(thousands(nar.nunique()), thousands(len(nar))),
        "",
    ] + preprocessing_summary(nar)
    return "\n".join(body)


def preprocessing_summary(nar: pd.Series) -> List[str]:
    """Report what the shared Task #6 rules (src/preprocess.py) change."""
    ws = clean_whitespace(nar)
    norm = normalize_abbreviations(ws)

    rows = [{"step": "Collapse repeated whitespace",
             "narratives changed": thousands((nar != ws).sum()),
             "share": pct((nar != ws).mean())}]
    for shown, pattern, expansion in ABBREVIATIONS:
        hit = ws.str.contains(pattern, regex=True)
        rows.append({"step": "{} -> {}".format(shown, expansion.strip()),
                     "narratives changed": thousands(hit.sum()),
                     "share": pct(hit.mean())})
    changed = ws != norm
    rows.append({"step": "Any abbreviation expanded",
                 "narratives changed": thousands(changed.sum()),
                 "share": pct(changed.mean())})
    has_dx = split_diagnosis(norm)["diagnosis"] != ""
    rows.append({"step": "Diagnosis split off into its own column",
                 "narratives changed": thousands(has_dx.sum()),
                 "share": pct(has_dx.mean())})
    red = has_redaction(nar)
    rows.append({"step": "Flagged as containing a redaction (*)",
                 "narratives changed": thousands(red.sum()),
                 "share": pct(red.mean())})

    example = ""
    if changed.any():
        i = changed.idxmax()
        example = "\n".join(["", "Example:", "",
                             "- before: `{}`".format(ws[i][:120]),
                             "- after:  `{}`".format(norm[i][:120])])

    return [
        "#### Preprocessing applied (Task #6)",
        "",
        "The cleaning rules live in `src/preprocess.py`, shared by the "
        "notebook, this report and the model code, so Marsh's narratives are "
        "cleaned exactly like the training data. The raw `Narrative_1` is "
        "never modified; the results are the `narrative_normalized`, "
        "`event_text`, `diagnosis_text` and `redacted` columns in "
        "`data/processed/neiss2025_clean.csv`.",
        "",
        md_table(pd.DataFrame(rows)),
        example,
        "",
        "Deliberately left unchanged: age/sex codes (`YOM`, `YOF`, `MOM`, "
        "`MOF`), which are structured conventions (`MOM` can mean "
        "\"month-old male\" or \"mother\"); `DX`, kept as the split marker; "
        "and `R`/`L`, `INJ` and `LOC`, which have more than one meaning. Text is "
        "tokenized without stop-word removal or stemming, since GLiNER works "
        "on natural text.",
    ]


# --------------------------------------------------------------------------
# Section 4 — label structure
# --------------------------------------------------------------------------

def section_labels(df: pd.DataFrame, pmap: Dict[str, str]) -> str:
    slots = {c: df[c].astype(str).str.strip() for c in PRODUCT_COLS}

    occ = [{"slot": c,
            "coded (non-zero)": thousands((s != NO_PRODUCT).sum()),
            "share of cases": pct((s != NO_PRODUCT).mean()),
            "distinct codes": thousands(s[s != NO_PRODUCT].nunique())}
           for c, s in slots.items()]

    n_coded = sum((s != NO_PRODUCT).astype(int) for s in slots.values())
    per_case = n_coded.value_counts().sort_index()
    pcrows = [{"products coded": int(k), "cases": thousands(v), "share": pct(v / len(df))}
              for k, v in per_case.items()]

    stacked = pd.concat(list(slots.values()))
    stacked = stacked[stacked != NO_PRODUCT]
    vc = stacked.value_counts()
    cum = vc.cumsum() / vc.sum()

    conc = [{"top N codes": n, "share of all product mentions": pct(cum.iloc[n - 1])}
            for n in [10, 25, 50, 100, 200, 300, 500] if n <= len(cum)]

    tail = [{"mentions per code": lab,
             "number of codes": thousands(cnt)}
            for lab, cnt in [(">= 1000", (vc >= 1000).sum()),
                             ("100 - 999", ((vc >= 100) & (vc < 1000)).sum()),
                             ("10 - 99", ((vc >= 10) & (vc < 100)).sum()),
                             ("< 10", (vc < 10).sum())]]

    top = vc.head(25)
    toprows = [{"code": c, "product": pmap.get(c, "?"), "mentions": thousands(n),
                "share": pct(n / vc.sum())} for c, n in top.items()]

    # Ordered pairs vs unordered sets: are the slots meaningfully ordered?
    multi = df[slots["Product_2"] != NO_PRODUCT]
    a = multi["Product_1"].astype(str).str.strip()
    b = multi["Product_2"].astype(str).str.strip()
    ordered = (a + "+" + b).nunique()
    unordered = pd.Series([" + ".join(sorted(p)) for p in zip(a, b)]).nunique()

    pair_counts = pd.Series([" + ".join(sorted(p)) for p in zip(a, b)]).value_counts().head(10)
    pairrows = []
    for key, n in pair_counts.items():
        x, y = key.split(" + ")
        pairrows.append({"cases": thousands(n),
                         "product A": pmap.get(x, x)[:40],
                         "product B": pmap.get(y, y)[:40]})

    body = [
        "The label for \"which object was involved\" already exists: it is the "
        "`Product_1/2/3` triple, recorded as codes from the NEISS product "
        "taxonomy. The `NEISS_FMT` sheet carries **{}** code definitions; "
        "**{}** of them are actually used in this file.".format(
            thousands(len(pmap)), thousands(len(vc))),
        "",
        "#### Slot occupancy",
        "",
        md_table(pd.DataFrame(occ)),
        "",
        md_table(pd.DataFrame(pcrows)),
        "",
        "**Implication:** {} of cases carry exactly one product, so a "
        "single-label classifier is a legitimate v1 and multi-label is the "
        "refinement.".format(pct(per_case.get(1, 0) / len(df))),
        "",
        "#### Label concentration",
        "",
        md_table(pd.DataFrame(conc)),
        "",
        md_table(pd.DataFrame(tail)),
        "",
        "**Implication:** the distribution is steep. A model that only ever "
        "predicts the top 25 codes would already touch most mentions, which "
        "makes micro-F1 flattering and macro-F1 the honest metric. The "
        "**{}** codes with fewer than 10 examples cannot be learned from this "
        "file alone - decide deliberately whether to merge them into an OTHER "
        "bucket or pool several years of data.".format(thousands((vc < 10).sum())),
        "",
        "#### Most frequent products",
        "",
        md_table(pd.DataFrame(toprows)),
        "",
        "#### Are the slots ordered?",
        "",
        "- Distinct **ordered** (Product_1, Product_2) pairs: **{}**".format(thousands(ordered)),
        "- Distinct **unordered** pairs: **{}**".format(thousands(unordered)),
        "",
        "The gap means abstractors disagree about which object goes in slot 1 - "
        "BED + FLOOR and FLOOR + BED describe the same event. **Evaluate "
        "predictions as a set, not positionally**, or the model gets punished "
        "for the coders' inconsistency rather than for being wrong.",
        "",
        md_table(pd.DataFrame(pairrows)),
    ]
    return "\n".join(body)


# --------------------------------------------------------------------------
# Section 5 — is the object actually named in the narrative?
# --------------------------------------------------------------------------

STOPWORDS = {
    "OR", "AND", "NOT", "SPECIFIED", "OTHER", "NEC", "ELSEWHERE", "CLASSIFIED",
    "EXCL", "EXCLUDING", "ACTIVITY", "APPAREL", "EQUIPMENT", "RELATED", "THE",
    "OF", "WITH", "WITHOUT", "USE", "ITEMS", "PART", "UNSPECIFIED", "INCL",
    "INCLUDING", "EXCEPT", "NON", "TYPE", "MATERIALS",
}

# Codes whose taxonomy label uses a word clinicians never type. Auto-deriving
# keywords from the label fails for these, so the patterns are curated. Each
# entry is the vocabulary an ED nurse would actually use for that product.
CURATED = {
    "464":  r"\bKNI(?:FE|VES)\b|\bBLADE\b",
    "679":  r"\bCOUCH(?:ES)?\b|\bSOFA(?:S)?\b|\bRECLINER\b|\bDIVAN\b|\bLOVESEAT\b",
    "1615": r"\bSHOE(?:S)?\b|\bBOOT(?:S)?\b|\bSANDAL(?:S)?\b|\bSLIPPER(?:S)?\b|\bCLOG(?:S)?\b|\bHEEL(?:S)?\b|\bFLIP.?FLOP|\bFOOTWEAR\b|\bSHOELACE",
    "1616": r"\bJEWEL|\bRING(?:S)?\b|\bNECKLACE|\bEARRING|\bBRACELET|\bPIERCING",
    "3299": r"\bEXERCIS|\bWORK(?:ING)?.?OUT\b|\bGYM\b|\bRUNNING\b|\bJOGG|\bHIK(?:E|ING)\b|\bYOGA\b|\bSTRETCH",
    "5040": r"\bBIKE(?:S)?\b|\bBICYCLE(?:S)?\b|\bBIKING\b|\bCYCLING\b",
    "4056": r"\bCABINET(?:S)?\b|\bSHEL(?:F|VES)\b|\bRACK(?:S)?\b|\bDRESSER\b|\bDIVIDER",
    "1807": r"\bFLOOR|\bTILE(?:S)?\b|\bHARDWOOD\b|\bLINOLEUM\b|\bCONCRETE\b",
}


def word_variants(word: str) -> List[str]:
    """Surface forms of a keyword, handling the irregular plurals in NEISS labels.

    KNIVES -> KNIFE, COUCHES -> COUCH, BERRIES -> BERRY. Without this, naive
    "strip the trailing S" matching reports near-zero overlap for these codes
    and badly understates how often the object is named.
    """
    w = word.upper()
    forms = {w}
    if w.endswith("VES"):
        forms.update({w[:-3] + "FE", w[:-3] + "F"})
    if w.endswith("IES"):
        forms.add(w[:-3] + "Y")
    if w.endswith(("SSES", "CHES", "SHES", "XES", "ZES")):
        forms.add(w[:-2])
    if w.endswith("S") and len(w) > 3:
        forms.add(w[:-1])
    else:
        forms.add(w + "S")
    return sorted(forms)


def keyword_pattern(code: str, name: str) -> Optional[str]:
    """Regex matching any plausible mention of the product named by `name`."""
    if code in CURATED:
        return CURATED[code]
    toks = [t for t in re.findall(r"[A-Z]{3,}", name.upper()) if t not in STOPWORDS]
    if not toks:
        return None
    alts = []
    for t in toks[:4]:
        alts += word_variants(t)
    return "|".join(r"\b{}\b".format(a) for a in sorted(set(alts)))


def section_linkage(df: pd.DataFrame, pmap: Dict[str, str], top_n: int = 25) -> str:
    p1 = df["Product_1"].astype(str).str.strip()
    top = p1.value_counts().head(top_n)

    rows = []
    for code, n in top.items():
        name = pmap.get(code, "?")
        pat = keyword_pattern(code, name)
        if not pat:
            continue
        sub = df.loc[p1 == code, "Narrative_1"]
        hit = sub.str.contains(pat, regex=True, na=False).mean()
        rows.append({
            "code": code,
            "product": name[:44],
            "cases": thousands(n),
            "narrative names it": pct(hit),
            "matching": "curated" if code in CURATED else "auto",
            "_sort": hit,
        })
    # Sort on the numeric rate, not the formatted string, so rates below 10%
    # do not sort above rates above 10%.
    tbl = pd.DataFrame(rows).sort_values("_sort").drop(columns="_sort")

    # Examples where the object is genuinely inferred rather than stated.
    examples = []
    for code in ["464", "3299"]:
        name = pmap.get(code, "?")
        pat = keyword_pattern(code, name)
        sub = df.loc[p1 == code, "Narrative_1"]
        miss = sub[~sub.str.contains(pat, regex=True, na=False)]
        examples.append("**Code {} = {}** - {} of {} cases never name it:".format(
            code, name, thousands(len(miss)), thousands(len(sub))))
        examples += ["  - `{}`".format(s[:150]) for s in miss.head(3)]
        examples.append("")

    body = [
        "This is the feasibility question for the whole project: when an "
        "abstractor assigned a product code, is that object actually written "
        "in the narrative?",
        "",
        "Method: for each of the top {} products, match the narrative against "
        "keywords drawn from the taxonomy label, with inflection handling. "
        "Eight codes use taxonomy vocabulary that clinicians never type "
        "(FOOTWEAR vs \"shoe\", JEWELRY vs \"ring\") and carry a curated "
        "pattern instead - marked `curated` below. This is an approximate "
        "lower bound, not an exact measurement.".format(top_n),
        "",
        md_table(tbl),
        "",
        "**Two different problems are mixed together here:**",
        "",
        "1. **Vocabulary mismatch** - the object is named, but not with the "
        "taxonomy's word. A lexicon or embedding lookup solves this. It is "
        "why the `curated` rows exist.",
        "2. **Genuine inference** - the object is never named and a human "
        "coder inferred it from the activity. No lexicon solves this; it needs "
        "a model with world knowledge. Expect GLiNER to miss most of these, "
        "since it extracts spans that are actually in the text.",
        "",
        "Examples of category 2:",
        "",
    ] + examples
    return "\n".join(body)


# --------------------------------------------------------------------------
# Section 6 — domain gap
# --------------------------------------------------------------------------

# Objects a workers'-compensation narrative is likely to mention. Used to test
# whether the gap is a *label space* problem (taxonomy lacks the code) or a
# *training distribution* problem (code exists but is rare here). These are very
# different problems: the second is fixable by sampling, the first is not.
WORKPLACE_PROBES = [
    "FORK LIFT", "SCAFFOLD", "LADDER", "NAIL GUN", "WELDING", "CHAIN SAW",
    "POWER SAW", "POWER DRILL", "GRINDER", "SANDER", "AIR COMPRESSOR",
    "TRACTOR", "HOIST", "CONVEYOR", "PALLET", "CRANE",
]


def probe_workplace_taxonomy(df: pd.DataFrame, pmap: Dict[str, str]) -> pd.DataFrame:
    """For each workplace object, does a code exist and how many cases use it?"""
    stacked = pd.concat([df[c].astype(str).str.strip() for c in PRODUCT_COLS])
    counts = stacked[stacked != NO_PRODUCT].value_counts()

    rows = []
    for probe in WORKPLACE_PROBES:
        matches = [c for c, n in pmap.items() if probe.replace(" ", "") in
                   re.sub(r"[^A-Z]", "", n.upper())]
        if not matches:
            rows.append({"workplace object": probe.title(), "codes in taxonomy": "NONE",
                         "cases in this file": "-"})
            continue
        total = sum(int(counts.get(c, 0)) for c in matches)
        rows.append({"workplace object": probe.title(),
                     "codes in taxonomy": str(len(matches)),
                     "cases in this file": thousands(total)})
    return pd.DataFrame(rows)


def section_domain_gap(df: pd.DataFrame, pmap: Dict[str, str]) -> str:
    loc = df["Location"].astype(str).str.strip()
    counts = loc.value_counts().sort_index()
    loc_names = {"0": "Not recorded", "1": "Home", "2": "Farm/Ranch",
                 "3": "Apartment", "4": "Street or highway",
                 "5": "Other public property", "6": "Mobile/manufactured home",
                 "7": "Industrial", "8": "School/daycare",
                 "9": "Place of recreation or sports"}
    lrows = [{"code": c, "location": loc_names.get(c, "?"),
              "cases": thousands(n), "share": pct(n / len(df))}
             for c, n in counts.items()]

    ind = df[loc == "7"]
    work_words = r"\bAT WORK\b|\bWORKPLACE\b|\bON THE JOB\b|\bJOB SITE\b|\bWORKSITE\b"
    work_mention = df["Narrative_1"].str.contains(work_words, regex=True, na=False)

    body = [
        "The model trains on consumer-product injuries but will be tested on "
        "workers'-compensation narratives. That gap is the single largest risk "
        "in the project, and it is measurable.",
        "",
        "#### Incident location",
        "",
        md_table(pd.DataFrame(lrows)),
        "",
        "- Cases coded **Industrial**: **{}** ({})".format(
            thousands(len(ind)), pct(len(ind) / len(df))),
        "- Narratives mentioning work explicitly (\"at work\", \"workplace\", "
        "\"on the job\"): **{}** ({})".format(
            thousands(work_mention.sum()), pct(work_mention.mean())),
        "",
        "**This is by design, not a data error.** Public NEISS is a *consumer "
        "product* surveillance system and largely excludes occupational "
        "injuries. Occupational cases live in **NEISS-Work**, the separate "
        "NIOSH occupational supplement, which the project brief lists as a "
        "stretch goal.",
        "",
        "#### But is the *label space* also missing, or only the examples?",
        "",
        "The `Location` field above understates how usable this data is. The "
        "NEISS product taxonomy turns out to cover workplace objects well - "
        "what is thin is the number of training examples, not the vocabulary.",
        "",
        md_table(probe_workplace_taxonomy(df, pmap)),
        "",
        "Read that table in two columns, because they say different things:",
        "",
        "- **`codes in taxonomy`** - the label space is in good shape. Only "
        "PALLET and CRANE have no code at all. Ladders get seven codes, "
        "grinders nine. NEISS can express most workers'-comp answers.",
        "- **`cases in this file`** - the *examples* are thin to absent. "
        "Ladders are fine at ~3,900. But FORK LIFT and CONVEYOR have a code and "
        "**zero cases in this year**, sanders and air compressors sit in the "
        "dozens, and scaffolding is just over a hundred. You cannot learn a class from zero "
        "examples no matter how good the model is.",
        "",
        "So this is mostly a **training-distribution** problem rather than a "
        "**label-space** problem, which is the better of the two to have: a "
        "missing label space would mean the taxonomy cannot express the answer "
        "and the project needs a different target, whereas a skewed "
        "distribution has ordinary fixes. The zero-case rows are the real "
        "warning, and they are the concrete argument for pooling years or "
        "getting NEISS-Work.",
        "",
        "Mitigations, in rough order of effort:",
        "",
        "1. **Evaluate on a workplace-like slice**, not a random split, so the "
        "validation number tracks the real target rather than flattering it "
        "with bedroom falls.",
        "2. **Oversample or reweight** the tool, machinery and structural codes "
        "identified above.",
        "3. **Pool several years** of NEISS. Rare workplace codes gain roughly "
        "linearly with each year added, at no annotation cost.",
        "4. **Obtain NEISS-Work** and treat it as the primary training set - "
        "the highest-value option, and the one that needs the advisor's help.",
    ]
    return "\n".join(body)


# --------------------------------------------------------------------------
# Section 7 — class imbalance in the target
# --------------------------------------------------------------------------

def section_imbalance(df: pd.DataFrame, pmap: Dict[str, str]) -> str:
    p1 = df["Product_1"].astype(str).str.strip()
    vc = p1.value_counts()

    tiers = [{"class size": lab, "classes": thousands(n), "cases covered": thousands(c)}
             for lab, n, c in [
                 ("1 case only", (vc == 1).sum(), vc[vc == 1].sum()),
                 ("2-4 cases", ((vc >= 2) & (vc < 5)).sum(), vc[(vc >= 2) & (vc < 5)].sum()),
                 ("5-19 cases", ((vc >= 5) & (vc < 20)).sum(), vc[(vc >= 5) & (vc < 20)].sum()),
                 ("20-99 cases", ((vc >= 20) & (vc < 100)).sum(), vc[(vc >= 20) & (vc < 100)].sum()),
                 ("100-999 cases", ((vc >= 100) & (vc < 1000)).sum(), vc[(vc >= 100) & (vc < 1000)].sum()),
                 ("1000+ cases", (vc >= 1000).sum(), vc[vc >= 1000].sum()),
             ]]

    folds = [{"scheme": "{}-fold stratified".format(k),
              "classes with too few examples": thousands((vc < k).sum()),
              "cases in those classes": thousands(vc[vc < k].sum()),
              "% of data at risk": pct(vc[vc < k].sum() / len(df))}
             for k in [3, 5, 10]]

    body = [
        "The target is `Product_1`: **{}** distinct classes in this file.".format(thousands(len(vc))),
        "",
        "- Largest class: **{}** ({} cases, {} of the data)".format(
            pmap.get(vc.index[0], vc.index[0]), thousands(vc.iloc[0]), pct(vc.iloc[0] / len(df))),
        "- Smallest class: **{} case(s)**".format(thousands(vc.iloc[-1])),
        "- Imbalance ratio: **{}:1**".format(thousands(vc.iloc[0] / vc.iloc[-1])),
        "",
        md_table(pd.DataFrame(tiers)),
        "",
        "#### What this rules out",
        "",
        "Stratified k-fold requires at least k examples of every class. That is "
        "not satisfied here:",
        "",
        md_table(pd.DataFrame(folds)),
        "",
        "The good news is in the last column: the classes that break "
        "stratification are collectively **well under 1% of the data**. So the "
        "fix is cheap - you are not throwing away signal, you are tidying up a "
        "tail that could never have been learned from one year anyway.",
        "",
        "**Recommended handling, in order of preference:**",
        "",
        "1. **Merge** classes below a threshold (say 20 cases) into an `OTHER` "
        "bucket. Keeps every row, makes stratification legal, and `OTHER` "
        "becomes an honest \"I don't know\" class that is genuinely useful in "
        "production.",
        "2. **Report macro-F1 twice** - once over all classes, once over only "
        "classes above the threshold. The gap tells your advisor how much of "
        "the score is tail noise.",
        "3. **Do not simply drop** rare rows without saying so. Dropping the "
        "hard cases and reporting the resulting accuracy is the single easiest "
        "way to produce a number that does not survive contact with the test "
        "set.",
    ]
    return "\n".join(body)


# --------------------------------------------------------------------------
# Section 8 — hidden confounders
# --------------------------------------------------------------------------

def _tvd(part: pd.Series, reference: pd.Series) -> float:
    """Total-variation distance between two label distributions."""
    a = part.value_counts(normalize=True).reindex(reference.index, fill_value=0.0)
    return 0.5 * float(np.abs(a - reference).sum())


def section_confounders(df: pd.DataFrame, pmap: Dict[str, str]) -> str:
    p1 = df["Product_1"].astype(str).str.strip()
    psu = df["PSU"].astype(str).str.strip()
    nar = df["Narrative_1"].fillna("")
    dates = pd.to_datetime(df["Treatment_Date"], errors="coerce")

    # --- hospital writing style -------------------------------------------
    style = pd.DataFrame({
        "cases": psu.value_counts(),
        "mean_chars": nar.groupby(psu).apply(lambda s: s.str.len().mean()),
        "dx_rate": nar.groupby(psu).apply(lambda s: s.str.contains(r"\bDX\s*:", regex=True).mean()),
        "co_rate": nar.groupby(psu).apply(lambda s: s.str.contains(r"\bC\s*/\s*O\b", regex=True).mean()),
    })
    srows = [
        {"style marker": "mean narrative length (chars)",
         "min hospital": "{:.0f}".format(style.mean_chars.min()),
         "median hospital": "{:.0f}".format(style.mean_chars.median()),
         "max hospital": "{:.0f}".format(style.mean_chars.max())},
        {"style marker": "uses 'DX:' marker",
         "min hospital": pct(style.dx_rate.min()),
         "median hospital": pct(style.dx_rate.median()),
         "max hospital": pct(style.dx_rate.max())},
        {"style marker": "uses 'C/O' shorthand",
         "min hospital": pct(style.co_rate.min()),
         "median hospital": pct(style.co_rate.median()),
         "max hospital": pct(style.co_rate.max())},
    ]

    # --- hospital label mix -----------------------------------------------
    glob = p1.value_counts(normalize=True)
    tvd = pd.Series({h: _tvd(p1[psu == h], glob) for h in psu.unique()})

    hc = psu.value_counts()
    conc = [
        {"group": "largest hospital", "cases": thousands(hc.iloc[0]),
         "share of data": pct(hc.iloc[0] / len(df))},
        {"group": "top 5 hospitals", "cases": thousands(hc.head(5).sum()),
         "share of data": pct(hc.head(5).sum() / len(df))},
        {"group": "top 10 hospitals", "cases": thousands(hc.head(10).sum()),
         "share of data": pct(hc.head(10).sum() / len(df))},
        {"group": "smallest 20 hospitals", "cases": thousands(hc.tail(20).sum()),
         "share of data": pct(hc.tail(20).sum() / len(df))},
    ]

    # --- seasonality -------------------------------------------------------
    month = dates.dt.month
    srows2 = []
    for c in p1.value_counts().head(40).index:
        m = month[p1 == c].value_counts(normalize=True).reindex(range(1, 13), fill_value=0)
        srows2.append({"product": pmap.get(c, c)[:38], "cases": int((p1 == c).sum()),
                       "peak month": int(m.idxmax()), "peak share": pct(m.max()),
                       "_idx": m.max() * 12})
    sea = pd.DataFrame(srows2).sort_values("_idx", ascending=False)
    sea["seasonality index"] = sea["_idx"].map("{:.1f}x".format)
    sea = sea.drop(columns="_idx")
    sea["cases"] = sea["cases"].map(thousands)

    # --- duplicates --------------------------------------------------------
    dup = nar.duplicated(keep=False)
    dd = pd.DataFrame({"nar": nar[dup], "p1": p1[dup]})
    agree = dd.groupby("nar")["p1"].nunique()
    conflicts = agree[agree > 1]

    conf_examples = []
    for t in conflicts.index[:3]:
        labs = dd[dd.nar == t]["p1"].unique()
        conf_examples.append("  - `{}`".format(t[:110]))
        conf_examples.append("    labelled both: {}".format(
            " / ".join("**{}**".format(pmap.get(l, l)[:36]) for l in labs)))

    body = [
        "Three things in this data correlate with the label but are not the "
        "text's meaning. Each one can inflate a validation score.",
        "",
        "### 8a. Hospital (`PSU`) - the big one",
        "",
        "NEISS narratives are written by abstractors at **{}** hospitals, and "
        "house style varies enormously:".format(thousands(psu.nunique())),
        "",
        md_table(pd.DataFrame(srows)),
        "",
        "Read the `DX:` row carefully: **some hospitals use that marker on "
        "every single case and others never use it at all.** Same for `C/O`. "
        "Average narrative length varies more than fivefold between sites.",
        "",
        "This matters because of *how the project will be graded*. The final "
        "test set is Marsh's workers'-compensation narratives - a completely "
        "different source with its own conventions. A model validated on a "
        "random split has seen every hospital's style during training, so its "
        "score says nothing about transferring to a new writing style. That is "
        "exactly the thing the project needs to predict.",
        "",
        "Hospitals also differ in *what* they see. Total-variation distance "
        "between one hospital's label mix and the national mix:",
        "",
        "- min **{:.2f}**, median **{:.2f}**, max **{:.2f}** (0 = identical, "
        "1 = no overlap)".format(tvd.min(), tvd.median(), tvd.max()),
        "",
        "And case volume is very unevenly distributed, which constrains *how* "
        "you can split by hospital:",
        "",
        md_table(pd.DataFrame(conc)),
        "",
        "**Consequence:** holding out a random 20% *of hospitals* could yield "
        "anywhere from ~2% to ~50% of the rows. Hospitals must be allocated to "
        "folds by case count, not at random. Section 10 does this.",
        "",
        "### 8b. Season",
        "",
        "Some products are strongly seasonal. The index below is peak-month "
        "share divided by the 1/12 you would expect if cases were spread "
        "evenly:",
        "",
        md_table(sea.head(10)),
        "",
        "**Consequence:** do **not** use a chronological split (train on "
        "Jan-Sep, test on Oct-Dec). It would put almost no swimming or "
        "baseball in the test set and an unrepresentative pile of football, "
        "measuring the calendar rather than the model. Split randomly or by "
        "hospital, and if you later pool multiple years, stratify by month.",
        "",
        "### 8c. Duplicate narratives",
        "",
        "- Rows whose narrative text is not unique: **{} ({})**, across **{}** "
        "distinct texts".format(thousands(dup.sum()), pct(dup.mean()),
                                thousands(nar[dup].nunique())),
        "- Of those texts, **{}** carry a single label and **{}** carry "
        "**conflicting** labels".format(
            thousands((agree == 1).sum()), thousands(len(conflicts))),
        "",
        "The volume is too small to distort a split much, but deduplicate "
        "across the train/validation boundary anyway - it is one line of code "
        "and removes a free-points effect.",
        "",
        "The conflicting cases are more interesting, because they put a "
        "**ceiling on achievable accuracy**: identical text, different gold "
        "label, so no model can get both right.",
        "",
    ] + conf_examples + [
        "",
        "The first is a genuinely ambiguous event (rolled off a bed onto a "
        "floor - which object caused the injury?). Expect a few points of "
        "irreducible error from this kind of coder disagreement, and say so "
        "when you report results rather than chasing it.",
    ]
    return "\n".join(body)


# --------------------------------------------------------------------------
# Section 9 — balancing exposure across incident contexts
# --------------------------------------------------------------------------

LOC_NAMES = {"0": "Not recorded", "1": "Home", "2": "Farm/Ranch", "3": "Apartment",
             "4": "Street or highway", "5": "Other public property",
             "6": "Mobile/manufactured home", "7": "Industrial",
             "8": "School/daycare", "9": "Recreation or sports"}


def _norm_entropy(s: pd.Series) -> float:
    """Shannon entropy of a categorical series, normalized to [0, 1]."""
    q = s.value_counts(normalize=True)
    if len(q) <= 1:
        return 0.0
    return float(-(q * np.log(q)).sum() / np.log(len(q)))


def section_exposure(df: pd.DataFrame, pmap: Dict[str, str]) -> str:
    p1 = df["Product_1"].astype(str).str.strip()
    loc = df["Location"].astype(str).str.strip()
    recorded = loc != "0"
    # Conditional vs marginal entropy of venue: how much does the product tell
    # us about where the incident happened?
    top = p1[recorded].value_counts().head(20).index
    ent_rows = []
    for c in top:
        l = loc[recorded & (p1 == c)]
        vcl = l.value_counts(normalize=True)
        ent_rows.append({"product": pmap.get(c, c)[:40], "cases": thousands(len(l)),
                         "dominant venue": LOC_NAMES.get(vcl.index[0], "?"),
                         "its share": pct(vcl.iloc[0]),
                         "venue entropy": "{:.2f}".format(_norm_entropy(l))})
    ent = pd.DataFrame(ent_rows)
    mean_cond = np.mean([_norm_entropy(loc[recorded & (p1 == c)]) for c in top])
    marginal = _norm_entropy(loc[recorded])

    body = [
        "Venue is largely *determined by* the product code. Measuring venue "
        "entropy within each product (0 = the product fixes the venue "
        "completely, 1 = the product tells you nothing):",
        "",
        md_table(ent),
        "",
        "- Mean venue entropy **within** a product: **{:.2f}**".format(mean_cond),
        "- Venue entropy across the data **overall**: **{:.2f}**".format(marginal),
    ]
    return "\n".join(body)


# --------------------------------------------------------------------------
# Section 10 — the recommended recipe, plus the split itself
# --------------------------------------------------------------------------

def assign_psu_splits(df: pd.DataFrame,
                      fracs: Tuple[float, float, float] = (0.70, 0.15, 0.15)
                      ) -> pd.DataFrame:
    """Allocate whole hospitals to train/val/test, balanced by case count.

    Every case from a given hospital lands in exactly one fold, so the
    validation score measures transfer to *unseen writing conventions* rather
    than memorization of house style. Hospitals are assigned largest-first to
    whichever fold is furthest below its target size (longest-processing-time
    scheduling), because hospital sizes are extremely uneven - assigning them
    at random gives wildly off-target fold sizes.
    """
    counts = df["PSU"].astype(str).str.strip().value_counts()
    total = int(counts.sum())
    names = ["train", "val", "test"]
    targets = {n: f * total for n, f in zip(names, fracs)}
    current = {n: 0 for n in names}
    assignment = {}

    for psu, n in counts.items():
        # deficit = how far below target this fold currently is, as a fraction
        pick = max(names, key=lambda k: (targets[k] - current[k]) / max(targets[k], 1))
        assignment[psu] = pick
        current[pick] += int(n)

    return pd.DataFrame({
        "PSU": list(assignment.keys()),
        "fold": [assignment[p] for p in assignment],
        "cases": [int(counts[p]) for p in assignment],
    }).sort_values(["fold", "cases"], ascending=[True, False])


def section_recipe(df: pd.DataFrame, pmap: Dict[str, str]) -> str:
    p1 = df["Product_1"].astype(str).str.strip()
    psu = df["PSU"].astype(str).str.strip()
    loc = df["Location"].astype(str).str.strip()

    splits = assign_psu_splits(df)
    fold_of = dict(zip(splits["PSU"], splits["fold"]))
    fold = psu.map(fold_of)

    srows = []
    for name in ["train", "val", "test"]:
        m = fold == name
        srows.append({
            "fold": name,
            "hospitals": thousands((splits.fold == name).sum()),
            "cases": thousands(m.sum()),
            "share": pct(m.mean()),
            "distinct products": thousands(p1[m].nunique()),
        })

    # Diagnostic: classes present in val/test but absent from train.
    train_classes = set(p1[fold == "train"].unique())
    unseen = {}
    for name in ["val", "test"]:
        cls = set(p1[fold == name].unique())
        missing = cls - train_classes
        n_rows = int(p1[(fold == name) & p1.isin(missing)].shape[0])
        unseen[name] = (len(missing), n_rows)

    # Diagnostic: does the grouped split preserve the venue mix (section 9)?
    vrows = []
    rec = loc != "0"
    for name in ["train", "val", "test"]:
        m = (fold == name) & rec
        vc = loc[m].value_counts(normalize=True)
        vrows.append({"fold": name,
                      "Home": pct(vc.get("1", 0)),
                      "Recreation/sports": pct(vc.get("9", 0)),
                      "Public property": pct(vc.get("5", 0)),
                      "School": pct(vc.get("8", 0)),
                      "Street": pct(vc.get("4", 0))})

    glob = p1.value_counts(normalize=True)
    drift = {n: _tvd(p1[fold == n], glob) for n in ["train", "val", "test"]}

    # Largest venue-share gap between any two folds, so the prose below states
    # what the split actually produced rather than what we hoped it would.
    venue_gap, venue_worst = 0.0, ""
    for label in ["Home", "Recreation/sports", "Public property", "School", "Street"]:
        vals = [float(r[label].rstrip("%")) for r in vrows]
        if max(vals) - min(vals) > venue_gap:
            venue_gap, venue_worst = max(vals) - min(vals), label

    body = [
        "Putting Sections 7-9 together, for the task **`Narrative_1` -> "
        "`Product_1`**.",
        "",
        "### Step 1 - Collapse the unlearnable tail",
        "",
        "Merge every `Product_1` class with fewer than **20 cases** into a "
        "single `OTHER` class. This affects {} classes but only {} of rows, "
        "and it is what makes stratification legal at all.".format(
            thousands((p1.value_counts() < 20).sum()),
            pct(p1.value_counts()[p1.value_counts() < 20].sum() / len(df))),
        "",
        "### Step 2 - Split by hospital",
        "",
        "Assign **whole hospitals** to folds, so no hospital's writing style "
        "appears in both training and validation.",
        "",
        "Because hospital sizes are so uneven (Section 8a), assign them "
        "largest-first to whichever fold is furthest below target rather than "
        "at random.",
        "",
        md_table(pd.DataFrame(srows)),
        "",
        "Label-mix drift from the national distribution (total-variation "
        "distance): train **{:.3f}**, val **{:.3f}**, test **{:.3f}**. Low "
        "values mean the grouped split did not accidentally skew the "
        "labels.".format(drift["train"], drift["val"], drift["test"]),
        "",
        "Venue mix across folds - the Section 9 diagnostic, computed on rows "
        "where `Location` is recorded:",
        "",
        md_table(pd.DataFrame(vrows)),
        "",
        "**This table shows the real cost of splitting by hospital.** The "
        "label mix stays close to national (the drift figures above are "
        "small), but the *venue* mix does not: **{}** differs by **{:.0f} "
        "percentage points** between folds. Hospitals serve different "
        "catchments - a paediatric or suburban site sees a different mix of "
        "home and sports injuries than an inner-city one - and holding out "
        "whole sites inherits that.".format(venue_worst, venue_gap),
        "",
        "This is a trade-off:",
        "",
        "- A **grouped** split buys independence from writing style and pays "
        "for it with venue drift.",
        "- A **stratified random** split holds venue and labels steady and pays "
        "for it by letting house style leak.",
        "",
        "",
        "### Step 3 - Also keep a stratified random split, as a control",
        "",
        "Build a second, ordinary split stratified on `Product_1`. Train the "
        "same model on both.",
        "",
        "- The **stratified** score is your optimistic ceiling.",
        "- The **grouped-by-hospital** score is your realistic estimate for "
        "Marsh's data.",
        "- **The gap between them is a result in its own right** - it "
        "quantifies how much your model leans on house style rather than on "
        "the description of the object. That comparison is worth a slide on "
        "its own in the November presentation.",
        "",
        "### Step 4 - Balance exposure during training",
        "",
        "- **Fine-tuned / classical classifier:** use class weights in the "
        "loss (`class_weight='balanced'`, or `1/sqrt(frequency)` which is less "
        "aggressive and usually behaves better across 700+ classes), or a "
        "class-balanced batch sampler. Either gives FLOORS and NAIL GUNS "
        "comparable influence per step without inventing data.",
        "- **Zero-shot extraction (GLiNER, the current plan):** there is no "
        "training loss to weight, so balancing applies to evaluation. Report "
        "per-class and macro scores, and when picking narratives to review or "
        "tune entity labels on, sample across product groups rather than from "
        "the head of the distribution, and only from the training fold.",
        "",
        "### Step 5 - Deduplicate across the fold boundary",
        "",
        "Drop exact narrative duplicates that would otherwise straddle train "
        "and validation. Small effect here ({} of rows) but free to "
        "do.".format(pct(df["Narrative_1"].duplicated(keep=False).mean())),
    ]
    return "\n".join(body)


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------

def build_report(df: pd.DataFrame, fmt: pd.DataFrame, path: str) -> str:
    pmap = product_labels(fmt)
    nar = df["Narrative_1"].fillna("")

    sections = [
        ("1. Dataset overview", section_overview(df, path)),
        ("2. Narrative length", section_length(nar)),
        ("3. Narrative structure and preprocessing", section_structure(nar)),
        ("4. Label structure: product codes", section_labels(df, pmap)),
        ("5. Is the object named in the narrative?", section_linkage(df, pmap)),
        ("6. The consumer-vs-workplace domain gap", section_domain_gap(df, pmap)),
        ("7. Class imbalance in the target", section_imbalance(df, pmap)),
        ("8. Hidden confounders", section_confounders(df, pmap)),
        ("9. Balancing exposure across contexts (balancing the label)",
         section_exposure(df, pmap)),
        ("10. Propose split and sampling recipe",
         section_recipe(df, pmap)),
    ]

    head = textwrap.dedent("""\
        # NEISS Exploratory Data Analysis


        **Modelling task:** `Narrative_1` (free-text injury description)
        -> `Product_1` (single NEISS product code).

        Generated by `src/neiss_eda.py`. Re-run with `python src/neiss_eda.py`
        to regenerate against a different year of NEISS data.

        Sections 1-6 describe the data. Sections 7-10 cover confounders and
        splitting.

        ## Contents

        """)
    head += "\n".join("- [{}](#{})".format(t, anchor(t)) for t, _ in sections)

    parts = [head]
    for title, content in sections:
        parts.append("\n---\n\n## {}\n\n{}".format(title, content))
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA, help="path to the NEISS .xlsx")
    ap.add_argument("--out", default=os.path.join(REPORT_DIR, "eda_report.md"),
                    help="where to write the markdown report")
    ap.add_argument("--splits", default=os.path.join(REPORT_DIR, "split_by_psu.csv"),
                    help="where to write the hospital-to-fold assignment")
    ap.add_argument("--no-cache", action="store_true", help="re-parse the .xlsx")
    args = ap.parse_args()

    df, fmt = load_data(args.data, use_cache=not args.no_cache)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.splits), exist_ok=True)

    # The split assignment is a deliverable in its own right: joining it onto
    # the data by PSU reproduces the recommended folds exactly.
    assign_psu_splits(df).to_csv(args.splits, index=False)

    report = build_report(df, fmt, args.data)
    with open(args.out, "w") as fh:
        fh.write(report)

    print(report)
    print("\n[report written to {}]".format(os.path.relpath(args.out, REPO)), file=sys.stderr)
    print("[split assignment written to {}]".format(
        os.path.relpath(args.splits, REPO)), file=sys.stderr)


if __name__ == "__main__":
    main()
