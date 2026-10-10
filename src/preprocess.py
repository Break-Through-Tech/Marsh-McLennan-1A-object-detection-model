"""Narrative preprocessing for the Marsh object-extraction project (Task #6).

One shared copy of the cleaning rules, used by:
  - notebooks/milestone1_eda.ipynb  (builds the cleaned dataset)
  - src/neiss_eda.py                (reports what the rules change)
  - the Milestone 2 model code      (must clean Marsh's narratives the same way)

The raw text is never modified; each step returns a new Series.
"""

import pandas as pd

# (shown as, regex, expansion). Only abbreviations whose meaning was confirmed
# from NEISS narratives. Order matters: W/O must be expanded before W/.
#
# Deliberately NOT expanded:
#   YOM/YOF/MOM/MOF  structured age/sex codes; MOM can also mean "mother"
#   DX               used as a marker to split off the diagnosis (see below)
#   R / L            single letters are too ambiguous to replace safely
#   INJ, LOC         can mean injury/injection and location/loss of consciousness
ABBREVIATIONS = [
    ("C/O",    r"\bC/O\b",       "COMPLAINTS OF"),
    ("S/P",    r"\bS/P\b",       "STATUS POST"),
    ("W/O",    r"\bW/O\b",       "WITHOUT"),
    ("W/",     r"\bW/(?!O\b)",   "WITH "),
    ("PT",     r"\bPT\b",        "PATIENT"),
    ("BIBEMS", r"\bBIBEMS\b",    "BROUGHT IN BY EMS"),
    ("FB",     r"\bFB\b",        "FOREIGN BODY"),
    ("FX",     r"\bFX\b",        "FRACTURE"),
    ("LAC",    r"\bLAC\b",       "LACERATION"),
    ("CHI",    r"\bCHI\b",       "CLOSED HEAD INJURY"),
]

# Words, keeping hyphenated and apostrophe forms together (SIDE-BY-SIDE, PT'S).
TOKEN_PATTERN = r"\b\w+(?:[-']\w+)*\b"

# NEISS replaces identifying words with asterisks. Sometimes the redacted word
# is the object itself ("2 YOM * IN NOSE"), which no model can recover.
REDACTION_PATTERN = r"\*"

# Everything after the first DX marker describes the injury, not the object.
DX_SPLIT_PATTERN = r"^(?P<event>.*?)\s*\bDX\b\s*:?\s*(?P<diagnosis>.*)$"


def clean_whitespace(narratives: pd.Series) -> pd.Series:
    """Collapse runs of spaces, tabs and line breaks to one space; trim the ends."""
    return narratives.str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_abbreviations(narratives: pd.Series) -> pd.Series:
    """Expand the verified clinical abbreviations in ABBREVIATIONS."""
    for _, pattern, expansion in ABBREVIATIONS:
        narratives = narratives.str.replace(pattern, expansion, regex=True)
    # "W/KNIFE" -> "WITH KNIFE" can leave a double space; tidy it.
    return clean_whitespace(narratives)


def preprocess(narratives: pd.Series, lowercase: bool = False) -> pd.Series:
    """Full Task #6 text pipeline: whitespace cleanup, then abbreviation expansion.

    lowercase=True is for testing whether the model does better on mixed-case
    text (all NEISS narratives are uppercase). Off by default.
    """
    out = normalize_abbreviations(clean_whitespace(narratives))
    return out.str.lower() if lowercase else out


def split_diagnosis(narratives: pd.Series) -> pd.DataFrame:
    """Split each narrative into the event description and the diagnosis.

    Returns columns `event` (where the object is mentioned) and `diagnosis`
    (empty when the narrative has no DX marker).
    """
    parts = narratives.str.extract(DX_SPLIT_PATTERN)
    no_dx = parts["event"].isna()
    parts.loc[no_dx, "event"] = narratives[no_dx]
    parts["diagnosis"] = parts["diagnosis"].fillna("")
    return parts


def has_redaction(narratives: pd.Series) -> pd.Series:
    """True where the narrative contains a redacted (asterisked) word."""
    return narratives.str.contains(REDACTION_PATTERN, regex=True)


def tokenize(narratives: pd.Series) -> pd.Series:
    """Split each narrative into a list of word tokens."""
    return narratives.str.findall(TOKEN_PATTERN)
