"""Charts of the key findings for the final write-up (Milestone 2, Task #12).

Reads the Task #11 summary tables (and the Task #7 / #9 evaluation details)
from reports/, draws the charts into reports/figures/, and writes
reports/task12_visualization.md, which embeds them with short findings.
It needs no raw NEISS data, so anyone can rerun it from a fresh clone.

Run from the repo root:

    python -m src.visualize
    python -m src.visualize --reports-dir reports --top 25
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # write files only, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FALLBACK = "OTHER"
DEFAULT_REPORTS = Path("reports")
FIG_SUBDIR = "figures"
DPI = 200

# --- Object categories ---------------------------------------------------
# Hand-assigned for the 150 most frequent objects (about 65% of extracted
# cases). Anything not listed is counted as "Long tail (uncategorized)".
# "Not an object" collects spans the model returned that are body actions,
# symptoms or rooms rather than products; they are extraction errors.
CATEGORIES: Dict[str, List[str]] = {
    "Floors and ground": [
        "FLOOR", "GROUND LEVEL", "RUG", "CARPET", "CONCRETE", "TILE FLOOR", "CONCRETE FLOOR",
        "MAT", "CURB", "ROCK", "TREE",
    ],
    "Stairs and steps": [
        "STAIRS", "STEPS", "FLIGHT OF STAIRS", "STEP", "PORCH", "DECK", "RAMP", "RAILING",
        "ESCALATOR", "BLEACHERS",
    ],
    "Walls, doors and fixtures": [
        "WALL", "DOOR", "DOOR FRAME", "WINDOW", "CEILING", "GATE", "GARAGE DOOR", "FENCE",
        "POLE", "POST", "METAL POLE", "FIREPLACE", "COUNTER",
    ],
    "Furniture and beds": [
        "BED", "CHAIR", "COUCH", "TABLE", "COFFEE TABLE", "CABINET", "DRESSER", "FURNITURE",
        "BENCH", "STOOL", "NIGHTSTAND", "DESK", "RECLINER", "SOFA", "SHELF", "BUNK BED",
        "BED FRAME", "MATTRESS", "HIGH CHAIR", "CRIB", "BEDFRAME", "NIGHT STAND",
        "PIECE OF FURNITURE", "RECLINER CHAIR", "TV STAND", "MIRROR", "BLANKET",
    ],
    "Bathroom and kitchen": [
        "TOILET", "BATHTUB", "TUB", "SHOWER", "SINK", "REFRIGERATOR", "STOVE",
    ],
    "Sports and recreation": [
        "BASKETBALL", "BALL", "TRAMPOLINE", "FOOTBALL", "HELMET", "BASEBALL", "MONKEY BARS",
        "SLIDE", "WEIGHTS", "VOLLEYBALL", "SOCCER BALL", "SOCCER", "SWING",
        "PLAYGROUND EQUIPMENT", "SOFTBALL", "POOL", "TREADMILL", "SWIMMING POOL",
        "FLAG FOOTBALL", "PICKLEBALL", "BOUNCE HOUSE", "ROLLER SKATING", "GOLF", "KICKBALL",
        "PLAYGROUND",
    ],
    "Bikes, scooters and vehicles": [
        "BICYCLE", "SCOOTER", "BIKE", "ELECTRIC SCOOTER", "DIRT BIKE", "SKATEBOARD", "ATV",
        "E-BIKE", "GOLF CART", "HANDLEBARS", "MOTORIZED SCOOTER", "ELECTRIC BICYCLE",
        "ELECTRIC BIKE", "MOPED",
    ],
    "Tools and sharp objects": [
        "KNIFE", "LADDER", "KITCHEN KNIFE", "GLASS", "NAIL", "TABLE SAW", "BOX CUTTER",
        "HAMMER", "SAW", "FISH HOOK", "CHAINSAW", "SCISSORS", "RAZOR BLADE", "PENCIL",
    ],
    "Small items, toys and personal": [
        "TOY", "BOX", "BEAD", "LEASH", "SHOE", "SHOPPING CART", "SHOES", "COIN", "BOTTLE",
        "RING", "EARRING", "TRASH CAN", "BOXES", "STROLLER", "SOCKS", "PENNY", "PHONE",
        "WHEELCHAIR", "WALKER",
    ],
    "Not an object (extraction error)": [
        "STRIKING HEAD", "SYNCOPE", "GARAGE", "GYM", "RASH", "LIVING ROOM", "TWISTED ANKLE",
        "HITTING HEAD", "BASEMENT", "CHURCH",
    ],
}
LONG_TAIL = "Long tail (uncategorized)"
OBJECT_TO_CATEGORY = {obj: cat for cat, objs in CATEGORIES.items() for obj in objs}

# One fixed color per category so every chart matches.
PALETTE = {
    "Floors and ground": "#4C72B0",
    "Stairs and steps": "#55A868",
    "Walls, doors and fixtures": "#8172B2",
    "Furniture and beds": "#DD8452",
    "Bathroom and kitchen": "#64B5CD",
    "Sports and recreation": "#C44E52",
    "Bikes, scooters and vehicles": "#937860",
    "Tools and sharp objects": "#DA8BC3",
    "Small items, toys and personal": "#CCB974",
    "Not an object (extraction error)": "#8C8C8C",
    LONG_TAIL: "#D0D0D0",
}

AGE_ORDER = ["0-4", "5-14", "15-24", "25-44", "45-64", "65+"]


def categorize(obj: str) -> str:
    return OBJECT_TO_CATEGORY.get(str(obj).upper(), LONG_TAIL)


# --- Data prep -----------------------------------------------------------

def load_ranking(path: Path) -> pd.DataFrame:
    """Task #11 ranking with a category column; the fallback row is kept."""
    r = pd.read_csv(path, keep_default_na=False, na_values=[""])
    r["object"] = r["object"].astype(str)
    r["category"] = r["object"].map(categorize)
    r.loc[r["object"] == FALLBACK, "category"] = FALLBACK
    return r


def real_objects(ranking: pd.DataFrame) -> pd.DataFrame:
    """Ranking without the fallback, with share of extracted (non-OTHER) cases."""
    real = ranking[ranking["object"] != FALLBACK].copy().reset_index(drop=True)
    total = real["count"].sum()
    real["pct_of_extracted"] = 100 * real["count"] / total
    real["cum_pct_of_extracted"] = real["pct_of_extracted"].cumsum()
    return real


def category_shares(ranking: pd.DataFrame) -> pd.DataFrame:
    """Cases and share of extracted cases per category, largest first (long tail last)."""
    real = real_objects(ranking)
    g = real.groupby("category").agg(count=("count", "sum"), objects=("object", "size")).reset_index()
    g["pct_of_extracted"] = 100 * g["count"] / g["count"].sum()
    g["order"] = (g["category"] == LONG_TAIL).astype(int)
    return g.sort_values(["order", "count"], ascending=[True, False]).drop(columns="order").reset_index(drop=True)


def field_matrix(by_field: pd.DataFrame, field: str, objects: List[str], values: List[str] | None = None,
                 metric: str = "pct_of_value") -> pd.DataFrame:
    """objects x values matrix of `metric` for one NEISS field (0 where the pair never occurs)."""
    sub = by_field[by_field["field"] == field]
    m = sub.pivot_table(index="object", columns="value", values=metric, aggfunc="sum").reindex(objects).fillna(0)
    if values is not None:
        m = m.reindex(columns=[v for v in values if v in m.columns]).fillna(0)
    return m


def value_totals(by_field: pd.DataFrame, field: str) -> pd.Series:
    """Cases per field value (non-OTHER), recovered from count / pct_of_value."""
    sub = by_field[(by_field["field"] == field) & (by_field["pct_of_value"] > 0)]
    return (100 * sub["count"] / sub["pct_of_value"]).groupby(sub["value"]).median().round().sort_values(ascending=False)


def object_totals(by_field: pd.DataFrame) -> pd.Series:
    """Cases per object, recovered from count / pct_of_object."""
    sub = by_field[by_field["pct_of_object"] > 0]
    return (100 * sub["count"] / sub["pct_of_object"]).groupby(sub["object"]).median().round()


def admission_rates(by_field: pd.DataFrame) -> pd.DataFrame:
    """Share of each top object's cases that were admitted, held or transferred.

    Spans in the "Not an object" category (e.g. STRIKING HEAD) are left out.
    """
    sub = by_field[by_field["field"] == "disposition"].copy()
    code = sub["value"].str.extract(r"^(\d+)")[0]
    sub["serious"] = code.isin(["2", "4", "5"])
    sub = sub[sub["object"].map(categorize) != "Not an object (extraction error)"]
    out = sub.groupby("object").apply(
        lambda g: pd.Series({
            "cases": g["count"].sum(),
            "serious_pct": 100 * g.loc[g["serious"], "count"].sum() / g["count"].sum(),
        }),
        include_groups=False,
    )
    return out.sort_values("serious_pct", ascending=False)


def eval_metrics(details: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Precision / recall / F1 from a Task #7 or #9 details CSV (same rules as src/evaluate.py)."""
    answered = details["extracted_object"].fillna(FALLBACK).astype(str).str.upper() != FALLBACK
    out = {}
    for rule in ("strict", "lenient"):
        tp = int(((details[f"{rule}_correct"].astype(str) == "True") & answered).sum())
        p = tp / answered.sum() if answered.sum() else 0.0
        r = tp / len(details) if len(details) else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out[rule] = {"precision": p, "recall": r, "f1": f1}
    return out


# --- Charts --------------------------------------------------------------

def _style(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#555555", va="bottom")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_top_objects(ranking: pd.DataFrame, path: Path, top: int = 25) -> Path:
    real = real_objects(ranking).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.32 * top + 1.6))
    colors = [PALETTE[c] for c in real["category"]]
    ax.barh(real["object"], real["count"], color=colors)
    xmax = real["count"].max()
    for y, (c, p) in enumerate(zip(real["count"], real["pct_of_extracted"])):
        ax.text(c + xmax * 0.01, y, f"{c:,}  ({p:.1f}%)", va="center", fontsize=8)
    ax.set_xlim(0, xmax * 1.22)
    ax.set_xlabel("Cases")
    ax.tick_params(axis="y", labelsize=9)
    _style(ax, f"Top {top} extracted objects",
           "410,201 NEISS 2025 cases; % of cases with an extracted object (OTHER fallback excluded)")
    used = list(dict.fromkeys(real["category"].iloc[::-1]))
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[c]) for c in used]
    ax.legend(handles, used, loc="lower right", fontsize=8, frameon=False, title="Category", title_fontsize=8)
    return _save(fig, path)


def plot_case_composition(ranking: pd.DataFrame, path: Path) -> Path:
    """Single stacked bar: where every case ends up."""
    real = real_objects(ranking)
    n = ranking["count"].sum()
    fallback = int(ranking.loc[ranking["object"] == FALLBACK, "count"].sum())
    c = real["count"].to_numpy()
    singles = int(real.loc[real["count"] == 1, "count"].sum())
    parts = [
        ("FLOOR (rank 1)", c[:1].sum(), "#4C72B0"),
        ("Ranks 2-10", c[1:10].sum(), "#6F93C9"),
        ("Ranks 11-50", c[10:50].sum(), "#9DB6DD"),
        ("Ranks 51+ (seen 2+ times)", c[50:].sum() - singles, "#C9D6EC"),
        ("Seen once", singles, "#E6ECF6"),
        ("OTHER\n(no object found)", fallback, "#8C8C8C"),
    ]
    fig, ax = plt.subplots(figsize=(10, 2.6))
    left = 0
    for label, val, col in parts:
        pct = 100 * val / n
        ax.barh([0], [pct], left=left, color=col, edgecolor="white")
        text = f"{label}\n{pct:.1f}%" if pct >= 10 else f"{label.split(' (')[0]}\n{pct:.1f}%"
        if pct < 7:  # narrow segment: label above the bar instead of inside it
            ax.text(left + pct / 2, 0.48, text, ha="center", va="bottom", fontsize=7)
        else:
            ax.text(left + pct / 2, 0, text, ha="center", va="center", fontsize=8,
                    color="white" if col in ("#4C72B0", "#6F93C9", "#8C8C8C") else "black")
        left += pct
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("% of all 410,201 cases")
    for side in ("left",):
        ax.spines[side].set_visible(False)
    _style(ax, "Where every case lands",
           f"{len(real):,} distinct objects; {int((real['count'] == 1).sum()):,} appear only once")
    return _save(fig, path)


def plot_long_tail(ranking: pd.DataFrame, path: Path) -> Path:
    real = real_objects(ranking)
    ranks = np.arange(1, len(real) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ranks, real["cum_pct_of_extracted"], color="#4C72B0", lw=2)
    ax.set_xscale("log")
    for k in (1, 10, 50, 150, 1000):
        if k <= len(real):
            v = real["cum_pct_of_extracted"].iloc[k - 1]
            ax.scatter([k], [v], color="#C44E52", zorder=3, s=20)
            ax.annotate(f"top {k:,}: {v:.0f}%", (k, v), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Object rank (log scale)")
    ax.set_ylabel("Cumulative % of extracted cases")
    ax.grid(alpha=0.3, which="both")
    _style(ax, "A few objects dominate, then a very long tail",
           "Cumulative share of cases with an extracted object, by object rank")
    return _save(fig, path)


def plot_categories(ranking: pd.DataFrame, path: Path) -> Path:
    cats = category_shares(ranking).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(cats["category"], cats["pct_of_extracted"], color=[PALETTE[c] for c in cats["category"]])
    for y, (p, c, k) in enumerate(zip(cats["pct_of_extracted"], cats["count"], cats["objects"])):
        ax.text(p + 0.4, y, f"{p:.1f}%  ({c:,} cases, {k:,} objects)", va="center", fontsize=8)
    ax.set_xlim(0, cats["pct_of_extracted"].max() * 1.45)
    ax.set_xlabel("% of cases with an extracted object")
    _style(ax, "Extracted objects by category",
           "Top 150 objects hand-grouped (src/visualize.py); everything else is the long tail")
    return _save(fig, path)


def plot_heatmap(matrix: pd.DataFrame, path: Path, title: str, subtitle: str, xlabel: str,
                 col_labels: List[str] | None = None) -> Path:
    data = matrix.to_numpy()
    fig, ax = plt.subplots(figsize=(max(6, 0.75 * data.shape[1] + 3), 0.38 * data.shape[0] + 2))
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(col_labels or list(matrix.columns), rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8)
    vmax = data.max() if data.size else 1
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if v > vmax * 0.55 else "black")
    ax.set_xlabel(xlabel)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("% of column's cases", fontsize=8)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#555555", va="bottom")
    return _save(fig, path)


def plot_admission(rates: pd.DataFrame, ranking: pd.DataFrame, path: Path) -> Path:
    r = rates.iloc[::-1]
    overall = 100 * rates["cases"].mul(rates["serious_pct"]).sum() / rates["cases"].sum() / 100
    cats = [categorize(o) for o in r.index]
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(r) + 1.8))
    ax.barh(r.index, r["serious_pct"], color=[PALETTE[c] for c in cats])
    ax.axvline(overall, color="#333333", ls="--", lw=1)
    ax.text(overall + 0.3, -0.9, f"average across these objects: {overall:.1f}%", fontsize=8)
    for y, v in enumerate(r["serious_pct"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, r["serious_pct"].max() * 1.2)
    ax.set_ylim(-1.3, len(r) - 0.4)
    ax.set_xlabel("% of the object's cases admitted, transferred or held for observation")
    ax.tick_params(axis="y", labelsize=8)
    _style(ax, "Which common objects lead to more serious outcomes",
           "Top 25 objects (non-object spans removed); NEISS disposition codes 2, 4 and 5")
    return _save(fig, path)


def plot_eval(before: Dict, after: Dict, path: Path) -> Path:
    metrics = ["precision", "recall", "f1"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, rule in zip(axes, ("strict", "lenient")):
        x = np.arange(len(metrics))
        b = [before[rule][m] for m in metrics]
        a = [after[rule][m] for m in metrics]
        ax.bar(x - 0.18, b, 0.36, label="Task #7 baseline (v1)", color="#BBBBBB")
        ax.bar(x + 0.18, a, 0.36, label="Task #9 refined (v2)", color="#4C72B0")
        for xi, (bv, av) in enumerate(zip(b, a)):
            ax.text(xi - 0.18, bv + 0.02, f"{bv:.2f}", ha="center", fontsize=8)
            ax.text(xi + 0.18, av + 0.02, f"{av:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Precision", "Recall", "F1"])
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{rule.capitalize()} match", fontsize=11)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Extraction accuracy on the 25-case labeled sample", x=0.01, ha="left", fontweight="bold")
    return _save(fig, path)


# --- Report --------------------------------------------------------------

def build_report(ranking: pd.DataFrame, by_field: pd.DataFrame, figs: Dict[str, Path], reports_dir: Path,
                 before: Dict | None, after: Dict | None) -> str:
    real = real_objects(ranking)
    n = int(ranking["count"].sum())
    fallback = int(ranking.loc[ranking["object"] == FALLBACK, "count"].sum())
    cats = category_shares(ranking)
    named = cats[~cats["category"].isin([LONG_TAIL])]
    noise = cats.loc[cats["category"] == "Not an object (extraction error)", "pct_of_extracted"].sum()
    rates = admission_rates(by_field)
    ages = field_matrix(by_field, "age_group", ["FLOOR"], AGE_ORDER).loc["FLOOR"]
    loc_tot = value_totals(by_field, "location")

    def cum_at(k: int) -> float:
        return float(real["cum_pct_of_extracted"].iloc[min(k, len(real)) - 1])

    def rel(p: Path) -> str:
        return p.relative_to(reports_dir).as_posix()

    top_cat = named.iloc[0]
    lines = [
        "# Key findings: charts",
        "",
        "Marsh McLennan 1A, Break Through Tech AI Studio, Fall 2026",
        "Milestone 2, Task #12",
        "",
        "Charts of the objects extracted from all 410,201 NEISS 2025 narratives in the v4 full run, built from the "
        "Task #11 tables (`task11_object_ranking.csv`, `task11_object_by_field.csv`). Generated by "
        "`python -m src.visualize`; PNGs are in `reports/figures/` and can be dropped into the final slides or write-up.",
        "",
        "## 1. Most frequently extracted objects",
        "",
        f"![Top extracted objects]({rel(figs['top_objects'])})",
        "",
        f"- **FLOOR** is by far the most common object: {real['count'].iloc[0]:,} cases, "
        f"{real['pct_of_extracted'].iloc[0]:.1f}% of cases with an extracted object. "
        f"The next three, {', '.join(real['object'].iloc[1:4])}, add another "
        f"{real['pct_of_extracted'].iloc[1:4].sum():.1f}%.",
        "- The top 25 mix home furnishings (bed, chair, couch, table), building structure (stairs, wall, door) "
        "and sports equipment (basketball, football, trampoline). Many of them (floor, bed, stairs, chair) are typical fall surfaces.",
        "- Several near-duplicates appear separately (STAIRS / STEPS / FLIGHT OF STAIRS, BICYCLE / BIKE), so normalizing "
        "synonyms would push stairs past BED into second place.",
        "",
        "## 2. How cases are spread across objects",
        "",
        f"![Case composition]({rel(figs['composition'])})",
        "",
        f"![Long tail]({rel(figs['long_tail'])})",
        "",
        f"- {fallback:,} cases ({100 * fallback / n:.1f}%) got no object (OTHER).",
        f"- The top 10 objects cover {cum_at(10):.0f}% of extracted cases and the top 50 cover "
        f"{cum_at(50):.0f}%, but there are {len(real):,} distinct objects and "
        f"{int((real['count'] == 1).sum()):,} of them appear only once.",
        "- The long tail includes spelling variants, modifiers (\"TILE FLOOR\", \"KITCHEN KNIFE\") and rare products. "
        "A normalization step that maps these to canonical names is the clearest next improvement.",
        "",
        "## 3. Distribution across object categories",
        "",
        f"![Categories]({rel(figs['categories'])})",
        "",
        f"- The 150 most frequent objects were hand-grouped into categories (mapping in `src/visualize.py`). "
        f"**{top_cat['category']}** is the largest group at {top_cat['pct_of_extracted']:.1f}% of extracted cases, "
        "driven almost entirely by FLOOR.",
        "- " + "; ".join(f"{r.category}: {r.pct_of_extracted:.1f}%" for r in named.iloc[1:6].itertuples()) + ".",
        f"- About {noise:.1f}% of extracted cases are spans that are not products at all (STRIKING HEAD, SYNCOPE, "
        "TWISTED ANKLE, rooms like GARAGE). These are extraction errors worth adding to the stoplist.",
        "",
        "## 4. Objects by patient and incident fields",
        "",
        "Each heatmap cell is the share of that column's cases (excluding OTHER) that involve the object, for the top 15 objects.",
        "",
        "### Age group",
        "",
        f"![Objects by age]({rel(figs['age'])})",
        "",
        f"- FLOOR rises steeply with age: {ages.get('0-4', 0):.0f}% of cases for ages 0-4 versus "
        f"{ages.get('65+', 0):.0f}% for 65+, consistent with falls in older adults.",
        "- Young children (0-4) are most often hurt by BED and COUCH; ages 5-24 by BASKETBALL and other sports equipment.",
        "",
        "### Location",
        "",
        f"![Objects by location]({rel(figs['location'])})",
        "",
        f"- HOME is the largest known location ({int(loc_tot.get('HOME', 0)):,} extracted cases); beds, couches and floors lead there.",
        "- SPORTS locations are dominated by balls (BASKETBALL, BALL, FOOTBALL); STREET by BICYCLE, HELMET and SCOOTER.",
        "",
        "### Body part",
        "",
        f"![Objects by body part]({rel(figs['body_part'])})",
        "",
        "- KNIFE and DOOR stand out for FINGER injuries, and WALL / KNIFE for HAND; nearly every other body part is led by FLOOR.",
        "",
        "### Diagnosis",
        "",
        f"![Objects by diagnosis]({rel(figs['diagnosis'])})",
        "",
        "- FLOOR accounts for nearly a quarter of fractures; KNIFE narrowly leads lacerations; BASKETBALL ties STAIRS as the top non-floor object for strains and sprains.",
        "",
        "## 5. Severity: admissions by object",
        "",
        f"![Admission rates]({rel(figs['admission'])})",
        "",
        f"- Fall-related objects have the highest share of serious outcomes: "
        + ", ".join(f"{o} {v:.0f}%" for o, v in rates["serious_pct"].head(4).items())
        + " of cases were admitted, transferred or held.",
        "- Sports objects and KNIFE are mostly treated and released.",
        "",
    ]
    if before and after:
        lines += [
            "## 6. Extraction accuracy (context for the numbers above)",
            "",
            f"![Evaluation]({rel(figs['eval'])})",
            "",
            f"- The v2 rules (stoplist and floor default) raised strict F1 from {before['strict']['f1']:.2f} to "
            f"{after['strict']['f1']:.2f} and lenient F1 from {before['lenient']['f1']:.2f} to {after['lenient']['f1']:.2f}.",
            "- The labeled sample is only 25 cases, so treat these scores as rough.",
            "",
        ]
    lines += [
        "## Limitations",
        "",
        "- Objects are model predictions, not ground truth, so every count inherits the extraction error above.",
        "- FLOOR is inflated by the Task #9 default rule (a fall with no named object becomes FLOOR).",
        "- Category groupings are hand-assigned for the top 150 objects only and are a judgment call.",
        "- Field breakdowns only cover the top 25 objects (the Task #11 table) and exclude OTHER.",
        "- NEISS case weights are not applied: counts are sampled ED visits, not national estimates.",
        "",
    ]
    return "\n".join(lines)


# --- Main ----------------------------------------------------------------

def make_all(reports_dir: Path, top: int = 25) -> Dict[str, Path]:
    ranking = load_ranking(reports_dir / "task11_object_ranking.csv")
    by_field = pd.read_csv(reports_dir / "task11_object_by_field.csv", keep_default_na=False, na_values=[""])
    fig_dir = reports_dir / FIG_SUBDIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    heat_objs = real_objects(ranking)["object"].head(15).tolist()
    heat_objs = [o for o in heat_objs if o in set(by_field["object"])]

    figs: Dict[str, Path] = {}
    figs["top_objects"] = plot_top_objects(ranking, fig_dir / "task12_top_objects.png", top)
    figs["composition"] = plot_case_composition(ranking, fig_dir / "task12_case_composition.png")
    figs["long_tail"] = plot_long_tail(ranking, fig_dir / "task12_long_tail.png")
    figs["categories"] = plot_categories(ranking, fig_dir / "task12_categories.png")

    figs["age"] = plot_heatmap(
        field_matrix(by_field, "age_group", heat_objs, AGE_ORDER), fig_dir / "task12_objects_by_age.png",
        "Top objects by age group", "% of each age group's cases (OTHER excluded)", "Age group (years)")

    loc_tot = value_totals(by_field, "location")
    locs = [v for v, t in loc_tot.items() if v != "Unknown" and t >= 500] + ["Unknown"]  # drop tiny groups
    figs["location"] = plot_heatmap(
        field_matrix(by_field, "location", heat_objs, locs), fig_dir / "task12_objects_by_location.png",
        "Top objects by location", "Locations with 500+ cases; % of each location's cases (OTHER excluded)", "Location")

    parts = value_totals(by_field, "body_part").drop("Unknown", errors="ignore").head(12).index.tolist()
    figs["body_part"] = plot_heatmap(
        field_matrix(by_field, "body_part", heat_objs, parts), fig_dir / "task12_objects_by_body_part.png",
        "Top objects by body part", "12 most common body parts; % of each body part's cases (OTHER excluded)",
        "Body part", [p.split(" - ", 1)[-1].title() for p in parts])

    diags = value_totals(by_field, "diagnosis").drop("71 - OTHER", errors="ignore").head(10).index.tolist()
    figs["diagnosis"] = plot_heatmap(
        field_matrix(by_field, "diagnosis", heat_objs, diags), fig_dir / "task12_objects_by_diagnosis.png",
        "Top objects by diagnosis", "10 most common diagnoses (excluding 'Other'); % of each diagnosis's cases",
        "Diagnosis", [d.split(" - ", 1)[-1].title() for d in diags])

    figs["admission"] = plot_admission(admission_rates(by_field), ranking, fig_dir / "task12_admission_by_object.png")

    before = after = None
    p7, p9 = reports_dir / "task7_eval_details.csv", reports_dir / "task9_eval_details.csv"
    if p7.exists() and p9.exists():
        before, after = eval_metrics(pd.read_csv(p7)), eval_metrics(pd.read_csv(p9))
        figs["eval"] = plot_eval(before, after, fig_dir / "task12_eval_v1_vs_v2.png")

    report = build_report(ranking, by_field, figs, reports_dir, before, after)
    figs["report"] = reports_dir / "task12_visualization.md"
    figs["report"].write_text(report, encoding="utf-8")
    return figs


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Charts of the key extraction findings")
    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--top", type=int, default=25, help="objects shown in the top-objects chart")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    for name in ("task11_object_ranking.csv", "task11_object_by_field.csv"):
        if not (args.reports_dir / name).exists():
            raise SystemExit(f"{args.reports_dir / name} not found. Run `python -m src.summarize` first.")
    out = make_all(args.reports_dir, args.top)
    print(f"Wrote {len(out) - 1} charts to {args.reports_dir / FIG_SUBDIR} and {out['report']}")


if __name__ == "__main__":
    main()
