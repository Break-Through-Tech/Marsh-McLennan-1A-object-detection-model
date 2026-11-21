"""Task #3 reproducibility QA. Read-only checks on committed final-evaluation artifacts.

No model download, no pipeline rerun:  python -m pytest tests/test_task3_reproducibility.py
"""

import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TEST_SET = ROOT / "data/eval/test_set_neiss.csv"
SPLIT = ROOT / "reports/split_by_psu.csv"
METRICS = ROOT / "reports/final_eval_metrics.csv"
DETAILS = ROOT / "reports/final_eval_details.csv"

N_TEST = 3000
SYSTEMS = {"v1", "v2", "v3", "keyword", "majority"}
MAX_FILE_BYTES = 5 * 1024 * 1024
FORBIDDEN_PATH_PARTS = ("final_eval_cache", "data/processed/", "data/raw/", ".cache/", "huggingface")
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".bin", ".safetensors", ".onnx", ".ckpt", ".pkl", ".npy", ".npz")
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


@pytest.fixture(scope="module")
def test_set():
    return pd.read_csv(TEST_SET)


def test_test_set_has_3000_rows(test_set):
    assert len(test_set) == N_TEST


def test_case_numbers_unique(test_set):
    assert test_set["case_number"].is_unique


def test_narratives_unique(test_set):
    assert test_set["narrative"].is_unique


def test_all_test_psus_in_test_fold(test_set):
    folds = dict(zip(pd.read_csv(SPLIT)["PSU"].astype(int), pd.read_csv(SPLIT)["fold"]))
    psus = set(test_set["psu"].astype(int))
    missing = psus - folds.keys()
    assert not missing, f"PSUs absent from split file: {sorted(missing)}"
    assert {folds[p] for p in psus} == {"test"}


def test_metrics_contains_expected_systems():
    m = pd.read_csv(METRICS)
    assert set(m["system"]) == SYSTEMS
    assert m["system"].is_unique


def test_v3_metrics_match_committed_report():
    row = pd.read_csv(METRICS).set_index("system").loc["v3"]
    assert row["n"] == N_TEST
    assert row["micro_f1"] == pytest.approx(0.739, abs=0.001)
    assert row["macro_f1"] == pytest.approx(0.471, abs=0.001)
    assert row["exact_match_product_1"] == pytest.approx(0.712, abs=0.001)


def test_details_has_3000_rows_matching_test_set(test_set):
    d = pd.read_csv(DETAILS)
    assert len(d) == N_TEST
    assert set(d["case_number"]) == set(test_set["case_number"])


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git unavailable")
    return out.stdout


def test_no_caches_models_large_files_or_credentials_tracked():
    tracked = [p for p in _git("ls-files").splitlines() if p]
    assert tracked
    problems = []
    for rel in tracked:
        norm = rel.replace("\\", "/")
        path = ROOT / rel
        if not path.is_file():
            continue
        if any(part in norm for part in FORBIDDEN_PATH_PARTS) or norm.lower().endswith(FORBIDDEN_SUFFIXES):
            problems.append(f"forbidden path: {norm}")
        if norm == ".env" or norm.endswith("/.env"):
            problems.append(f"env file: {norm}")
        if path.stat().st_size > MAX_FILE_BYTES:
            problems.append(f"large file (>5MB): {norm}")
            continue
        if path.suffix.lower() in {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".cfg", ".ipynb"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            problems += [f"possible secret in {norm}" for pat in SECRET_PATTERNS if pat.search(text)]
    assert not problems, "\n".join(problems)
