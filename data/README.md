# Data

The NEISS data files are not stored in this repo (they are large). Download and convert them as described below. `data/raw/` and `data/processed/` are gitignored.

## 1. Download

Get the NEISS 2025 workbook (the **Excel** version, not tab-delimited text) from the CPSC: https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data

Save it as `data/raw/neiss2025.xlsx`. It has two sheets:

| Sheet | Contents |
|---|---|
| `NEISS_2025` | The data: 410,201 rows x 25 columns, one row per ED visit |
| `NEISS_FMT` | The codebook: decodes the numeric columns (e.g. `Product_1`, `Body_Part`) |

The tab-delimited download has only the data, not the codebook, so use the Excel file.

## 2. Convert to CSV

The notebook reads CSVs, which load much faster than the xlsx. Run this once from the repo root (needs pandas and openpyxl, both in `requirements.txt`):

```python
import pandas as pd

for sheet in ["NEISS_2025", "NEISS_FMT"]:
    pd.read_excel("data/raw/neiss2025.xlsx", sheet_name=sheet) \
      .to_csv(f"data/raw/neiss2025 1({sheet}).csv", index=False)
```

This takes a few minutes.

## 3. Build the cleaned dataset

Run `notebooks/milestone1_eda.ipynb` (Restart, then Run All). Its last step writes the cleaned dataset that Milestone 2 uses.

## 4. Check

```
data/
├── README.md
├── raw/
│   ├── neiss2025.xlsx                  downloaded
│   ├── neiss2025 1(NEISS_2025).csv     step 2
│   └── neiss2025 1(NEISS_FMT).csv      step 2
└── processed/
    ├── neiss2025_clean.csv             step 3 (notebook)
    └── .cache/                         created by src/neiss_eda.py
```

- `notebooks/milestone1_eda.ipynb` reads the two CSVs and writes `neiss2025_clean.csv`.
- `src/neiss_eda.py` reads `neiss2025.xlsx` directly and caches it in `.cache/`.
