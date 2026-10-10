# Data

The NEISS data files are **not stored in this repo** (they are large). Download them and
convert them to CSV as described below. Everything in `data/` except this file is gitignored.

## 1. Download

Get the NEISS 2025 workbook from the CPSC:
https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data

Save it as `data/neiss2025.xlsx`. It has two sheets:

| Sheet | Contents |
|---|---|
| `NEISS_2025` | The data: 410,201 rows x 25 columns, one row per ED visit |
| `NEISS_FMT` | The codebook: decodes the numeric columns (e.g. `Product_1`, `Body_Part`) |

## 2. Convert to CSV

The notebook reads CSVs, which load much faster than the xlsx. Run this once from the repo root
(needs `pandas` and `openpyxl`, both in `requirements.txt`):

```python
import pandas as pd

for sheet in ["NEISS_2025", "NEISS_FMT"]:
    pd.read_excel("data/neiss2025.xlsx", sheet_name=sheet) \
      .to_csv(f"data/neiss2025 1({sheet}).csv", index=False)
```

This takes a few minutes. Alternatively, open the xlsx in Excel, go to each sheet, and use
**File > Save As > CSV** — name the files exactly as below.

## 3. Check

`data/` should now contain:

```
neiss2025.xlsx
neiss2025 1(NEISS_2025).csv
neiss2025 1(NEISS_FMT).csv
```

`notebooks/milestone1_eda.ipynb` loads the two CSVs by these exact names.
