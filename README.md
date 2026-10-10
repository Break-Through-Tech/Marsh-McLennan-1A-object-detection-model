# Marsh McLennan: Object Extraction from Injury Narratives

Break Through Tech AI Studio, Fall 2026 · Host company: Marsh McLennan

---

### 👥 **Team Members**

| Name           | GitHub Handle                                         |
|----------------|-------------------------------------------------------|
| Daisy Fregoso  | [@dazyfrgso](https://github.com/dazyfrgso)            |
| Gloria Yip     | [@gyip40](https://github.com/gyip40)                  |
| Gigi Rurua     | [@GigiRurua](https://github.com/GigiRurua)            |
| Christine Ng   | [@ChristineNg2027](https://github.com/ChristineNg2027) |
| Bhoomika Gupta | [@BhoomikaGuptaa](https://github.com/BhoomikaGuptaa)  |
| Jacky Jiang    | [@Zhejiej](https://github.com/Zhejiej)                |

---

## 🎯 **Project Highlights**

- Explored all 410,201 records of the 2025 NEISS dataset to decide which fields can be used to extract the object involved in an injury.
- Chose `Narrative_1` (free-text injury description) as the only feature and `Product_1` (the product the NEISS coder recorded) as the label, because Marsh's test set is raw workers' compensation text with no NEISS codes.
- Found that the product name appears in the narrative text for 66% of records (72% once vague labels are excluded), which shows how much of the label can be recovered from the text alone.
- Planned a local extraction pipeline (GLiNER with spaCy cleanup) that needs no API costs.

---

## 👩🏽‍💻 **Setup and Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/Break-Through-Tech/Marsh-McLennan-1A-object-detection-model.git
   cd Marsh-McLennan-1A-object-detection-model
   ```
2. **Create an environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
   GLiNER installs PyTorch, so the install is large.
3. **Get the data.** The data files are not stored in the repo. Follow [data/README.md](data/README.md) to download the NEISS 2025 workbook and convert it to CSV.
4. **Run the notebook.** Open `notebooks/milestone1_eda.ipynb` in Jupyter and run all cells.

---

## 🏗️ **Project Overview**

This project is part of the Break Through Tech AI Studio. Marsh McLennan is a global leader in insurance brokerage and risk management. Our goal is to extract the object involved in an injury (a ladder, a knife, a floor) from a short free-text description.

We train and test on hospital emergency department injury descriptions from the U.S. Consumer Product Safety Commission's NEISS database. Marsh will evaluate the final model on anonymized workers' compensation accident descriptions and compare it with their existing object detection model.

Better object extraction helps Marsh find which objects and products drive workplace injuries, which supports injury prevention, lower injury-related costs, and better loss prevention.

| Month | Milestone |
|---|---|
| September | Data exploration and modeling plan |
| October | Model v1 and initial validation |
| November | Model refinement and final evaluation |

---

## 📊 **Data Exploration**

**Dataset:** CPSC National Electronic Injury Surveillance System (NEISS), 2025. 410,201 emergency department visits and 25 columns. Source: https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data

**Analysis:** [notebooks/milestone1_eda.ipynb](notebooks/milestone1_eda.ipynb)

| Finding | Detail |
|---|---|
| Narrative length | Median 114 characters. 168 narratives are cut off at the 400-character limit and 3,016 are under 40 characters. |
| Clinical shorthand | Narratives use abbreviations such as `YOM`, `DX:`, `C/O`, `S/P`, `PT` and `W/`, and some words are redacted with `*`. These need normalizing. |
| Missing data | There are no true nulls. Blanks are conditional fields, and the real missingness is coded (for example `Location = 0`, `Diagnosis = 71`). |
| Label concentration | `Product_1` uses 767 classes. The top 157 cover 90% of records, and 461 classes have fewer than 100 examples. |
| Vague labels | 22% of records have a label such as "NOT SPECIFIED" or "N.E.C." |
| Label in text | The product name appears in the narrative for 66% of records. |
| Duplicates | 1,861 records share a narrative with another record. These are short formulaic cases, not errors, so we group on text when splitting. |

**Column decisions**

| Decision | Columns |
|---|---|
| Feature | `Narrative_1` |
| Label | `Product_1` |
| Future use | `Age`, `Body_Part`, `Diagnosis`, `Disposition`, `Fire_Involvement`, `Alcohol`, `Drug` |
| Eliminate | `CPSC_Case_Number`, `Treatment_Date`, `Race`, `Other_Race`, `Hispanic`, `Sex`, `PSU`, `Weight`, `Stratum` |
| Undecided | `Location`, `Product_2`, `Product_3` |

**Assumptions and challenges**
- Only the narrative is available at test time, so every other column is limited to analysis and later work.
- The coder's label is not always a word in the text, so exact-match scoring against `Product_1` will undercount correct extractions.
- Narrative style varies by hospital, so splits should group on `PSU`.

---

## 🧠 **Model Development**

Full plan: [reports/model_logic_plan.md](reports/model_logic_plan.md)

Planned pipeline (v1 is in progress):

```
NEISS narrative → text preprocessing → GLiNER (extract product/object)
→ spaCy / rule-based cleanup and normalization → predicted product → compare with Product_1
```

- **GLiNER** does zero-shot extraction of the object, so the model needs no labeled training set.
- **spaCy** handles preprocessing, normalization of clinical shorthand, and rule-based cleanup.
- Everything runs locally with no API costs.

**Evaluation metrics:** precision, recall, F1 (micro and macro), and exact match accuracy.

---

## 📈 **Results & Key Findings**

Model results will be added after the October validation.

---

## 🚀 **Next Steps**

- Build the v1 extraction pipeline and run it on a sample.
- Normalize clinical shorthand and handle missing values and outliers (Milestone 1 findings).
- Validate against `Product_1` and review cases where the label is not in the text.
- Refine the model and evaluate it on Marsh's workers' compensation descriptions.
- Stretch goal: explore other NEISS fields, including NEISS Work.

---

## 📝 **License**

To be decided with our Challenge Advisor.

---

## 📄 **References**

- [CPSC NEISS injury data and documentation](https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data)
- [GLiNER](https://github.com/urchade/GLiNER)
- [spaCy 101](https://spacy.io/usage/spacy-101)

---

## 🙏 **Acknowledgements**

Thank you to our Challenge Advisor, Kamraan Kamal (Marsh McLennan), and our AI Studio Coach, Ayush Amberkar.
