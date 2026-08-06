---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, which will allow you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---

## 📋 BTT Internal Evaluation Notes
*(This section is for BTT staff only — remove before sharing with students)*

| Check | Status | Notes |
|-------|--------|-------|
| Python Compatibility | 🟢 | The project focuses on using Python for NLP tasks, ensuring compatibility with students' skill sets in ML Foundations. |
| Data Readiness | 🟢 | The dataset is accessible and already structured in Excel format, making it ready for use with minimal cleaning required. |
| Resource Check | 🟢 | Free-tier tools like Google Colab are adequate for the project's needs and pose no barriers for students. |

**Student Fit Score:** 8/10  
**Technical Depth Score:** 7/10  
**Overall Recommendation:** REVISE

**Advisor Feedback Draft:**
The project leverages real-world data, providing a valuable learning opportunity. However, ensure students are equipped with the necessary NLP knowledge early on. Additionally, consider simplifying the model's expectations for iterative development to allow space for learning. Encourage students to document the challenges they face in model development for reflective learning. Overall, this project appears to be a suitable fit for the fellows with some adjustments.

---


# Marsh Object Detection Model

**Company / Org:** Marsh Mclennan  
**Challenge Advisor:** Kamraan Kamal, kamraan.kamal@marsh.com  
**Program:** Break Through Tech AI Studio - Fall 2026

---

## 🏢 About Marsh Mclennan

Marsh McLennan is a global leader in insurance brokerage and risk management, helping clients identify and manage risk effectively. Our initiatives aim to improve workplace safety and reduce liability through data-driven insights and innovative solutions.

---

## 🎯 The Challenge

### Project Summary
In this project, you will use hospital emergency patient injury descriptions from the U.S. Consumer Product Safety Commission’s National Electronic Injury Surveillance System (CPSC NEISS), Natural Language Processing (NLP) and Large Language Model (LLM) techniques to extract the objects involved in injury incidents. This will help our company address the business problem of workplace and product-related injury prevention, reducing injury-related costs, and improving safety and loss prevention strategies.

### Success Criteria
We will provide the students with a final testing set of anonymized workers' compensation accident descriptions. Their model should be able to correctly extract and classify the objects mentioned in the descriptions.

We will also compare it to our existing object detection model to see if the students' model performs favorably.

### Stretch Goal
If the students progress quickly enough, they can explore other fields mentioned in the NEISS (specifically NEISS Work) database.

### Project Milestones

Use these milestones to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

| Month | Milestone | Key Activities |
| :--- | :--- | :--- |
| September | Data Ingestion, Preprocessing & Baseline Text Classification | • Ingest and clean National Electronic Injury Surveillance System (NEISS) emergency room narrative datasets.<br>• Perform Exploratory Data Analysis (EDA) on product codes, injury narratives, demographics, and severity classifications.<br>• Preprocess text narratives (tokenization, lemmatization, stop-word removal) and extract baseline TF-IDF / keyword features.<br>• Train baseline classifiers (Logistic Regression / Naive Bayes) to identify involved consumer products and hazards. |
| October | Entity Extraction, NLP Modeling & Risk Classification | • Implement Named Entity Recognition (NER) or fine-tune transformer models (e.g., BioBERT, RoBERTa) to extract target objects, injury mechanisms, and affected body parts from text narratives.<br>• Train advanced multi-class models (XGBoost, LightGBM, or BERT-based classifiers) to categorize product-related risk severity.<br>• Conduct hyperparameter tuning, cross-validation, and evaluate performance using Precision, Recall, and F1-Score. |
| November / December | Model Interpretation, Interactive UI & Capstone Deliverables | • Apply model interpretability techniques (SHAP / LIME) to highlight key narrative trigger phrases driving product risk classifications.<br>• Build an interactive Streamlit application enabling users to input clinical narrative text and view real-time object detection and hazard severity scores.<br>• Finalize clean, reproducible GitHub repository, comprehensive technical documentation, and final presentation deck. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

**Name and Source:** National Electronic Injury Surveillance System (CPSC NEISS)  
**Format:** Excel (.xlsx)  
**Size:** 1gb to 5gb  
**Location:** https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data

### Key Details
- [Brief description of what's in the data]
- [Any known limitations or preprocessing needed]
- [Link to data dictionary or documentation, if available]

---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification, NLP,Large Language Models (LLMs)/ Generative AI

**Recommended Libraries:**
- [e.g., pandas, scikit-learn, TensorFlow, Hugging Face]

**Evaluation Metrics:**
- [e.g., Accuracy, Precision/Recall, RMSE, BLEU score]

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [e.g., Link to an article or blog post about the problem domain]
- [e.g., Link to an industry report or case study]

**Technical Tutorials:**
- [e.g., Link to a free tutorial on the ML technique(s) involved]
- [e.g., Link to documentation for a key library or tool]

**Code Examples:**
- [e.g., Link to a relevant GitHub repo]
- [e.g., Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [e.g., Your team's channel within Break Through Tech’s Discord space]
* [e.g., Email; please copy your teammates and AI Studio Coach]
* [e.g., Request a team check-in on Zoom]
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
