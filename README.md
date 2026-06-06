# 🎓 AI in Education: Helper or Harm?

> **Exploratory data analysis of how GenAI tools affect student performance, burnout, and skill retention**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)
![scipy](https://img.shields.io/badge/scipy-stats-8CAAE6?logo=scipy&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-50%2C000_students-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## About This Project

This is a **pet project** built to sharpen EDA and statistical analysis skills on a topic that feels genuinely important right now — how generative AI is reshaping student learning.

With tools like ChatGPT, Copilot, and Claude becoming part of everyday study routines, a simple question arises: **does using AI actually help students, or does it create hidden costs** — dependency, burnout, and eroded skills?

The analysis covers:
- How usage volume (hours/week) correlates with GPA change and burnout risk
- Whether institutional AI policy (ban vs encourage) affects outcomes
- Which use case — debugging, summarizing, or just getting direct answers — leads to the best results
- Whether prompt engineering skill is a meaningful differentiator
- 5 statistical tests (t-test, ANOVA, Spearman correlation)

**Key finding:** there's an optimal zone of 5–10 hours/week where GPA gains are highest. Beyond 20 hours/week, 74% of students fall into high burnout risk — and GPA growth drops off.

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | [Kaggle — Impact of AI on Students](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students) |
| **Author** | laveshjadon |
| **Rows** | 50,000 students |
| **Features** | 16 columns |
| **Missing values** | 0 |
| **File** | `ai_student_impact_dataset.csv` |

### Feature Overview

| Column | Type | Description |
|--------|------|-------------|
| `Major_Category` | categorical | Field of study: STEM, Business, Humanities, Medical, Arts |
| `Year_of_Study` | categorical | Academic year: Freshman → Graduate |
| `Pre_Semester_GPA` | float | GPA before the semester (1.0–4.0) |
| `Post_Semester_GPA` | float | GPA after the semester (1.0–4.0) |
| `Weekly_GenAI_Hours` | float | Hours per week using AI tools |
| `Primary_Use_Case` | categorical | Main use: Debugging, Ideation, Copywriting, Summarizing, Direct Answers |
| `Prompt_Engineering_Skill` | categorical | Skill level: Beginner / Intermediate / Advanced |
| `Tool_Diversity` | int | Number of different AI tools used |
| `Paid_Subscription` | bool | Paid AI plan or not |
| `Traditional_Study_Hours` | float | Hours/week of non-AI study |
| `Perceived_AI_Dependency` | 1–10 | Self-reported dependency on AI |
| `Institutional_Policy` | categorical | University stance: Strict_Ban / Allowed_With_Citation / Actively_Encouraged |
| `Anxiety_Level_During_Exams` | 1–10 | Exam anxiety score |
| `Skill_Retention_Score` | 0–100 | Retained knowledge score |
| `Burnout_Risk_Level` | categorical | Burnout risk: Low / Medium / High |

The dataset was designed to help researchers, educators, and policymakers explore the **benefits and risks of AI adoption in higher education**.

---

## Structure

```
ai-education-analyst/
├── analysis.md          # Full EDA with code, tables, and charts
└── README.md
```

The analysis is written in Markdown with all Python code blocks included — run cells top to bottom in a Jupyter notebook or extract into a `.py` script.

---

## Key Results

### GPA Change vs AI Usage Hours

| Hours/week | Avg GPA Δ | Burnout High % |
|------------|-----------|----------------|
| 0–2 h | +0.19 | 8.9% |
| 2–5 h | +0.20 | 11.9% |
| **5–10 h** | **+0.23** | **19.0%** |
| 10–20 h | +0.22 | 40.0% |
| 20+ h | +0.16 | **74.3%** |

### Use Case Matters More Than Hours

| Use Case | GPA Δ | Skill Retention |
|----------|-------|-----------------|
| **Debugging / Troubleshooting** | **+0.249** | **78.1** |
| Ideation | +0.200 | 75.5 |
| Copywriting / Drafting | +0.200 | 75.2 |
| Summarizing Reading | +0.197 | 75.2 |
| **Direct Answer Generation** | **+0.133** | **73.7** |

### Prompt Engineering Skill Gap

| Skill Level | GPA Δ | Skill Retention |
|-------------|-------|-----------------|
| Beginner | +0.185 | 71.1 |
| Intermediate | +0.187 | 75.8 |
| **Advanced** | **+0.248** | **82.1** |

> Advanced users outperform Beginners by **+34% in GPA growth** and **+15% in skill retention**.

---

## Stack

```
pandas · numpy · scipy · matplotlib · seaborn
```

```bash
pip install pandas numpy scipy matplotlib seaborn
```

---

## Setup

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students) and place `ai_student_impact_dataset.csv` in the project root.
2. Open `analysis.md` and copy code blocks into a Jupyter notebook.
3. Run cells sequentially — the dataset has zero missing values, no preprocessing surprises.

---

## Author

**Yaroslav Zinchenko** — aspiring data analyst, Python/pandas learner  
Building a portfolio of EDA projects on real-world datasets.

---

*Data source: [Kaggle — Impact of AI on Students](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students) by laveshjadon*
