# 📄 ACIS Insurance Risk Analytics & Predictive Modeling

This project delivers an end-to-end insurance risk analytics solution, featuring **hypothesis testing** and **ML-driven premium optimization** for **AlphaCare Insurance Solutions (ACIS)**.

It focuses on data engineering, statistical modeling, and ML-driven pricing strategies using real-world auto insurance data from South Africa.

---

## 📌 Project Objectives

The core mission of this analytics pipeline is to provide actionable insights and robust predictive capabilities to enhance profitability and risk management.

* **Customer Segmentation:** Discover and profile **low-risk customer segments** for targeted marketing.
* **Profitability Mapping:** Identify the most **profitable regions** and **vehicle types**.
* **Predictive Modeling:** Build and deploy models for two critical insurance metrics:
    * **Claim Occurrence** (Classification: predicting if a claim will happen).
    * **Claim Severity** (Regression: predicting the size/cost of a claim).
* **Premium Optimization:** Develop an optimization framework to suggest **data-backed, competitive premiums**.
* **MLOps & Reproducibility:** Establish a fully reproducible analytics pipeline using **Git, DVC, CI/CD**, and structured **ML engineering** practices.

---

## 📁 Project Structure

The repository is structured to separate raw data, analytical notebooks, production-grade code, and results for clarity and MLOps compliance.

| Directory | Purpose | Key Contents |
| :--- | :--- | :--- |
| `data/` | Data Storage & Versioning | Raw and processed datasets (DVC-tracked). |
| `notebooks/` | Exploration & Analysis | **EDA**, statistical analysis, and initial modeling experiments (`eda.ipynb`). |
| `src/` | Production Code | Core **Python modules** for data cleaning, feature engineering, and final model training/prediction. |
| `reports/` | Documentation & Submissions | Interim and final project reports/submissions. |
| `visuals/` | Plot Exports | Exported high-quality plots and visual summaries. |
| `configs/` | Configuration Files | **YAML** files for DVC pipeline definitions, model hyper-parameters, and training configurations. |
| `README.md` | Project Overview | This file. |

---

## 🧪 Technologies

This project leverages a modern, open-source data science and MLOps stack.

* **Programming:** **Python**
* **Data Manipulation:** **Pandas, NumPy**
* **Visualization:** **Matplotlib, Seaborn**
* **Statistics:** **SciPy, StatsModels** (for rigorous hypothesis testing)
* **Machine Learning:** **Scikit-Learn, XGBoost**
* **MLOps:** **DVC (Data Version Control)**
* **Version Control & Automation:** **Git, GitHub Actions** (for CI/CD)
* **Environment:** **Jupyter Notebooks**

---

## 🚀 Tasks (Overview)

The project is structured into four sequential tasks, building from initial exploration to final deployment readiness.

### **Task 1 (Current Focus: Setup & Discovery)**

* ✅ Git + GitHub Setup: Initialize repository and establish version control.
* Exploratory Data Analysis (EDA): Deep dive into the data.
* Statistical Analysis: Analyze statistical distributions of key variables.
* Insight Generation: Create focused visualizations to drive business insights.

### **Task 2 (Data Integrity & Versioning)**

* Data Version Control: Implement **DVC** to version and manage all datasets (`data/`).
* Pipeline Definition: Define the data preprocessing pipeline in DVC for full reproducibility.

### **Task 3 (Statistical Validation & Hypothesis Testing)**

* A/B Hypothesis Testing: Conduct formal statistical tests (e.g., t-tests, ANOVA) to validate business hypotheses.
* Statistical Validation: Rigorously validate feature importance and variable significance.

### **Task 4 (Modeling & Optimization)**

* Predictive Modeling: Train and evaluate classification (occurrence) and regression (severity) models.
* Explainability: Apply **SHAP values** to interpret model predictions and ensure fairness/transparency.
* Premium Optimization: Integrate the predictive models into a framework for **data-driven premium suggestions**.

---

## 📌 How to Run

Follow these simple steps to set up the project environment and run the analysis notebooks.

1.  **Clone the Repository:**
    ```bash
    git clone [repository-url]
    cd acis-insurance-risk-analytics
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    ```
4.  Navigate to the `notebooks/` directory and open `eda.ipynb` to begin your analysis.

---
