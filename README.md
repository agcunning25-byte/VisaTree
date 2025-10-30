# 🧠 EasyVisa: Automated Visa Approval Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Made%20with-Python-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Modeling-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![UT Austin](https://img.shields.io/badge/University%20of%20Texas%20at%20Austin-Great%20Learning-CC5500?logo=graduation-cap&logoColor=white)

---

## 🎓 Project Overview

This project was developed as part of my **Postgraduate Diploma in Machine Learning and Artificial Intelligence for Business Applications** from the **University of Texas at Austin**, in partnership with **Great Learning**.

**EasyVisa** is a machine learning solution that predicts the likelihood of a **visa application being approved or denied** based on applicant demographics, work history, and prior visa outcomes. The project demonstrates how **AI-driven decision systems** can enhance efficiency, reduce bias, and improve transparency in immigration and HR analytics.

---

## 🎯 Objective

The goal of EasyVisa is to design a supervised learning pipeline that automates visa decision prediction.  
By analyzing historical visa data, we aim to help organizations and consultants forecast application outcomes more accurately.

### Key predictive factors include:
- Education Level  
- Occupation Type  
- Country of Origin  
- Prior Visa Status  
- Sponsorship Type  
- Offered Wage  

---

## 🧩 Methodology

This project followed a full **data science lifecycle** — from data acquisition to deployment readiness:

1. **Data Collection & Exploration**
   - Loaded real-world immigration-style datasets with visa and applicant attributes.  
   - Conducted statistical summaries and data profiling.

2. **Data Preprocessing**
   - Encoded categorical variables using One-Hot and Label Encoding.  
   - Scaled numeric features and addressed missing values.  
   - Split data into **train/test** sets for validation.

3. **Exploratory Data Analysis (EDA)**
   - Examined relationships between features and visa outcomes using **Seaborn** and **Matplotlib**.  
   - Identified strong correlations between education, wage level, and approval likelihood.

4. **Model Development**
   - Trained and evaluated multiple algorithms:
     - Logistic Regression  
     - Random Forest Classifier  
     - Support Vector Machine (SVM)  
     - Gradient Boosting Classifier  
   - Selected the final model based on cross-validation and ROC-AUC.

5. **Hyperparameter Tuning**
   - Used **GridSearchCV** for optimization of model parameters.  
   - Reduced overfitting and maximized precision-recall balance.

6. **Model Export**
   - Serialized final model using **Joblib** for integration into APIs or dashboards.

---

## 📈 Results and Insights

- **Best Model:** Random Forest Classifier  
- **Accuracy:** ~95%  
- **ROC-AUC:** 0.92  

### Top Insights
- Applicants with advanced education and higher wage offers have a **significantly higher approval rate**.  
- Country of origin influences success rates — policy alignment and employment sponsorship affect results.  
- STEM-designated occupations show higher approval consistency than non-STEM roles.  

---

## 🧰 Tools & Technologies

**Languages:** Python  
**Libraries:** Pandas · NumPy · Matplotlib · Seaborn · Scikit-Learn · Joblib  
**Environment:** Jupyter Notebook  
**Frameworks:** SciPy · Scikit-Learn Pipelines  
**Version Control:** GitHub  

---

## 🪙 Key Takeaways

✅ Built a complete **machine learning classification pipeline** for business decision automation.  
✅ Demonstrated effective **EDA**, **feature engineering**, and **model tuning**.  
✅ Delivered data-driven insights for real-world policy and HR applications.  
✅ Showcased practical understanding of **AI ethics**, fairness, and interpretability.  

---

## 🗂️ Repository Structure
EasyVisa-ML-Project/
* Adam_Cunningham_EasyVisa_Full_Code_Notebook.html
* Adam_Cunningham_EasyVisa_Full_Code_Notebook.ipynb
* data/ (sample dataset used for experimentation)
* README.md

---

## 📚 Academic Context

This project was completed as part of the **Postgraduate Diploma in Machine Learning and Artificial Intelligence for Business Applications** offered by the **University of Texas at Austin**, in collaboration with **Great Learning**.

The curriculum emphasizes hands-on machine learning, deep learning, and applied AI for solving **real-world business challenges**, focusing on interpretability, deployment readiness, and ethical implementation.

---

## 🏆 Acknowledgments

Special thanks to **The University of Texas at Austin** and **Great Learning** faculty for their exceptional mentorship and for providing the structure, datasets, and guidance to bring this project to life.

---

© 2025 **Adam Cunningham** — All Rights Reserved.

