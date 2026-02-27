# Gold Simple Linear Regression

A **web-based application** for analyzing **gold price data** using **Simple Linear Regression**, built with **Flask and Python**.

This application allows users to upload CSV files and automatically generate:
- Descriptive statistics
- Simple linear regression models
- Model evaluation metrics (MSE, R²)
- Data previews and visualizations

---

## 🚀 Features

- Upload CSV files containing gold price data  
- Descriptive statistical analysis  
- Simple Linear Regression (with and without data splitting)  
- Model evaluation using MSE and R²  
- Tabular data previews (raw, regression, prediction)  
- Automatic visualization of regression results  

---

## 🛠️ Tech Stack

- **Python**
- **Flask**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Gunicorn**

---

## 📂 Project Structure

```text
gold-linear-regression/
├── app.py
├── requirements.txt
├── model/
│   └── regression.py
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── *.png
└── README.md
