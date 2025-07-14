# Blueberry Yield Prediction using Linear Models and SGDRegressor

This project explores the prediction of blueberry yields using a progression of linear modeling techniques. Starting from basic linear regression, the workflow evolves into more structured and optimized machine learning using pipelines and hyperparameter tuning with `GridSearchCV`.

## 📌 Project Description

The goal is to predict blueberry yield based on agricultural and environmental features from the Kaggle dataset. Multiple modeling approaches are explored to show performance gains through scaling, regularization, and parameter tuning.

## 🚀 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Seaborn (for visualization)  

## 🛠️ Modeling Approaches Explored

1. **Linear Regression** — baseline model  
2. **Linear Regression + Scaling** — with `StandardScaler`  
3. **SGDRegressor** — regression with stochastic gradient descent  
4. **SGDRegressor + Scaling** — manually scaled features  
5. **Pipeline (SGDRegressor + Scaler)** — clean, modular structure  
6. **GridSearchCV + Scaling** — hyperparameter tuning without pipeline  
7. **Pipeline + GridSearchCV** — full optimization in one reusable workflow  

## ⚙️ Hyperparameters Tuned via GridSearchCV

A two-part parameter grid was used to explore both regular and ElasticNet penalties:

```python
params = [
    {
        'model__penalty': ['l1', 'l2', None],
        'model__eta0': [0.001, 0.01, 0.1, 1],
        'model__alpha': [0.001, 0.01, 0.1, 1],
        'model__max_iter': [100, 500, 1000]
    },
    {
        'model__penalty': ['elasticnet'],
        'model__eta0': [0.001, 0.01, 0.1, 1],
        'model__alpha': [0.001, 0.01, 0.1, 1],
        'model__max_iter': [100, 500, 1000],
        'model__l1_ratio': [0.15, 0.5, 0.85]
    }
]
```
## 🧪 Evaluation Metrics

To evaluate and compare model performance:

- **Mean Squared Error (MSE)** — Measures average squared difference between predicted and actual values  
- **Root Mean Squared Error (RMSE)** — Square root of MSE; easier to interpret in original units  
- **R² Score** — Proportion of variance explained by the model  
- **Cross-validation score** — Average performance across folds (optional with GridSearchCV)

## 📂 Dataset

- From Kaggle: [2022 USDA Blueberry Yield Forecasting](https://www.kaggle.com/competitions/blueberry-regression-sentence)  
