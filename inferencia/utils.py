import pandas as pd
import numpy as np
from sklearn import pipeline


DEBUG = True

# Asegúrate de que 'binarias' esté incluida o pasada
binarias = ['Gender', 'OverTime']

EXPLICAR_COL = "Explicar"
TARGET = 'Attrition'


# Verificar que todas las columnas necesarias existen
categoricas = [
    'MaritalStatus', 'BusinessTravel', 'Department', 'RelationshipSatisfaction',
    'JobSatisfaction', 'JobInvolvement', 'StockOptionLevel', 'WorkLifeBalance',
    'EnvironmentSatisfaction', 'JobLevel', 'Education', 'EducationField',
    'TrainingTimesLastYear', 'JobRole'
]

numericas = [
    'NumCompaniesWorked', 'PercentSalaryHike', 'YearsSinceLastPromotion',
    'YearsWithCurrManager', 'YearsInCurrentRole', 'DistanceFromHome',
    'YearsAtCompany', 'TotalWorkingYears', 'Age', 'HourlyRate', 'DailyRate',
    'MonthlyIncome', 'MonthlyRate', 'PerformanceRating'
]


expected_cols = set(categoricas + binarias + numericas)


def convertir_binarias(X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=binarias)
    X_bin = X.copy()
    mapping = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 1: 1, 0: 0}
    for col in X_bin.columns:
        unknown_values = set(X_bin[col].dropna().unique()) - set(mapping.keys())
        if unknown_values:
            raise ValueError(f"❌ Valores inesperados en '{col}': {unknown_values}")

        X_bin[col] = X_bin[col].map(mapping).astype(int)
    return X_bin.values

def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
