# Breast Cancer Logistic Regression by Zach W.


# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Load Data
df = pd.read_csv("wdbc.data")
df.columns = ["ID", "Diagnosis", "radius1", "texture1", "perimeter1", "area1", "smoothness1", "compactness1", "concavity1", "concave_points1", "symmetry1", "fractal_dimension1", "radius2", "texture2", "perimeter2", "area2", "smoothness2", "compactness2", "concavity2", "con2cave_points", "symmetry2", "fractal_dimension2", "radius3", "texture3", "perimeter3", "area3", "smoothness3", "compactness3", "concavity3", "concave_points3", "symmetry3", "fractal_dimension3"]
column_list = df.columns.values

# Exploratory Analysis
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())


# Change M = malignant, B = benign to M = 1, B = 0 so that all our columns are numeric
df["Diagnosis"] = df["Diagnosis"].astype("category")
df["Diagnosis"] = df["Diagnosis"].cat.codes
#print(df.head())


# Create our X and y
X = df.drop(columns = ["ID", "Diagnosis"])
#X = df[["radius3", "concavity3", "radius1"]]
y = df["Diagnosis"]


# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .8, test_size = .2, random_state = 27)

print(X_train)
print(X_test)


# Scale
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Fit model
lr = LogisticRegression()

lr.fit(X_train_scaled, y_train)

lr.predict(X_train_scaled)

print("train score:", lr.score(X_train_scaled, y_train))
print("test score:", lr.score(X_test_scaled, y_test))


# Print the model coefficients
model_coef = lr.coef_
print("model coefficients:\n", model_coef)



# Find and print features with greatest impact
abs_model_coef = abs(model_coef)
abs_model_coef = abs_model_coef.flatten()

column_list = column_list[2:]

for i in range(len(column_list)):
    if abs_model_coef[i] > 1.0:
        print(column_list[i])
    
    




        

