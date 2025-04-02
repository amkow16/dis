import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
 
#Load California Housing dataset
#data = fetch_california_housing()
df = pd.read_csv('F:\CollegeSubjects\ML_24-25\LabPrograms\housing.csv')
#Convert to DataFrame
 
#df = pd.DataFrame(data.data, columns=data.feature_names)
print(df)
df['Target'] = df.median_house_value
 # Adding the target variable (median house value)
 
print("\nBasic Information about Dataset:")
print(df.info())  # Overview of dataset
print("\nFirst Five Rows of Dataset:")
print(df.head())  # Display first few rows
print("\nSummary Statistics:")
print(df.describe())  # Summary statistics of dataset
 
print("\nMissing Values in Each Column:")
print(df.isnull().sum())  # Count of missing values
plt.figure(figsize=(12, 8))
df.hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Boxplots of Features to Identify Outliers")
plt.show()
plt.figure(figsize=(10, 6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()
sns.pairplot(df[['median_income', 'housing_median_age', 'total_bedrooms', 'Target']], diag_kind='kde')
plt.show()