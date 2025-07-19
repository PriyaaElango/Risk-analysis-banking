import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sqlalchemy import create_engine

# Create a SQLAlchemy engine
engine = create_engine("mysql+mysqlconnector://root:root@localhost:3306/banking_case")

# Run SQL query and read into DataFrame
query = "SELECT * FROM customer"
df = pd.read_sql(query, engine)

#print (df.shape)
print(df.describe(include='all'))

# Create Income Band column
bins = [0, 100000, 300000, float('inf')]
labels = ['Low', 'Med', 'High']
df['Income Band'] = pd.cut(df['Estimated Income'], bins=bins, labels=labels, right=False)

# Plot bar chart of Income Band distribution
df['Income Band'].value_counts().plot(kind='bar')
plt.title('Income Band Distribution')
plt.xlabel('Income Band')
plt.ylabel('Count')
plt.show()
plt.close()

# Define categorical columns once
categorical_cols = ["BRId", "GenderId", "IAId", "Amount of Credit Cards", "Nationality", "Occupation",
                    "Fee Structure", "Loyalty Classification", "Properties Owned", "Risk Weighting", "Income Band"]

# Examine the distribution of unique categories in categorical columns
for col in categorical_cols:
    print(f"Value Counts for '{col}':")
    print(df[col].value_counts())
    print()

# Univariate Analysis: countplots for categorical columns
for i, predictor in enumerate(categorical_cols):
    plt.figure(i)
    sns.countplot(data=df, x=predictor)
    plt.xticks(rotation=45)
    plt.title(f'Countplot of {predictor}')
    plt.tight_layout()
    plt.show()
    plt.close()

# Bivariate Analysis: countplots with hue='Nationality'
for i, predictor in enumerate(categorical_cols):
    plt.figure(i)
    sns.countplot(data=df, x=predictor, hue='Nationality')
    plt.xticks(rotation=45)
    plt.title(f'Countplot of {predictor} by Nationality')
    plt.tight_layout()
    plt.show()
    plt.close()

# Histplot of value counts for different categorical columns (except Occupation)
for col in categorical_cols:
    if col == "Occupation":
        continue
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col])
    plt.title(f'Histogram of {col} Count')
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()

# Numerical columns for analysis
numerical_cols = ['Estimated Income', 'Superannuation Savings', 'Credit Card Balance', 'Bank Loans',
                  'Bank Deposits', 'Checking Accounts', 'Saving Accounts', 'Foreign Currency Account', 'Business Lending']

# Univariate analysis and visualization of numerical columns
plt.figure(figsize=(18, 12))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 3, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.tight_layout()

plt.show()
plt.close()
