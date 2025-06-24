import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

df = pd.read_csv("Titanic-Dataset.csv")
print("Summary using decribe:\n")
print(df.describe())

numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
for col in numeric_cols:
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f'{col} Histogram')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.title(f'{col} Boxplot')

    plt.tight_layout()
    plt.show()

sns.pairplot(df[numeric_cols + ['Survived']].dropna(), hue='Survived')
plt.suptitle("Pairplot", y=1.02)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols + ['Survived']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

print("\nMissing values:\n", df.isnull().sum())
print("\nClass survival rate:\n", df.groupby('Pclass')['Survived'].mean())

sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Sex survival rate")
plt.show()

sns.barplot(x='Embarked', y='Survived', data=df)
plt.title("Embarked survival rate")
plt.show()

sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Passenger class survival rate")
plt.show()

df_cleaned = df.dropna(subset=['Age', 'Fare', 'Embarked'])
df_cleaned['Cabin'] = df_cleaned['Cabin'].fillna('Unknown')
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned.to_csv("Titanic-Dataset-Cleaned.csv", index=False)

print("\nCleaned dataset saved as 'Titanic-Dataset-Cleaned.csv'")
