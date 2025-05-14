import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
import pickle

"""# Clustering Countries for Strategic Aid Allocation"""

df=pd.read_csv("Country-data.csv")

df

df.shape

df.info()

df.describe()

df.duplicated().sum()

"""All columns have 0 missing values and 0% missing percentage.

# Mean imputation
"""

numerical_columns=df.select_dtypes(include=["number"])
df.fillna(numerical_columns.mean(), inplace=True)

def detect_outliers(df,features):
    outliers={}
    for feature in features:
        q1=df[feature].quantile(0.25)
        q3=df[feature].quantile(0.75)
        IQR=q3-q1
        lower_limit=q1-(1.5*IQR)
        upper_limit=q3+(1.5*IQR)
        outliers[feature]=df[(df[feature] < lower_limit) | (df[feature] > upper_limit )]
        return outliers

features_to_check = ['child_mort', 'income', 'gdpp']
outliers = detect_outliers(df, features_to_check)
print("Outliers detected:\n", outliers)

"""# Univariate Analysis"""

def plot_distributions(df, features):
    for feature in features:
        plt.figure(figsize=(15, 3))
        plt.subplot(1, 2, 1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Histogram of {feature}')

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[feature])
        plt.title(f'Boxplot of {feature}')

        plt.show()

plot_distributions(df, ['child_mort', 'income', 'life_expec'])

df_numeric = df.select_dtypes(include=['number'])

correlation_matrix = df_numeric.corr()

plt.figure(figsize=(10, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(6,6))
sns.pairplot(df_numeric[['health', 'life_expec', 'income', 'child_mort']])
plt.show()

# Hypothesis: Increased health spending leads to higher life expectancy
high_health_spending = df[df['health'] > df['health'].median()]['life_expec']
low_health_spending = df[df['health'] <= df['health'].median()]['life_expec']

t_stat, p_value = stats.ttest_ind(high_health_spending, low_health_spending)
print(f"T-test for Health Spending and Life Expectancy: t_stat={t_stat}, p_value={p_value}")

# Hypothesis: Countries with higher Total_fertility rates have lower Income per person
correlation_fertility_income, _ = stats.pearsonr(df['total_fer'], df['income'])
print(f"Correlation between Fertility and Income: {correlation_fertility_income}")

high_fertility = df[df['total_fer'] > df['total_fer'].median()]['income']
low_fertility = df[df['total_fer'] <= df['total_fer'].median()]['income']

t_stat, p_value = stats.ttest_ind(high_fertility, low_fertility)
print(f"T-test for Fertility and Income: t_stat={t_stat}, p_value={p_value}")

# Hypothesis: Higher income levels are associated with lower child mortality rates
correlation_income_child_mortality, _ = stats.pearsonr(df['income'], df['child_mort'])
print(f"Correlation between income and child Mort: {correlation_income_child_mortality}")

high_income = df[df['income'] > df['income'].median()]['child_mort']
low_income = df[df['income'] <= df['income'].median()]['child_mort']

t_stat, p_value = stats.ttest_ind(high_income, low_income)
print(f"T-test for Income and Child Mortality: t_stat={t_stat}, p_value={p_value}")

# Hypothesis: Higher inflation rates are associated with lower GDP per capita
correlation_inflation_gdp, _ = stats.pearsonr(df['inflation'], df['gdpp'])
print(f"Correlation between Inflation and GDP per Capita: {correlation_inflation_gdp}")

high_inflation = df[df['inflation'] > df['inflation'].median()]['gdpp']
low_inflation = df[df['inflation'] <= df['inflation'].median()]['gdpp']

t_stat, p_value = stats.ttest_ind(high_inflation, low_inflation)
print(f"T-test for Inflation and GDP per Capita: t_stat={t_stat}, p_value={p_value}")

"""# ML Modeling

Data Preprocessing

Missing Value Imputation with Median
"""

numeric_df = df.select_dtypes(include=[np.number])
non_numeric_df = df.select_dtypes(exclude=[np.number])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
numeric_df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)

df_imputed = pd.concat([numeric_df_imputed, non_numeric_df.reset_index(drop=True)], axis=1)
df_imputed = df_imputed[df.columns]
df_imputed.head()

non_numeric_df_dropped = non_numeric_df.dropna()
non_numeric_df_dropped

from sklearn.preprocessing import StandardScaler

# Normalize numerical features
scaler = StandardScaler()
numeric_df_normalized = pd.DataFrame(scaler.fit_transform(numeric_df_imputed), columns=numeric_df_imputed.columns)

# Combine the normalized numeric data with the non-numeric data
df_normalized = pd.concat([numeric_df_normalized, non_numeric_df_dropped.reset_index(drop=True)], axis=1)

from sklearn.preprocessing import OneHotEncoder

# One-hot encode non-numerical variables
encoder = OneHotEncoder(sparse_output=False, drop='first')
non_numeric_encoded = encoder.fit_transform(non_numeric_df_dropped)
non_numeric_encoded_df = pd.DataFrame(non_numeric_encoded, columns=encoder.get_feature_names_out(non_numeric_df_dropped.columns))

# Combine the encoded non-numerical data with the normalized numerical data
df_final = pd.concat([numeric_df_normalized, non_numeric_encoded_df.reset_index(drop=True)], axis=1)

# Feature Engineering
df_final['High_Child_Mort'] = (df['child_mort'] > 50).astype(int)  # Example threshold for high child mortality
df_final['Exports_Imports_Ratio'] = df['exports'] / df['imports']

# Ensure all columns are in the final DataFrame
df_final = df_final.reset_index(drop=True)

df_final.head(10)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Example with 5 clusters
kmeans_labels = kmeans.fit_predict(df_final)

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=5)
hierarchical_labels = hierarchical.fit_predict(df_final)

# DBSCAN - Try adjusting eps and min_samples
dbscan = DBSCAN(eps=5, min_samples=3)  # Increased eps, decreased min_samples
dbscan_labels = dbscan.fit_predict(df_final)

# Evaluate Clustering Performance (only calculate if there are at least 2 clusters)
if len(set(dbscan_labels)) > 1:
    dbscan_silhouette = silhouette_score(df_final, dbscan_labels)
    print(f'DBSCAN Silhouette Score: {dbscan_silhouette}')
else:
    print("DBSCAN resulted in a single cluster. Adjust parameters.")

# Evaluate Clustering Performance
kmeans_silhouette = silhouette_score(df_final, kmeans_labels)
hierarchical_silhouette = silhouette_score(df_final, hierarchical_labels)

print(f'K-Means Silhouette Score: {kmeans_silhouette}')
print(f'Hierarchical Clustering Silhouette Score: {hierarchical_silhouette}')

# Visualization using PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_final)

plt.figure(figsize=(18, 5))

# K-Means
plt.subplot(1, 3, 1)
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')

# Hierarchical Clustering
plt.subplot(1, 3, 2)
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=hierarchical_labels, cmap='viridis')
plt.title('Hierarchical Clustering')

# DBSCAN (only plot if there are at least 2 clusters)
if len(set(dbscan_labels)) > 1:
    plt.subplot(1, 3, 3)
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=dbscan_labels, cmap='viridis')
    plt.title('DBSCAN Clustering')

with open ('kmeans.pkl','wb') as f :
    pickle.dump(kmeans,f)

!zip -r ./kmeans.pkl.zip ./kmeans-model.pkl

with open ('hierarchical.pkl','wb') as f:
    pickle.dump(hierarchical,f)

!zip -r ./hierarchical.pkl.zip ./hierarchical.pkl

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Elbow Method for K-Means
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_final)
    wcss.append(kmeans.inertia_)

# Silhouette Analysis for K-Means
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    labels = kmeans.fit_predict(df_final)
    silhouette_scores.append(silhouette_score(df_final, labels))

fig, ax = plt.subplots(1, 2, figsize=(10, 3))

# Elbow Method plot
ax[0].plot(range(1, 11), wcss, marker='o')
ax[0].set_title('Elbow Method')
ax[0].set_xlabel('Number of clusters')
ax[0].set_ylabel('WCSS')

# Silhouette Analysis plot
ax[1].plot(range(2, 11), silhouette_scores, marker='o')
ax[1].set_title('Silhouette Analysis')
ax[1].set_xlabel('Number of clusters')
ax[1].set_ylabel('Silhouette Score')

plt.tight_layout()
plt.show()
