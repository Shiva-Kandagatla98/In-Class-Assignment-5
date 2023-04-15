
# Question - 1
#
# 1. Principal Component Analysis
#     a. Apply PCA on CC dataset.
#     b. Apply k-means algorithm on the PCA result and report your observation if the silhouette score has improved or not?
#     c. Perform Scaling+PCA+K-Means and report performance

# Reading dataset CC.csv
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('CC GENERAL.csv')

# Replace null values with mean
df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].mean(), inplace=True)
df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].mean(), inplace=True)

# Apply PCA on dataset
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.iloc[:, 1:18])
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
final_df = pd.concat([pca_df, df[['TENURE']]], axis=1)
print(final_df)

# Split into train and test sets
X = final_df.drop('TENURE', axis=1).values
y = final_df['TENURE'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train logistic regression model on training set
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
train_accuracy = metrics.accuracy_score(y_train, y_train_pred) * 100
print('Accuracy on training set with PCA: %.4f %%' % train_accuracy)

# Apply k-means clustering on original data and calculate Silhouette score
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(df.iloc[:, 1:18])
y_cluster_kmeans = kmeans.predict(df.iloc[:, 1:18])
score = metrics.silhouette_score(df.iloc[:, 1:18], y_cluster_kmeans)
print('Silhouette Score for original data with k-means: ', score)

# Apply scaling, PCA, and k-means clustering on data and calculate Silhouette score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.iloc[:, 1:18])
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)
pca_df2 = pd.DataFrame(data=X_pca2, columns=['PC1', 'PC2'])
final_df2 = pd.concat([pca_df2, df[['TENURE']]], axis=1)
n_clusters = 2
kmeans2 = KMeans(n_clusters=n_clusters)
kmeans2.fit(X_scaled)
y_cluster_kmeans2 = kmeans2.predict(X_scaled)
score2 = metrics.silhouette_score(X_scaled, y_cluster_kmeans2)
print('Silhouette Score for scaled data with PCA and k-means: ', score2)
