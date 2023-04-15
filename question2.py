# 2. Use pd_speech_features.csv
#     a. Perform Scaling
#     b. Apply PCA (k=3)
#     c. Use SVM to report performance

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load dataset
data = pd.read_csv("pd_speech_features.csv")
features = data.drop('class', axis=1).values
labels = data['class'].values

# Apply scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# PCA
pca = PCA(n_components=3)
pca_components = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(data=pca_components,
                      columns=['PC 1', 'PC 2', 'PC 3'])
final_df = pd.concat([pca_df, data[['class']]], axis=1)
print('Final dataset after PCA:')
print(final_df)

# Train and evaluate SVM model
X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.3, random_state=0)
svm_model = SVC(max_iter=1000)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
svm_acc = round(svm_model.score(X_train, y_train) * 100, 2)
print("SVM accuracy =", svm_acc)
