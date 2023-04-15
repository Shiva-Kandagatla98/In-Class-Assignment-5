# 3. Apply Linear Discriminant Analysis (LDA) on Iris.csv dataset to reduce dimensionality of data tok=2.

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set(style="white", color_codes=True)


import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import matplotlib.pyplot as plt

# Load iris dataset
df = pd.read_csv("iris.csv")

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(df.iloc[:, :4].values)

# Encode class labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Species'].values)

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Create new dataframe with LDA components and class labels
data = pd.DataFrame(X_lda, columns=['LD1', 'LD2'])
data['class'] = y

print(data)

# 4. Briefly identify the difference between PCA and LDA.

# PCA (Principal Component Analysis) and LDA (Linear Discriminant Analysis) are both linear transformation techniques used for dimensionality reduction. However, they differ in their objectives and applications.

# PCA is an unsupervised technique that seeks to reduce the dimensionality of a dataset while preserving most of its variance. It does so by transforming the data into a new coordinate system where the new axes (principal components) are orthogonal and ordered by the amount of variance they capture. PCA is often used for data visualization, data compression, and feature extraction.

# LDA, on the other hand, is a supervised technique that seeks to find a linear combination of features that maximizes the separation between different classes of data. It does so by finding a projection that maximizes the between-class scatter and minimizes the within-class scatter. LDA is often used for classification and feature extraction.

# In summary, PCA is an unsupervised technique that preserves the maximum amount of variance in the data while reducing dimensionality, while LDA is a supervised technique that seeks to maximize the separation between classes by finding a projection that preserves the class structure of the data.