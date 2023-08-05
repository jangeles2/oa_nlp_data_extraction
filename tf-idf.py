# Code adapted from scikit learn and Bashkeel's BoW-TFIDF repository

# Sources
# Scikit Learn TfidfVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html retrieved in June 2023
# Scikit Learn TSNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html retrieved in June 2023
# Repository https://github.com/Bashkeel/BoW-TFIDF/blob/master/TF-IDF%20and%20N-Grams.ipynb retrieved in June 2023

# Term Frequency - Inverse Document Frequency Model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load data
data = pd.read_csv('open_ave_data.csv')
# Replace NaN values with empty strings
data.fillna('', inplace=True)
# Create corpus
findings = data["findings"].values.tolist()
clinical = data["clinicaldata"].values.tolist()
exam = data["ExamName"].values.tolist()
impression = data["impression"].values.tolist()
corpus_docs = findings + clinical + exam + impression
# Fit the TfidfVectorizer
vectorizer = TfidfVectorizer()
tfidf_documents = vectorizer.fit_transform(corpus_docs)
print('Feature Names:',vectorizer.get_feature_names_out())
print('TF-IDF Shape:', tfidf_documents.shape)
print('TF-IDF array:')
print(tfidf_documents.toarray())
# Color code each column
section_colors = {
    'findings': 'green',
    'clinical': 'yellow',
    'exam': 'red',
    'impression': 'blue'
}
# Combine the TF-IDF documents
combined_tfidf_documents = tfidf_documents
colors = np.concatenate([
    np.full(len(findings), section_colors['findings']),
    np.full(len(clinical), section_colors['clinical']),
    np.full(len(exam), section_colors['exam']),
    np.full(len(impression), section_colors['impression'])
])
# t-SNE: Dimensionality reduction to 2D
tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
combined_tfidf_documents_embedded = tsne.fit_transform(combined_tfidf_documents.toarray())
# Create a scatter plot for TF-IDF
scatter = plt.scatter(combined_tfidf_documents_embedded[:, 0], combined_tfidf_documents_embedded[:, 1], c=colors)
# Legend
legend_patches = [
    mpatches.Patch(color='green', label='Findings'),
    mpatches.Patch(color='yellow', label='Clinical Data'),
    mpatches.Patch(color='red', label='Exam Name'),
    mpatches.Patch(color='blue', label='Impression')
]
plt.legend(handles=legend_patches)
plt.title('2D TF-IDF Scatter Plot')
plt.show()
# Split the dataset into train and test data
sections = np.concatenate((exam, clinical, findings, impression))
labels = np.concatenate((np.zeros(len(exam)), np.ones(len(clinical)), np.full(len(findings), 2), np.full(len(impression), 3)))
X_train, X_test, y_train, y_test = train_test_split(tfidf_documents, labels, test_size=0.2, random_state=42)
print("Train data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
print("Train labels shape:", y_train.shape)
print("Test labels shape:", y_test.shape)
# Train a logistic regression model
lr = LogisticRegression(C=1, solver='saga')
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
# Accuracy
tf_idf_accuracy = accuracy_score(y_test, lr_preds)
print("\nTF-IDF Accuracy:", tf_idf_accuracy)
# Confusion Matrix
print("\nTF-IDF Confusion Matrix:")
print(confusion_matrix(y_test, lr_preds))
# Classification report
report = classification_report(y_test, lr_preds, target_names=['Findings', 'Clinical Data', 'Exam Name', 'Impression'])
print("\nClassification Report:")
print(report)
