# Code adapted from "Training Word2vec using gensim"

# Source
# Website: https://swatimeena989.medium.com/training-word2vec-using-gensim-14433890e8e4 retrieved in July 2023

# Word2Vec Model
import pandas as pd
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import re
from tqdm import tqdm
import multiprocessing
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv('open_ave_data.csv')
# Set stopwords
stopwords_list = set(stopwords.words('english'))


# Clean data
def clean_data(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        #text = re.sub(r"([0-9])", r" ", text)
        words = text.split()
        clean_words = [word for word in words if word not in stopwords_list and len(word) > 2]
        return clean_words
    else:
        return []


sentences = []
for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
    findings_cleaned = clean_data(row['findings'])
    clinical_cleaned = clean_data(row['clinicaldata'])
    exam_cleaned = clean_data(row['ExamName'])
    impression_cleaned = clean_data(row['impression'])
    sentence = findings_cleaned + clinical_cleaned + exam_cleaned + impression_cleaned
    sentences.append(sentence)
cores = multiprocessing.cpu_count()
model = Word2Vec(sentences, min_count=1, window=3, vector_size=300, workers=cores-1, seed = 42)
model.wv.save_word2vec_format('word2vec_model.bin', binary=True)
model = KeyedVectors.load_word2vec_format('word2vec_model.bin', binary=True)
# Top 5 similar words
keyword = 'exam'
most_similar_words = model.most_similar(keyword, topn=5)
print(f"Words most similar to '{keyword}':")
for word, similarity in most_similar_words:
    print(f"{word}: {similarity}")
# Word similarity
word1 = 'contour'
word2 = 'contours'
similarity_score = model.similarity(word1, word2)
threshold = 0.8
if similarity_score > threshold:
    print(f"The words '{word1}' and '{word2}' are semantically similar with a similarity score of {similarity_score:.2f}.")
else:
    print(f"The words '{word1}' and '{word2}' are not very similar with a similarity score of {similarity_score:.2f}.")
# Unmatching words
word_list = ['lungs', 'two', 'pulmonary']
outlier_word = model.doesnt_match(word_list)
print(f"The word that does not belong to the list is: {outlier_word}")
# Analogy difference
word1 = 'lungs'
word2 = 'pulmonary'
word3 = 'exam'
analogy_result = model.most_similar(positive=[word2, word3], negative=[word1], topn=5)
print(f"Top 5 words that complete the analogy '{word1}' is to '{word2}' as '{word3}' is to ___:")
for word, similarity in analogy_result:
    print(f"{word}: Similarity = {similarity:.3f}")
# Access the word vector
word_string = 'chest'
word_vector = model[word_string]
print("Word Vector: ")
print(word_vector)
print("Word Vector Size: ", word_vector.size)
# Word frequency
word_string = 'chest'
if word_string in model.key_to_index:
    word_vector = model.get_vecattr(word_string, 'count')
    print(f"The frequency of the word '{word_string}' is: {word_vector}")
else:
    print(f"The word '{word_string}' is not present in the vocabulary.")
# Print all the words in the vocabulary
all_words = model.index_to_key
print("All words in the vocabulary:")
print(all_words)
# Length of the vocabulary
vocab_length = len(all_words)
print("Length of the vocabulary:", vocab_length)


# Calculate sentence embeddings by averaging word embeddings
def calculate_sentence_embedding(sentence):
    sentence_embedding = np.zeros(model.vector_size)
    word_count = 0
    for word in sentence:
        if word in model:
            sentence_embedding += model[word]
            word_count += 1
    if word_count > 0:
        sentence_embedding /= word_count
    return sentence_embedding


# Create list for each column
green_embeddings = []  # Findings
yellow_embeddings = []  # Clinicaldata
red_embeddings = []  # ExamName
blue_embeddings = []  # Impression
# Embeddings
for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
    findings = clean_data(row['findings'])
    clinicaldata = clean_data(row['clinicaldata'])
    examname = clean_data(row['ExamName'])
    impression = clean_data(row['impression'])
    findings_embedding = calculate_sentence_embedding(findings)
    clinicaldata_embedding = calculate_sentence_embedding(clinicaldata)
    examname_embedding = calculate_sentence_embedding(examname)
    impression_embedding = calculate_sentence_embedding(impression)
    green_embeddings.append(findings_embedding)
    yellow_embeddings.append(clinicaldata_embedding)
    red_embeddings.append(examname_embedding)
    blue_embeddings.append(impression_embedding)
green_embeddings = np.array(green_embeddings)
yellow_embeddings = np.array(yellow_embeddings)
red_embeddings = np.array(red_embeddings)
blue_embeddings = np.array(blue_embeddings)
all_embeddings = np.concatenate((green_embeddings, yellow_embeddings, red_embeddings, blue_embeddings), axis=0)
# Check for NaN or infinite values
if np.isnan(all_embeddings).any() or np.isinf(all_embeddings).any():
    problematic_indices = np.where(np.isnan(all_embeddings) | np.isinf(all_embeddings))
    all_embeddings = np.delete(all_embeddings, problematic_indices, axis=0)
# Standardize the embeddings
scaler = StandardScaler()
all_embeddings = scaler.fit_transform(all_embeddings)
# t-SNE: Dimensionality reduction to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=100, learning_rate=1000, n_iter=1000, verbose=1)
all_embeddings_2d = tsne.fit_transform(all_embeddings)
green_embeddings_2d = all_embeddings_2d[:green_embeddings.shape[0], :]
yellow_embeddings_2d = all_embeddings_2d[green_embeddings.shape[0]:green_embeddings.shape[0] + yellow_embeddings.shape[0], :]
red_embeddings_2d = all_embeddings_2d[green_embeddings.shape[0] + yellow_embeddings.shape[0]:green_embeddings.shape[0] + yellow_embeddings.shape[0] + red_embeddings.shape[0], :]
blue_embeddings_2d = all_embeddings_2d[green_embeddings.shape[0] + yellow_embeddings.shape[0] + red_embeddings.shape[0]:, :]
plt.figure()
plt.scatter(green_embeddings_2d[:, 0], green_embeddings_2d[:, 1], c='green', label='Findings')
plt.scatter(yellow_embeddings_2d[:, 0], yellow_embeddings_2d[:, 1], c='yellow', label='Clinical Data')
plt.scatter(red_embeddings_2d[:, 0], red_embeddings_2d[:, 1], c='red', label='Exam Name')
plt.scatter(blue_embeddings_2d[:, 0], blue_embeddings_2d[:, 1], c='blue', label='Impression')
plt.legend()
plt.title('2D Word2Vec Scatter Plot')
plt.show()
# Split the data into training and testing sets
num_samples = green_embeddings.shape[0]
labels = np.array([0] * num_samples + [1] * num_samples + [2] * num_samples + [3] * num_samples)
X_train, X_test, y_train, y_test = train_test_split(all_embeddings, labels, test_size=0.2, random_state=42)
# Train the logistic regression model
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train, y_train)
y_pred = log_reg_model.predict(X_test)
# Accuracy
word2vec_accuracy = accuracy_score(y_test, y_pred)
print("\nWord2Vec Accuracy:", word2vec_accuracy)
# Classification report
report = classification_report(y_test, y_pred, target_names=['Findings', 'Clinical Data', 'Exam Name', 'Impression'])
print("\nClassification Report:")
print(report)