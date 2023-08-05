# Functions to import to Metaflow

# Define columns to analyze
def define_columns_to_analyze():
    columns_to_analyze = ['ReportText', 'findings', 'clinicaldata', 'ExamName', 'impression']
    return columns_to_analyze
    
# Fill missing values
def preprocess_data(data, column):
    data[column].fillna('', inplace=True)

# Clean data
def clean_data(text):
    import re
    from nltk.corpus import stopwords
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        stopwords_list = set(stopwords.words('english'))
        clean_words = [word for word in words if word not in stopwords_list and len(word) > 2]
        return ' '.join(clean_words)
    else:
        return ''

# Clean and preprocess all columns
def clean_and_preprocess_data(data):
    columns_to_analyze = define_columns_to_analyze()
    for column in columns_to_analyze:
        preprocess_data(data, column)
        data[column] = data[column].apply(clean_data)

# Generate n-grams
def generate_ngram_chart(data, n, column, chart_suffix):
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    import matplotlib.pyplot as plt
    import seaborn as sns
    vectorizer = CountVectorizer(ngram_range=(n, n))
    ngram_counts = vectorizer.fit_transform(data[column])
    ngram_count_df = pd.DataFrame(ngram_counts.toarray(), columns=vectorizer.get_feature_names_out())
    plt.figure(figsize=(10, 6))
    top_ngrams = ngram_count_df.sum().nlargest(10).sort_values(ascending=False)
    sns.barplot(x=top_ngrams.index, y=top_ngrams.values)
    if n == 2:
        n = 'Bi'
    else:
        n = 'Tri'
    plt.title(f'Top {n}grams in {column}')
    plt.xlabel(f'{n}gram')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save image file path to download
    ngram_image_path = f'/home/jessicaangelescs/projectmetaflow/{column}_{chart_suffix}_chart.png'
    plt.savefig(ngram_image_path)
    print(f'{chart_suffix.capitalize()} chart for {column} saved at: {ngram_image_path}')
    return ngram_image_path

# Generate word clouds
def generate_word_cloud(data, column):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud, STOPWORDS
    text_for_wordcloud = " ".join(data[column])
    wordcloud = WordCloud(width=1800, height=1400, background_color='white', stopwords=STOPWORDS).generate(
        text_for_wordcloud)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {column}')
    wordcloud_image_path = f'/home/jessicaangelescs/projectmetaflow/{column}_wordcloud.png'
    plt.savefig(wordcloud_image_path)
    print(f'Word cloud for {column} saved at: {wordcloud_image_path}')
    return wordcloud_image_path

# Define corpus
def corpus(data):
    findings = data["findings"].values.tolist()
    clinical = data["clinicaldata"].values.tolist()
    exam = data["ExamName"].values.tolist()
    impression = data["impression"].values.tolist()
    return findings, clinical, exam, impression

# Calculate TF-IDF
def calc_tfidf_docs(data):
    findings, clinical, exam, impression = corpus(data)
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus_docs = findings + clinical + exam + impression
    vectorizer = TfidfVectorizer()
    tfidf_documents = vectorizer.fit_transform(corpus_docs)
    return tfidf_documents

# Create 2D TF-IDF scatter plot
def tfidf_plot(data):
    findings, clinical, exam, impression = corpus(data)
    tfidf_documents = calc_tfidf_docs(data)
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    # Color code each column
    section_colors = {
        'findings': 'green',
        'clinical': 'yellow',
        'exam': 'red',
        'impression': 'blue'
    }
    combined_tfidf_documents = tfidf_documents
    colors = np.concatenate([
        np.full(len(findings), section_colors['findings']),
        np.full(len(clinical), section_colors['clinical']),
        np.full(len(exam), section_colors['exam']),
        np.full(len(impression), section_colors['impression'])
    ])
    # t-SNE: Dimensionality reduction to 2D
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=100)
    combined_tfidf_documents_embedded = tsne.fit_transform(combined_tfidf_documents.toarray())
    scatter = plt.scatter(combined_tfidf_documents_embedded[:, 0],
                            combined_tfidf_documents_embedded[:, 1], c=colors)
    legend_patches = [
        mpatches.Patch(color='green', label='Findings'),
        mpatches.Patch(color='yellow', label='Clinical Data'),
        mpatches.Patch(color='red', label='Exam Name'),
        mpatches.Patch(color='blue', label='Impression')
    ]
    plt.title('2D TF-IDF Scatter Plot')
    plt.legend(handles=legend_patches)
    tfidf_path = '/home/jessicaangelescs/projectmetaflow/tfidf_scatter_plot.png'
    plt.savefig(tfidf_path)
    return tfidf_path
    
# Tokenize sentences
def toke_sent(data):
    from tqdm import tqdm
    import multiprocessing
    from gensim.models import Word2Vec
    from nltk.tokenize import word_tokenize
    clean_and_preprocess_data(data) 
    sentences = []
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]): 
        findings_cleaned = row['findings']
        clinical_cleaned = row['clinicaldata']
        exam_cleaned = row['ExamName']
        impression_cleaned = row['impression']
        sentence = findings_cleaned + clinical_cleaned + exam_cleaned + impression_cleaned
        tokenized_sentence = word_tokenize(sentence)
        sentences.append(tokenized_sentence)
    cores = multiprocessing.cpu_count()
    model = Word2Vec(sentences, min_count=1, window=3, vector_size=300, workers=cores - 1)
    return model

# Calculate sentence embeddings
def calculate_sentence_embeddings(data):
    import numpy as np
    from nltk.tokenize import word_tokenize
    from tqdm import tqdm
    model = toke_sent(data)
    def calculate_sentence_embedding(sentence):
        sentence_embedding = np.zeros(model.vector_size)
        word_count = 0
        for word in sentence:
            if word in model.wv:
                sentence_embedding += model.wv[word]
                word_count += 1
        if word_count > 0:
            sentence_embedding /= word_count
        return sentence_embedding
    # Color code each column
    green_embeddings = []  # Findings
    yellow_embeddings = []  # Clinicaldata
    red_embeddings = []  # ExamName
    blue_embeddings = []  # Impression
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        findings = word_tokenize(row['findings'])
        clinicaldata = word_tokenize(row['clinicaldata'])
        examname = word_tokenize(row['ExamName'])
        impression = word_tokenize(row['impression'])
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
    return green_embeddings, yellow_embeddings, red_embeddings, blue_embeddings

# Concatenate embeddings for each column
def concatenate_embeddings(data):
    import numpy as np
    green_embeddings, yellow_embeddings, red_embeddings, blue_embeddings = calculate_sentence_embeddings(data)
    all_embeddings = np.concatenate(
        (green_embeddings, yellow_embeddings, red_embeddings, blue_embeddings), axis=0)
    if np.isnan(all_embeddings).any() or np.isinf(all_embeddings).any():
        problematic_indices = np.where(np.isnan(all_embeddings) | np.isinf(all_embeddings))
        all_embeddings = np.delete(all_embeddings, problematic_indices, axis=0)
    return all_embeddings

# Standarize embeddings
def standardize_embeddings(data):
    all_embeddings = concatenate_embeddings(data)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    all_embeddings = scaler.fit_transform(all_embeddings)
    return all_embeddings

# t-SNE: Dimensionality reduction to 2D
def apply_tsne(data):
    from sklearn.manifold import TSNE
    all_embeddings = standardize_embeddings(data)
    tsne = TSNE(n_components=2, random_state=50, perplexity=100, learning_rate=1000, n_iter=1000, verbose=1)
    all_embeddings_2d = tsne.fit_transform(all_embeddings)
    return all_embeddings_2d

# Create 2D Word2Vec scatter plot
def word2vec_plot(data):
    all_embeddings_2d = apply_tsne(data)
    green_embeddings, yellow_embeddings, red_embeddings, blue_embeddings = calculate_sentence_embeddings(data)
    import matplotlib.pyplot as plt
    green_embeddings_2d = all_embeddings_2d[:green_embeddings.shape[0], :]
    yellow_embeddings_2d = all_embeddings_2d[green_embeddings.shape[0]:green_embeddings.shape[0] + yellow_embeddings.shape[0], :]
    red_embeddings_2d = all_embeddings_2d[green_embeddings.shape[0] + yellow_embeddings.shape[0]: green_embeddings.shape[0] + yellow_embeddings.shape[0] + red_embeddings.shape[0], :]
    blue_embeddings_2d = all_embeddings_2d[green_embeddings.shape[0] + yellow_embeddings.shape[0] + red_embeddings.shape[0]:, :]
    plt.figure()
    plt.scatter(green_embeddings_2d[:, 0], green_embeddings_2d[:, 1], c='green', label='Findings')
    plt.scatter(yellow_embeddings_2d[:, 0], yellow_embeddings_2d[:, 1], c='yellow', label='Clinical Data')
    plt.scatter(red_embeddings_2d[:, 0], red_embeddings_2d[:, 1], c='red', label='Exam Name')
    plt.scatter(blue_embeddings_2d[:, 0], blue_embeddings_2d[:, 1], c='blue', label='Impression')
    plt.legend()
    plt.title('2D Word2Vec Scatter Plot')
    word2vec_path = '/home/jessicaangelescs/projectmetaflow/word2vec_scatterplot.png'
    plt.savefig(word2vec_path)
    return word2vec_path

# TF-IDF Logistic Regression
def tfidf_logistic_regression(data):
    findings, clinical, exam, impression = corpus(data)
    tfidf_documents = calc_tfidf_docs(data)
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    labels = np.concatenate((np.zeros(len(exam)), np.ones(len(clinical)), np.full(len(findings), 2),
                                np.full(len(impression), 3)))
    X_train, X_test, y_train, y_test = train_test_split(tfidf_documents, labels, test_size=0.2,
                                                        random_state=42)
    lr = LogisticRegression(C=1, solver='saga')
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    # Accuracy
    tfidf_accuracy = accuracy_score(y_test, lr_preds)
    # Classification report
    tfidf_report = classification_report(y_test, lr_preds,
                                        target_names=['Findings', 'Clinical Data', 'Exam Name', 'Impression'])
    return tfidf_accuracy, tfidf_report

# Word2Vec Logistic Regression
def word2vec_logistic_regression(data):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    all_embeddings = standardize_embeddings(data)
    num_samples = 954
    labels = np.array([0] * num_samples + [1] * num_samples + [2] * num_samples + [3] * num_samples)
    X_train, X_test, y_train, y_test = train_test_split(all_embeddings, labels, test_size=0.2, random_state=42)
    log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
    log_reg_model.fit(X_train, y_train)
    y_pred = log_reg_model.predict(X_test)
    # Accuracy
    word2vec_accuracy = accuracy_score(y_test, y_pred)
    # Classification Report
    word2vec_report = classification_report(y_test, y_pred,
                                    target_names=['Findings', 'Clinical Data', 'Exam Name', 'Impression'])
    return word2vec_accuracy, word2vec_report

# TF-IDF Model Runtime
def run_tfidf_logistic_regression(data):
    import time
    start_time = time.time()
    accuracy, report = tfidf_logistic_regression(data)
    end_time = time.time()
    runtime = end_time - start_time
    return accuracy, report, runtime

# Word2Vec Model Runtime
def run_word2vec_logistic_regression(data):
    import time
    start_time = time.time()
    accuracy, report = word2vec_logistic_regression(data)
    end_time = time.time()
    runtime = end_time - start_time
    return accuracy, report, runtime

# Print classification report, accuracy, runtime, and best model
def display_results(tf_idf_classrep, word2vec_classrep, word2vec_plot, tf_idf_plot,
                    tf_idf_acc, word2vec_acc, tf_idf_runtime, word2vec_runtime):
    result_output = (
        f"\nTF-IDF Classification Report: \n{tf_idf_classrep}\n"
        f"\n2D Scatter plot for TF-IDF saved at: {tf_idf_plot}\n"
        f"\nTF-IDF Accuracy: {tf_idf_acc}\n"
        f"\nTF-IDF Logistic Regression Runtime: {tf_idf_runtime} seconds\n"
        f"\nWord2Vec Classification Report: \n{word2vec_classrep}\n"
        f"\n2D Scatter plot for Word2Vec saved at: {word2vec_plot}\n"
        f"\nWord2Vec Accuracy: {word2vec_acc}\n"
        f"\nWord2Vec Logistic Regression Runtime: {word2vec_runtime} seconds\n"
    )
    best_model = "TF-IDF" if (tf_idf_runtime < word2vec_runtime) and (float(tf_idf_acc) > float(word2vec_acc)) else "Word2Vec"
    result_output += f"\nBest Model: {best_model}\n"
    return result_output
