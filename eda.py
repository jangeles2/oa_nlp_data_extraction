# Code adapted from "Exploratory Data Analysis for Natural Language Processing: A Complete Guide to Python Tools"

# Sources
# Website: https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools
# Repository: https://app.neptune.ai/o/neptune-ai/org/eda-nlp-tools/notebooks?notebookId=2-0-top-ngrams-barchart-671a187d-c3b4-475a-bc9e-8aa6c937923b retrieved in June 2023

# Exploratory Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
import nltk

# Load data
data = pd.read_csv('open_ave_data.csv')
# Define columns
columns = ['ReportText', 'findings', 'clinicaldata', 'ExamName', 'impression']
# Define colors
colors = {'ReportText': 'purple', 'findings': 'green',
          'clinicaldata': 'yellow', 'ExamName': 'red', 'impression': 'blue'}


# Character Length Histogram
def plot_character_length_histogram(text, color):
    text.str.len(). \
        hist(alpha=0.7, color=color)


fig, ax = plt.subplots()
for column in columns:
    plot_character_length_histogram(data[column], colors[column])
plt.xlabel('Character Length')
plt.ylabel('Frequency')
plt.title('Character Length Histogram')
plt.legend(columns)
plt.show()


# Word Number Histogram
def plot_word_number_histogram(text, color):
    text.dropna().str.split().\
        map(lambda x: len(x)).\
        hist(alpha=0.7, color=color)


fig, ax = plt.subplots()
for column in columns:
    plot_word_number_histogram(data[column], colors[column])
plt.xlabel('Word Number')
plt.ylabel('Frequency')
plt.title('Word Number Histogram')
plt.legend(columns)
plt.show()


# Word Length Histogram
def plot_word_length_histogram(text, color):
    text.dropna().str.split().\
        apply(lambda x: [len(i) for i in x]). \
        map(lambda x: np.mean(x)).\
        hist(alpha=0.7, color=color)


fig, ax = plt.subplots()
for column in columns:
    plot_word_length_histogram(data[column], colors[column])
plt.xlabel('Word Length')
plt.ylabel('Frequency')
plt.title('Word Length Histogram')
plt.legend(columns)
plt.show()


# Top Stopwords Barchart
def plot_top_stopwords_barchart(text, color):
    stop = set(stopwords.words('english'))
    new = text.dropna().str.split()
    new = new.values.tolist()
    corpus = [word for i in new for word in i]
    from collections import defaultdict
    dic = defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word] += 1
    top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:10]
    x, y = zip(*top)
    plt.bar(x, y, color=color)


for column in columns:
    plt.figure()
    plot_top_stopwords_barchart(data[column], colors[column])
    plt.xlabel('Stopwords')
    plt.ylabel('Frequency')
    plt.title(f'Top Stopwords Barchart for {column}')
    plt.show()


# Top Non-Stopwords Barchart
def plot_top_non_stopwords_barchart(text, color):
    stop = set(stopwords.words('english'))
    new = text.dropna().str.split()
    new = new.values.tolist()
    corpus = [word for i in new for word in i]
    counter = Counter(corpus)
    most = counter.most_common()
    x, y = [], []
    for word, count in most[:40]:
        if word not in stop:
            x.append(word)
            y.append(count)
    sns.barplot(x=y, y=x, color=color)


for column in columns:
    plt.figure(figsize=(14, 6))
    plt.tight_layout()
    plot_top_non_stopwords_barchart(data[column], colors[column])
    plt.xlabel('Frequency')
    plt.ylabel('Non-Stopwords')
    plt.title(f'Top Non-Stopwords Barchart for {column}')
    plt.show()


# Top N-grams Barchart
def plot_top_ngrams_barchart(text, color, n=2):
    set(stopwords.words('english'))
    text = text.fillna('')
    new = text.str.split()
    new = new.values.tolist()
    [word for i in new for word in i]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:10]

    top_n_bigrams = _get_top_ngram(text, n)[:10]
    x, y = map(list, zip(*top_n_bigrams))
    sns.barplot(x=y, y=x, color=color)


# Bigram
for column in columns:
    plt.figure(figsize=(16, 6))
    plt.tight_layout()
    plot_top_ngrams_barchart(data[column], colors[column], 2)
    plt.xlabel('Frequency')
    plt.ylabel('Bigram')
    plt.title(f'Bigram for {column}')
    plt.show()
# Trigram
for column in columns:
    plt.figure(figsize=(18, 6))
    plt.tight_layout()
    plot_top_ngrams_barchart(data[column], colors[column], 3)
    plt.xlabel('Frequency')
    plt.ylabel('Trigram')
    plt.title(f'Trigram for {column}')
    plt.show()


# Wordcloud
def plot_wordcloud(text, title=''):
    nltk.download('stopwords')
    stop = set(stopwords.words('english'))

    def _preprocess_text(text):
        corpus = []
        stem = PorterStemmer()
        lem = WordNetLemmatizer()
        for news in text:
            if isinstance(news, str):
                words = [w for w in word_tokenize(news) if (w not in stop)]
                words = [lem.lemmatize(w) for w in words if len(w) > 2]
                corpus.append(words)
        return corpus
    corpus = _preprocess_text(text)
    wordcloud = WordCloud(
        background_color='white',
        stopwords=set(STOPWORDS),
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)

    wordcloud = wordcloud.generate(str(corpus))
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.title(f'Wordcloud for {title} ')
    plt.show()


for column in columns:
    plot_wordcloud(data[column], title=column)
    plt.show()

# Temporal Distribution Line Plot
data['ExamDate'] = data['ExamName'].str.extract(r'EXAM DATE: (\d{2}/\d{2}/\d{4})')
data['ExamDate'] = pd.to_datetime(data['ExamDate'])
data.set_index('ExamDate', inplace=True)
exam_counts_yearly = data.resample('Y').size()
plt.figure(figsize=(10, 6))
plt.plot(exam_counts_yearly.index.year, exam_counts_yearly.values, marker='o')
plt.xlabel('Year')
plt.ylabel('Exam Count')
plt.title('Temporal Distribution of Exams (Yearly)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
