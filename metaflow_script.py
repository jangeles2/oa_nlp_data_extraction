# Code adapted from various sources
    
# Sources for Exploratory Data Analysis: 
# Website: https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools retrieved in June 2023
# Repository: https://app.neptune.ai/o/neptune-ai/org/eda-nlp-tools/notebooks?notebookId=2-0-top-ngrams-barchart-671a187d-c3b4-475a-bc9e-8aa6c937923b retrieved in June 2023

# Sources for TF-IDF Model: 
# Scikit Learn TfidfVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html retrieved in June 2023
# Scikit Learn TSNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html retrieved in June 2023
# Bashkeel's BoW-TFIDF Repository https://github.com/Bashkeel/BoW-TFIDF/blob/master/TF-IDF%20and%20N-Grams.ipynb retrieved in June 2023

# Source for Word2Vec Model: 
# Website: https://swatimeena989.medium.com/training-word2vec-using-gensim-14433890e8e4 retrieved in July 2023

# Source for Metaflow:
# Website: https://outerbounds.com/docs/intro-tutorial-overview/ retrieved in July 2023


# Metaflow for medical data extraction from unstructured radiology reports
from metaflow import FlowSpec, step


class DataAnalysisFlow(FlowSpec):
    # Read the data
    @step
    def start(self):
        import pandas as pd
        self.data = pd.read_csv('open_ave_data.csv')
        self.next(self.eda)

    # Exploratory Data Analysis: Bigram, Trigram, and Word Cloud
    @step
    def eda(self):
        from functions import define_columns_to_analyze, clean_and_preprocess_data, generate_ngram_chart, generate_word_cloud
        columns_to_analyze = define_columns_to_analyze()
        clean_and_preprocess_data(self.data)
        for column in columns_to_analyze:
            generate_ngram_chart(self.data, 2, column, 'bigram')  # Bigram 
            generate_ngram_chart(self.data, 3, column, 'trigram')  # Trigram 
            generate_word_cloud(self.data, column) # Word Cloud
        self.next(self.func, self.train_model)

    # Model functionality
    @step
    def func(self):
        from functions import tfidf_plot, word2vec_plot
        tfidf_path = tfidf_plot(self.data) # TF-IDF 
        self.tgraph = tfidf_path 
        word2vec_path = word2vec_plot(self.data) # Word2Vec 
        self.wgraph = word2vec_path
        self.next(self.join)

    # Model training
    @step
    def train_model(self):
        import time
        from functions import run_tfidf_logistic_regression, run_word2vec_logistic_regression
        self.tacc, self.tcr, self.tf_idf_runtime = run_tfidf_logistic_regression(self.data) # TF-IDF 
        self.wacc, self.wcr, self.word2vec_runtime = run_word2vec_logistic_regression(self.data) # Word2Vec
        self.next(self.join)

    # Join TF-IDF and Word2Vec models
    @step
    def join(self, inputs):
        from functions import display_results
        tf_idf_classrep = inputs.train_model.tcr
        word2vec_classrep = inputs.train_model.wcr
        word2vec_plot = inputs.func.wgraph
        tf_idf_plot = inputs.func.tgraph
        tf_idf_acc = inputs.train_model.tacc
        word2vec_acc = inputs.train_model.wacc
        tf_idf_runtime = inputs.train_model.tf_idf_runtime
        word2vec_runtime = inputs.train_model.word2vec_runtime
        result_output = display_results(tf_idf_classrep, word2vec_classrep, word2vec_plot, tf_idf_plot,
                                        tf_idf_acc, word2vec_acc, tf_idf_runtime, word2vec_runtime)
        print(result_output)
        self.next(self.end)

    # End of flow
    @step
    def end(self):
        print("Flow is complete")


if __name__ == '__main__':
    DataAnalysisFlow()
