import string
import numpy as np
import pandas as pd
import spacy
import os
import fitz
import wordcloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')


def pdf_to_text(path: str):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


class TextPreprocessor:
    def __init__(self, lemmatize: bool = True, remove_punct: bool = True, remove_digits: bool = True,
                 remove_stop_words: bool = True,
                 remove_short_words: bool = True, minlen: int = 1, maxlen: int = 1, top_p: float = None,
                 bottom_p: float = None):
        self.lemmatize = lemmatize
        self.remove_punct = remove_punct
        self.remove_digits = remove_digits
        self.remove_stop_words = remove_stop_words
        self.remove_short_words = remove_short_words
        self.minlen = minlen
        self.maxlen = maxlen
        self.top_p = top_p
        self.bottom_p = bottom_p
        self.words_to_remove = []
        self.stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                           "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                           'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
                           'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
                           'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                           'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                           'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                           'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                           'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                           'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                           'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                           'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                           'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
                           'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                           "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
                           "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
                           'won', "won't", 'wouldn', "wouldn't"]

    @staticmethod
    def __remove_double_whitespaces(string: str):
        return " ".join(string.split())

    @staticmethod
    def __lemmatize(string_series: pd.Series):
        nlp = spacy.load(os.path.join(".", "en_core_web_sm-3.4.1"))

        def str_lemmatize(string: str):
            doc = nlp(string)
            return " ".join([token.lemma_ for token in doc])

        return string_series.map(str_lemmatize)

    def __remove_punct(self, string_series: pd.Series):
        clean_string_series = string_series.str.replace(pat=f'[{string.punctuation}]', repl=" ", regex=True).copy()
        return clean_string_series.map(self.__remove_double_whitespaces)

    def __remove_digits(self, string_series: pd.Series):
        clean_string_series = string_series.str.replace(pat=r'\d', repl=" ", regex=True).copy()
        return clean_string_series.map(self.__remove_double_whitespaces)

    @staticmethod
    def __remove_short_words(string_series: pd.Series, minlen: int = 1, maxlen: int = 1):
        clean_string_series = string_series.map(lambda string: " ".join([word for word in string.split() if
                                                                         (len(word) > maxlen) or (len(word) < minlen)]))
        return clean_string_series

    def __remove_stop_words(self, string_series: pd.Series):
        def str_remove_stop_words(string: str):
            stops = self.stop_words
            return " ".join([token for token in string.split() if token not in stops])

        return string_series.map(str_remove_stop_words)

    def __remove_top_bottom_words(self, string_series: pd.Series, top_p: int = None,
                                  bottom_p: int = None, dataset: str = 'train'):
        if dataset == 'train':
            if top_p is None:
                top_p = 0
            if bottom_p is None:
                bottom_p = 0

            if top_p > 0 or bottom_p > 0:
                word_freq = pd.Series(" ".join(string_series).split()).value_counts()
                n_words = len(word_freq)

            if top_p > 0:
                self.words_to_remove.extend([*word_freq.index[: int(np.ceil(top_p * n_words))]])

            if bottom_p > 0:
                self.words_to_remove.extend([*word_freq.index[-int(np.ceil(bottom_p * n_words)):]])

        if len(self.words_to_remove) == 0:
            return string_series
        else:
            clean_string_series = string_series.map(lambda string: " ".join([word for word in string.split()
                                                                             if word not in self.words_to_remove]))
            return clean_string_series

    def preprocess(self, string_series: pd.Series, dataset: str = "train"):
        string_series = string_series.str.lower().copy()
        if self.lemmatize:
            string_series = self.__lemmatize(string_series=string_series)
        if self.remove_punct:
            string_series = self.__remove_punct(string_series=string_series)
        if self.remove_digits:
            string_series = self.__remove_digits(string_series=string_series)
        if self.remove_stop_words:
            string_series = self.__remove_stop_words(string_series=string_series)
        if self.remove_short_words:
            string_series = self.__remove_short_words(string_series=string_series,
                                                      minlen=self.minlen,
                                                      maxlen=self.maxlen)
        string_series = self.__remove_top_bottom_words(string_series=string_series,
                                                       top_p=self.top_p,
                                                       bottom_p=self.bottom_p, dataset=dataset)

        string_series = string_series.str.strip().copy()
        string_series.replace(to_replace="", value="this is an empty message", inplace=True)

        return string_series


def plot_wordcloud(pdf_path: str):
    text = pdf_to_text(pdf_path)
    text_preprocessor = TextPreprocessor()
    preprocessed_text = text_preprocessor.preprocess(pd.Series(text))[0]
    wc = wordcloud.WordCloud(width=500, height=500,
                             background_color='white', min_font_size=10).generate(preprocessed_text)
    plt.axis("off")
    plt.imshow(wc)
    plt.savefig(os.path.join('static', 'wordcloud.png'),
                dpi=300, bbox_inches='tight')
