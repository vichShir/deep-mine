import re
import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
nltk.download("punkt")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


class Vocabulary:

    def __init__(self):
        self.stemmer = PorterStemmer()  # stemming
        self.analyzer = CountVectorizer().build_analyzer()
        self.vectorizer = CountVectorizer(analyzer=self.stemmed_words)

    # to clean data
    def replace_special_chars(self, text):
        try:
            return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        except:
            return None

    def normalise_text(self, text):
        text = text.str.lower()
        text = text.str.replace(r"\#","")
        text = text.str.replace(r"http\S+","URL") # \s for whitespace character and \S for non-whitespace character
        text = text.str.replace(r"@"," ")
        text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
        text = text.str.replace(r"\s{2,}", " ")
        text = text.apply(self.replace_special_chars)
        return text

    def preprocess(self, raw_corpus):
        corpus = self.normalise_text(raw_corpus).values.tolist()
        corpus = [x for x in corpus if not isinstance(x, float)]  # remove float nan

        # remove stop-words
        result = []
        for i in corpus:
            #out = nltk.word_tokenize(i)
            out = [x for x in i.split() if x not in stop_words]
            result.append(" ". join(out))
        return result

    def stemmed_words(self, doc):
        return (self.stemmer.stem(w) for w in self.analyzer(doc))

    def fit(self, raw_corpus):
        # preprocessing
        corpus = self.preprocess(raw_corpus)

        print('Training corpus...')
        self.vectorizer.fit(corpus)

    def transform(self, text):
        if isinstance(text, str):
            text = [text]
        x = self.vectorizer.transform(text)
        return torch.tensor(x.toarray())

    def get_vocab(self):
        vocab = self.vectorizer.get_feature_names_out()
        return vocab

    def vocab_size(self):
        return len(self.get_vocab())