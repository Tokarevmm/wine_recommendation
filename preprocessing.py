
import re
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_lg')
import nltk
from typing import List

def normalize_text(text):
    text = text.lower()
    tm1 = re.sub('<pre>.*?</pre>', '', text, flags=re.DOTALL)
    tm2 = re.sub('<code>.*?</code>', '', tm1, flags=re.DOTALL)
    tm3 = re.sub('<[^>]+>©', '', tm1, flags=re.DOTALL)
    return tm3.replace("\n", "")


def cleanup_text(docs, nlp: spacy.Language, stopwords: List[str]):

    punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~©'

    texts = []
    doc = nlp(docs, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
    tokens = ' '.join(tokens)
    texts.append(tokens)
    return pd.Series(texts)

def preprocess_text(
        text: str,
        nlp: spacy.Language = spacy.load('en_core_web_lg'),
        stopwords: List[str] = nltk.corpus.stopwords.words('english')) -> str:
    """
    preprocess_text returns normalized and cleaned text.

    :param text: Text to process.
    :param nlp: spacy.Language object for language of text to be processed.
    :param stopwords: List of stopwords for selected language to be removed from processed text.
    :returns: Cleaned string.

    Example usage:
        preprocess_text(
            text = "Some text",
            nlp = spacy.load('en_core_web_lg'),
            stopwords = nltk.corpus.stopwords.words('english'))
    """
    text = normalize_text(text)
    text = cleanup_text(text, nlp, stopwords)
    return text
