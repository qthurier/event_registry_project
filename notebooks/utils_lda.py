import re, unicodedata, contractions
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Following functions perform string level transformations

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def remove_punct_and_digit(word):
    """Remove punctuation and digits in string of text"""
    return re.sub(r'[^\w\s]|_|\d', '', word)

def denoise(text):
    """Apply string level transformations"""
    text = replace_contractions(text)
    text = remove_punct_and_digit(text)
    return text

# Following functions perform token level transformations

def non_ascii(word):
    """Remove non-ASCII characters from a token"""
    return unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def uppercase(word):
    """Convert all characters to lowercase from a token"""
    return word.lower()

def short(word):
    """Filter a short token"""
    if len(word) > 2:
        return word

stop_words = set(stopwords.words('english')) 

def stopword(word):
    """Filter a stop word token"""
    if word not in stop_words:
        return word

import functools

def compose(*functions):
    def compose2(f, g):
        return lambda x: f(g(x)) 
    return functools.reduce(compose2, functions, lambda x: x)
 
normalize_pipeline = compose(stopword, short, uppercase, non_ascii)

def normalize(words):
    """Apply token level transformations"""
    new_words = []
    for word in words:
        new_word = normalize_pipeline(word)
        if new_word != '' and new_word is not None:
            new_words.append(new_word)
    return new_words

lemmatizer = WordNetLemmatizer()

def lemmatize(words):
    """Lemmatize verbs in list of tokenized words"""
    new_words = []
    for word in words:
        new_word = lemmatizer.lemmatize(word)
        new_words.append(new_word)
    return new_words

# Compose all transformations above

process_text = compose(lemmatize, normalize, lambda x: word_tokenize(x), denoise) 
