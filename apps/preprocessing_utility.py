import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import logging
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer


logger = logging.getLogger('data_Preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# transform the data
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text):
    try:
        lemmatizer= WordNetLemmatizer()
        text = text.split()
        text=[lemmatizer.lemmatize(y) for y in text]
        return " " .join(text)
    except Exception as e:
        logger.error(f"Error in lemmatization: {e}")
        raise

def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))
        Text=[i for i in str(text).split() if i not in stop_words]
        return " ".join(Text)
    except Exception as e:
        logger.error(f"Error in removing stop words: {e}")
        raise

def removing_numbers(text):
    try:
        text=''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        logger.error(f"Error in removing numbers: {e}")
        raise

def lower_case(text):
    try:
        text = text.split()
        text=[y.lower() for y in text]
        return " " .join(text)
    except Exception as e:
        logger.error(f"Error in converting to lower case: {e}")
        raise

def removing_punctuations(text):
    try:
        ## Remove punctuations
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛',"", )

        ## remove extra whitespace
        text = re.sub('\s+', ' ', text)
        text =  " ".join(text.split())
        return text.strip()
    except Exception as e:
        logger.error(f"Error in removing punctuations: {e}")
        raise

def removing_urls(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error(f"Error in removing urls: {e}")
        raise


def remove_small_sentences(text):
    """Remove sentences with less than 3 words"""
    try:
        if len(str(text).split()) < 3:
            return np.nan
        return text
    except Exception as e:
        logger.error(f"Error in removing small sentences: {e}")
        return text
    
def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    text = remove_small_sentences(text)
    return text
