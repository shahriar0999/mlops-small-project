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
    
def normalize_text(df):

    try:
        logger.info("Starting text normalization")
        df.content=df.content.apply(lambda content : lower_case(content))
        logger.debug("Converted text to lower case")
        df.content=df.content.apply(lambda content : remove_stop_words(content))
        logger.debug("Removed stop words")
        df.content=df.content.apply(lambda content : removing_numbers(content))
        logger.debug("Removed numbers")
        df.content=df.content.apply(lambda content : removing_punctuations(content))
        logger.debug("Removed punctuations")
        df.content=df.content.apply(lambda content : removing_urls(content))
        logger.debug("Removed urls")
        df.content=df.content.apply(lambda content : lemmatization(content))
        logger.debug("Applied lemmatization")
        df.content=df.content.apply(lambda content : remove_small_sentences(content))
        logger.debug("Removed small sentences")
        return df
    except Exception as e:
        logger.error(f"Error in text normalization: {str(e)}")
        return df

def main():
    try:
        # load the data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug("Successfully loaded train and test data")
    except FileNotFoundError as e:
        logger.error("File not found")
        raise

    # normalize the text
    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)

    # store the data inside data/interim
    data_path = os.path.join("data","interim")

    os.makedirs(data_path, exist_ok=True)

    train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"))
    test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"))

if __name__ == '__main__':
    main()