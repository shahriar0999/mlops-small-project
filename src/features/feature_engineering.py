import numpy as np
import pandas as pd
import os
import yaml
import logging
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Set up logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

try:
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('feature_engineering_errors.log')
    file_handler.setLevel(logging.ERROR)

    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Error setting up logging: {str(e)}")

def load_data():
    try:
        # Load the processed data
        logger.info("Loading processed data...")
        train_data = pd.read_csv('./data/interim/train_processed.csv')
        test_data = pd.read_csv('./data/interim/test_processed.csv')
        
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        
        logger.debug("Data loaded and null values handled")
        return train_data, test_data
    except FileNotFoundError as e:
        logger.error(f"Input data files not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def apply_bow_vectorization(X_train, X_test):
    try:
        with open("params.yaml", 'r') as file:
            params = yaml.safe_load(file)
        max_features = params['feature_engineering']['max_features']
        logger.debug('max_features retrieved')
        logger.info("Applying Bag of Words vectorization...")
        vectorizer = CountVectorizer(max_features=max_features)
        
        # Fit and transform training data
        X_train_bow = vectorizer.fit_transform(X_train)
        logger.debug("Training data vectorized")
        
        # Transform test data
        X_test_bow = vectorizer.transform(X_test)
        logger.debug("Test data vectorized")

        # save vectorizer for future use 
        pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
        
        return X_train_bow, X_test_bow
    except Exception as e:
        logger.error(f"Error in BOW vectorization: {str(e)}")
        raise

def create_feature_dataframes(X_train_bow, y_train, X_test_bow, y_test):
    try:
        logger.info("Creating feature dataframes...")
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test
        
        logger.debug("Feature dataframes created successfully")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error creating feature dataframes: {str(e)}")
        raise

def save_features(train_df, test_df):
    try:
        logger.info("Saving feature engineered data...")
        data_path = os.path.join("data", "features")
        os.makedirs(data_path, exist_ok=True)
        
        train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)
        
        logger.info("Features saved successfully")
    except Exception as e:
        logger.error(f"Error saving features: {str(e)}")
        raise

def main():
    try:
        # Load data
        train_data, test_data = load_data()
        
        # Prepare features and labels
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        
        # Apply BOW vectorization
        X_train_bow, X_test_bow = apply_bow_vectorization(X_train, X_test)
        
        # Create feature dataframes
        train_df, test_df = create_feature_dataframes(X_train_bow, y_train, X_test_bow, y_test)
        
        # Save features
        save_features(train_df, test_df)
        
        logger.info("Feature engineering pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in feature engineering pipeline: {str(e)}")
        raise
    finally:
        logger.info("Feature engineering process finished")

if __name__ == '__main__':
    main()