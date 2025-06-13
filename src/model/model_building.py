import numpy as np
import pandas as pd
import pickle
import yaml
import logging
from sklearn.linear_model import LogisticRegression

# Logging configuration
try:
    logger = logging.getLogger('model_training')
    logger.setLevel('DEBUG')

    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')

    file_handler = logging.FileHandler('model_training.log')
    file_handler.setLevel('ERROR')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Error configuring logging: {str(e)}")
    raise

def load_params():
    try:
        logger.info("Loading model parameters")
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
        model_params = params['model_building']
        logger.debug(f"Loaded parameters: {model_params}")
        return model_params
    except FileNotFoundError:
        logger.error("params.yaml file not found")
        raise
    except Exception as e:
        logger.error(f"Error loading parameters: {str(e)}")
        raise

def load_training_data(file_path : str) -> pd:
    try:
        logger.info("Loading training data")
        train_data = pd.read_csv(file_path)
        
        X_train = train_data.iloc[:,0:-1].values
        y_train = train_data.iloc[:,-1].values
        
        logger.debug(f"Data loaded successfully. Shape: {X_train.shape}")
        return X_train, y_train
    except FileNotFoundError:
        logger.error("Training data file not found")
        raise
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise

def train_model(X_train, y_train, model_params):
    try:
        logger.info("Training GradientBoostingClassifier")
        model = LogisticRegression(
            C = model_params['C'],
            solver = model_params['solver'],
            penalty = model_params['penalty'],
            random_state= model_params['random_state'],
        )
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def save_model(model):
    try:
        logger.info("Saving trained model")
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def main():
    try:
        # Load parameters
        model_params = load_params()
        
        # Load training data
        X_train, y_train = load_training_data('./data/features/train_bow.csv')
        
        # Train model
        model = train_model(X_train, y_train, model_params)
        
        # Save model
        save_model(model)
        
        logger.info("Model building pipeline completed successfully")
    except Exception as e:
        logger.error(f"Fatal error in model building pipeline: {str(e)}")
        raise
    finally:
        logger.info("Model building process finished")

if __name__ == "__main__":
    main()