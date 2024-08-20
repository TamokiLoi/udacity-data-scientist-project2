import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

# Download necessary NLTK data
nltk.download(['wordnet', 'punkt', 'stopwords'])

def load_data(database_filepath):
    """
    Function: Load data from the specified database file.
    
    Args:
    database_filepath (str): Path to the SQLite database file.
    
    Returns:
    X (DataFrame): DataFrame containing text messages.
    Y (DataFrame): DataFrame containing the target labels.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_messages_tbl', engine)
    X = df['message']  # Extract text messages column
    Y = df.iloc[:, 4:]  # Extract classification labels columns
    return X, Y

def tokenize(text):
    """
    Function: Tokenize and lemmatize the input text.
    
    Args:
    text (str): The text message to process.
    
    Returns:
    lemm (list of str): List of lemmatized words from the text.
    """
    # Normalize the text by replacing non-alphanumeric characters with spaces
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize the normalized text into words
    words = word_tokenize(text)
    
    # Remove stop words
    stop = stopwords.words("english")
    words = [word for word in words if word not in stop]
    
    # Lemmatize the words
    lemm = [WordNetLemmatizer().lemmatize(word) for word in words]
    
    return lemm

def build_model():
    """
    Function: Construct a machine learning pipeline and configure Grid Search for hyperparameter tuning.
    
    Returns:
    cv (GridSearchCV): The GridSearchCV object set up with the pipeline and parameter grid.
    """
    # Define the machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    # Set up Grid Search parameters
    parameters = {
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [50, 60, 70]
    }
    
    # Create GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test):
    """
    Function: Evaluate the performance of the model using classification metrics.
    
    Args:
    model: The trained model.
    X_test (DataFrame): Test set of text messages.
    Y_test (DataFrame): Test set of labels.
    """
    # Predict labels for the test set
    y_pred = model.predict(X_test)
    
    # Print classification report for each output category
    for i, col in enumerate(Y_test.columns):
        print(f'Feature {i + 1}: {col}')
        print(classification_report(Y_test[col], y_pred[:, i]))
    
    # Calculate and print model accuracy
    accuracy = (y_pred == Y_test.values).mean()
    print(f'The model accuracy is {accuracy:.3f}')

def save_model(model, model_filepath):
    """
    Function: Save the trained model as a pickle file.
    
    Args:
    model: The trained model.
    model_filepath (str): Path where the pickle file will be saved.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data from database: {database_filepath}')
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building and tuning model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)
        
        print(f'Saving model to: {model_filepath}')
        save_model(model, model_filepath)

        print('Model training complete and saved!')
    else:
        print('Usage: python train_classifier.py <database_filepath> <model_filepath>\n'
              'Example: python train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
