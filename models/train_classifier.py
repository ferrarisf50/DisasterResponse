# !pip install xgboost
import sys
import pandas as pd
import numpy as np
import sqlite3
import re
import nltk
import pickle
import xgboost as xgb

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') 
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score,classification_report
#from sqlalchemy import create_engine

pd.set_option("max_colwidth",1000000)
pd.set_option('max_columns', 15000)

def load_data(database_filepath):
    '''
    INPUT database path
    
    OUTPUT feature dataset and response dataset
    '''
    conn = sqlite3.connect('DisasterResponse.db')
    df = pd.read_sql('SELECT * FROM DisasterResponse', conn)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    '''
    INPUT  a text string
    
    OUTPUT  a list of tokenized words 
    '''
    text = text.lower()
    
    # remove urls:   
    text = re.sub(r'http(s)?://[^ ]+','',text)
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    words = word_tokenize(text)
    
    words = [w for w in words if w not in stopwords.words("english")]
    
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    result = stemmed
    return result


def build_model():
    '''

    
    OUTPUT: a classifier
    '''
    pipeline = Pipeline([       
              ('vect', CountVectorizer(tokenizer=tokenize)),
              ('tfidf', TfidfTransformer()),
#              ('clf', MultiOutputClassifier(RandomForestClassifier()))
         ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])
    
    parameters = {
#         'vect__ngram_range': ((1, 1), (1, 2)),
#         'vect__max_df': (0.5, 0.75, 1.0),
#         'vect__max_features': (None, 5000, 10000),
#         'tfidf__use_idf': (True, False),
#         'clf__n_estimators': [100, 200, 500],
        'clf__estimator__min_samples_split': [2, 3]
        
#         'features__transformer_weights': (
#             {'text_pipeline': 1, 'starting_verb': 0.5},
#             {'text_pipeline': 0.5, 'starting_verb': 1},
#             {'text_pipeline': 0.8, 'starting_verb': 1},
#         )
    }


    cv = GridSearchCV(pipeline, param_grid=parameters,verbose = 10, n_jobs=-1,return_train_score=True)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT: model - the classifier
        X_test - feature columns of the test data set 
        Y_test - response columns of the test data set 
        category_names - the list of the feature's names.
    '''
    
    Y_pred = model.predict(X_test)
    overall_accuracy = (Y_pred == Y_test).mean().mean()

    print('overall accuracy: '+str(overall_accuracy))
    predict = pd.DataFrame(Y_pred, columns = category_names)
    for col in Y_test.columns:
        print('class: '+col)
        print(classification_report(Y_test[col],predict[col]))

def save_model(model, model_filepath):
    '''
    INPUT:  model - the classifier
            model_filepath - the saved path and file name of the classifer
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()