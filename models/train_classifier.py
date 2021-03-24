import sys
import numpy as np
import pandas as pd
import re
import pickle
import nltk 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords'])
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sqlalchemy import create_engine
from sklearn.externals import joblib

def load_data(database_filepath):
    """ Load the database and define feature(X) and target(Y) variables
    [Args] 
        database_filepath (str): SQL database file path 
    [Returns]
        (pd.Series): Feature varaibles X 
        (pd.DataFrame): Multi-Target (or label) variables Y 
        (list(str)): list of the category names 
    """
    # load data from sql engine 
    engine = create_engine(f'sqlite:///{database_filepath}') 
    df = pd.read_sql_table(database_filepath[:-3], con=engine) 
    
    # define X and Y
    category_names = df.columns[4:].tolist()
    X = df['message'] # pandas series (not dataframe) for CountVectorizer input
    Y = df[category_names] 

    return X, Y, category_names
    
    
def tokenize(text):
    """ generate token features from input message
    [Args] 
        test (str): input message 
    [Returns] 
        (list[str]): list of the processed tokens
    """
    # replace punctuations with a whitespace 
    text = re.sub(r'[^a-zA-Z0-9]', '', text) 
    
    # tokenize with lemmatizing and stopwords removal 
    lemmatizer = WordNetLemmatizer() 
    tokens = [lemmatizer.lemmatize(t.lower().strip()) for t in word_tokenize(text) \
              if t not in stopwords.words("english")] 
    
    return tokens
                       
    
def build_model():
    """ build the pipeline using MultiOutputClassifier, 
        and then, wrap the pipeline with GridSearchCV 
        to tune the hyper parameters of the model
    [Args]
        None
    [Returns]
        (obj): GridSearchCV model 
    """
    # define the nested pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)), 
        ('tfidf', TfidfTransformer()), 
        ('predictor', MultiOutputClassifier(RandomForestClassifier())), #RandomForestClassifier, LogisticRegression
    ])

    # define Grid Search (just fix the best parameter that I found from the notebook)
    parameters = dict(
#         vect__ngram_range=[(1,1)], #[(1, 1), (1, 2)],
#         vect__max_df=[0.5], #[0.5, 0.75, 1.0],
#         tfidf__use_idf=[True], #[True, False],
#         tfidf__norm = ['l1'], #['l1', 'l2'],
        
        predictor__estimator__random_state=[13], 
        # random forest classifier
        predictor__estimator__criterion=['entropy'], # ['gini', 'entropy']  
        # LogisticRegression
#         predictor__estimator__max_iter=[100],
#         predictor__estimator__solver=['liblinear'], #['liblinear', 'lbfgs', 'saga']
#         predictor__estimator__class_weight=[None], #[None, 'balanced']
    )
    model = GridSearchCV(pipeline, parameters, return_train_score=True)
    
    return model 


def evaluate_model(model, X_test, Y_test, category_names):
    """ generate the classification report based on test dataset 
        and print the result over the category names
        and generate the markdown table
    [Args]
        model (obj): the best fit model 
        X_test (pd.Series): test features 
        Y_Test (pd.DataFrame): test targets(GroundTruth)
        category_names (list(str)): evaluation category
    [Returns]
        None 
    """
    # inference on the test dataset
    Y_pred = pd.DataFrame(model.predict(X_test), columns=Y_test.columns)
    
    # generate the report 
    df_report = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1-score', \
                                      'distribution-0', 'distribution-1'])
  
    for col in Y_test.columns:
        # accuracy score from sklearn 'accruacy_score'
        df_report.loc[col, 'accuracy'] = accuracy_score(Y_test[col], Y_pred[col])
        # other scores from sklearn 'classification_report'
        sample_weight = compute_sample_weight('balanced', Y_test[col]) # imbalanced dataset 
        report_str = classification_report(Y_test[col], Y_pred[col], sample_weight=sample_weight)
        metrics = report_str.split('avg / total')[1].split('\n')[0].split()      
        df_report.loc[col, 'precision'] = float(metrics[0]) 
        df_report.loc[col, 'recall'] = float(metrics[1]) 
        df_report.loc[col, 'f1-score'] = float(metrics[2]) 
        # # counts the class
        df_report.loc[col, 'distribution-0'] = 1 - (Y_test[col].sum() / Y_test.shape[0])
        df_report.loc[col, 'distribution-1'] = Y_test[col].sum() / Y_test.shape[0]
    
    # add an average metrics of all categories 
    df_report.loc['avg',:] = df_report.mean()
    
    # align the index in the final dataframe
    df_report = df_report.loc[category_names+['avg'], :]
    # print the report to the terminal 
    print(df_report)  
    print(model.best_params_) 
    
    # save the report to the markdown table text (Optional)
    with open('models/eval_report_markdown.txt', 'wt') as mf:
        # insert the index to the first column 
        df_idx = pd.DataFrame(df_report.index.T, columns=['---'], index=df_report.index)
        df_report = pd.concat([df_idx, df_report], axis=1)
        # generate the format 
        fmt = ['---' for i in range(len(df_report.columns))]
        df_fmt = pd.DataFrame([fmt], columns=df_report.columns)
        df_formatted = pd.concat([df_fmt, df_report])
        markdown_str = df_formatted.to_csv(sep="|", index=False)
        mf.write(markdown_str)
    print('Evaluation report is generated in models/eval_report_markdown.txt')
    
    
def save_model(model, model_filepath):
    """ export the model as a pickle file
    [Args]
        model (obj): the best fit model 
        model_filepath (str): the destination path of the pickle file 
    [Returns]
        None
    """
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
        
#         model = joblib.load('models/classifier_rf_36.pkl')
        
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