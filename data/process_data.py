import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine 


def load_data(messages_filepath, categories_filepath):
    ''' load the meesages and categories files and merge them
    [Args] 
       messages_filepath (str): csv file path of message data
       categories_filepath (str): csv file path of category data
    [returns] 
       (pd.DataFrame): a merged data frame 
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='outer', on='id') 
    
    return df


def clean_data(df):
    """ cleans the values of category data into separate values
        and remove the duplicates in the data frame 
    [Args] 
        df (pd.DataFrame): raw dataframe 
    [Returns] 
        (pd.DataFrame): cleaned dataframe 
    """
    # split the category column into separate columns 
    df_catg = df['categories'].str.split(pat=';', expand=True)
    # replace column names with something meaningful based on the first row
    df_catg.columns = df_catg.loc[0].str.replace(r"-[0,1]$", "", regex=True) 
    
    # convert the category values to numerical numbers either 0 or 1 
    for col in df_catg: 
        df_catg[col] = df_catg[col].str.replace(f"^{col}-", "", regex=True) 
        df_catg[col] = df_catg[col].astype(int) # str->int
        # make sure the value has either 0 or 1
        if df_catg[col].value_counts().shape[0] > 2:
            # replace the numbers of neither 0 or 1 into 0 
            df_catg.loc[df_catg[col] >= 2, col] = 0 
        if df_catg[col].value_counts().shape[0] < 2:
            # drop the column (No need to classify)
            df_catg = df_catg.drop([col], axis=1)

    # replace categories column in df with new df_catg 
    df_clean = pd.concat([df, df_catg], axis=1) 
    df_clean = df_clean.drop(['categories'], axis=1) 
    
    # remove the duplicates 
    df_clean = df_clean.drop_duplicates(subset=['id'], keep='first') 
    
    return df_clean
    
    
def save_data(df, database_filename):
    """ Save the dataframe to the relational database with SQLAlchemy library 
    [Args] 
        df (pd.Dataframe): the processed disaster data 
        database_filename (str): the desination db file path 
    [Returns] 
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}') 
    df.to_sql(database_filename[:-3], engine, index=False, if_exists='replace') 

    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()