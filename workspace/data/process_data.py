import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge datasets from specified file paths.

    Args:
    messages_filepath (str): Path to the CSV file containing messages data.
    categories_filepath (str): Path to the CSV file containing categories data.

    Returns:
    DataFrame: A DataFrame obtained by merging messages and categories data.
    """
    # Read CSV files into DataFrames
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge DataFrames on 'id' column
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """
    Process the DataFrame by cleaning and structuring category data.

    Args:
    df (DataFrame): DataFrame containing both messages and categories.

    Returns:
    DataFrame: The cleaned DataFrame with individual category columns.
    """
    # Split the 'categories' column into multiple columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract the category names from the first row
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0]).tolist()
    categories.columns = category_colnames
    
    # Convert the category values to numeric (0 or 1)
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
    
    # Drop the original 'categories' column from the DataFrame
    df = df.drop(['categories'], axis=1)
    
    # Concatenate the cleaned categories DataFrame with the original DataFrame
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    """
    Save the cleaned DataFrame into an SQLite database.

    Args:
    df (DataFrame): The DataFrame to be stored in the database.
    database_filename (str): The name of the SQLite database file.
    """
    # Create an SQLAlchemy engine for SQLite
    engine = create_engine(f'sqlite:///{database_filename}')
    
    # Write the DataFrame to the database
    df.to_sql('disaster_messages_tbl', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)
        
        print('Data processing complete. Cleaned data has been saved to the database!')
    else:
        print('Usage: python process_data.py <messages_filepath> <categories_filepath> <database_filepath>\n'\
              'Example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db')

if __name__ == '__main__':
    main()