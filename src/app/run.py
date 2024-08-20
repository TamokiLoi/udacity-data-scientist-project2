import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    """
    Function: Tokenize and lemmatize the input text.
    
    Args:
    text (str): The text message to process.
    
    Returns:
    clean_tokens (list of str): List of lemmatized and cleaned words from the text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Load data from the SQLite database
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages_tbl', engine)

# Load the pre-trained model from a pickle file
model = joblib.load("../models/classifier.pkl")

@app.route('/')
@app.route('/index')
def index():
    """
    Function: Render the index page with visualizations of the data.
    
    Returns:
    Rendered HTML template with visualizations.
    """
    # Extract data for visualizations
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Create visualizations for message categories and top categories
    catg_nam = df.iloc[:, 4:].columns
    bol = df.iloc[:, 4:] != 0
    cat_bol = bol.sum().values

    sum_cat = df.iloc[:, 4:].sum()
    top_cat = sum_cat.sort_values(ascending=False)[1:11]
    top_cat_names = list(top_cat.index)

    # Define the graphs to be displayed
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=catg_nam,
                    y=cat_bol
                )
            ],

            'layout': {
                'title': 'Message Categories Distribution',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_cat_names,
                    y=top_cat
                )
            ],

            'layout': {
                'title': 'Top 10 Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    
    # Encode plotly graphs to JSON format for rendering
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render the master HTML template with the graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    """
    Function: Handle user input, use the model to make predictions, and render results.
    
    Returns:
    Rendered HTML template showing the query and model predictions.
    """
    # Retrieve user query from request arguments
    query = request.args.get('query', '') 

    # Predict the classification results for the user query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render the go.html template with the query and classification results
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    """
    Function: Start the Flask web application.
    """
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
