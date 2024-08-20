# Disaster Response Pipelines

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Project Descriptions](#descriptions)
4. [Files Descriptions](#files)
5. [Instructions](#instructions)

## Installation <a name="installation"></a>

All necessary libraries are included in the Anaconda distribution of Python. The libraries used in this project are:

- pandas
- re
- sys
- json
- sklearn
- nltk
- sqlalchemy
- pickle
- Flask
- plotly
- sqlite3

Ensure that the code is run using Python version 3.*.

## Project Motivation <a name="motivation"></a>

The objective of this project is to classify disaster messages into predefined categories. 
The dataset for this project was sourced from [Figure Eight](https://www.figure-eight.com/). The project involves building a model that can classify disaster messages via an API. 
Users can input new messages through a web application and receive classification results across multiple categories. The web app also provides visualizations of the dataset.

## Project Descriptions <a name="descriptions"></a>

This project comprises three main components:

1. **ETL Pipeline:** The `process_data.py` file contains the script for the ETL (Extract, Transform, Load) pipeline, which performs the following tasks:
   - Loads the `messages` and `categories` datasets.
   - Merges these two datasets.
   - Cleans the combined data.
   - Stores the cleaned data in a SQLite database.

2. **ML Pipeline:** The `train_classifier.py` file includes the script for the ML (Machine Learning) pipeline, which:
   - Loads data from the SQLite database.
   - Splits the dataset into training and test sets.
   - Constructs a text processing and machine learning pipeline.
   - Trains and tunes a model using GridSearchCV.
   - Evaluates the model on the test set.
   - Exports the trained model as a pickle file.

3. **Flask Web App:** The web application allows users to input disaster messages and view their classifications across various categories. 
   The application also features visualizations to represent the data.

## Files Descriptions <a name="files"></a>

The project directory structure is as follows:

- `README.md`: This file.
- `workspace/`
  - `app/`
    - `run.py`: Flask script to start the web application.
  - `templates/`
    - `master.html`: Main page of the web application.
    - `go.html`: Result page showing classification outcomes.
  - `data/`
    - `disaster_categories.csv`: Dataset containing category labels.
    - `disaster_messages.csv`: Dataset containing messages.
    - `DisasterResponse.db`: SQLite database file for disaster response.
    - `process_data.py`: Script for the ETL process.
  - `models/`
    - `train_classifier.py`: Script for training the classification model.

## Instructions <a name="instructions"></a>

To run the application, follow these steps:

1. Execute the following commands in the root directory of the project to set up the database and model:

    - To run the ETL pipeline, which cleans data and stores it in the database:
        ```bash
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
    - To run the ML pipeline, which trains the classifier and saves it:
        ```bash
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. Navigate to the app directory and start the web application with the following command:
    ```bash
    python run.py
    ```

3. Access the web application in your browser at:
    ```
    http://0.0.0.0:3001/
    ```
