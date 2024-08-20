# Disaster Response Pipelines

## Overview

This project focuses on classifying disaster messages into distinct categories. The data used is sourced from [Figure Eight](https://www.figure-eight.com/), and the final product includes a web application that provides real-time message classification and data visualizations.

### Contents

- [Installation](#installation)
- [Motivation](#motivation)
- [Project Components](#project-components)
- [File Structure](#file-structure)
- [Usage Instructions](#usage-instructions)

## Installation

Ensure you have Python 3.* and the required libraries. The libraries included in the Anaconda distribution are:

- `pandas`
- `re`
- `sys`
- `json`
- `sklearn`
- `nltk`
- `sqlalchemy`
- `pickle`
- `Flask`
- `plotly`
- `sqlite3`

## Motivation

The primary goal is to build a system capable of classifying disaster messages into predefined categories. Users can input new messages through a web interface and get classification results. The project also visualizes the dataset for better insight.

## Project Components

### 1. ETL Pipeline

**File:** `process_data.py`

**Purpose:** This script manages the ETL (Extract, Transform, Load) process:

- Reads the `messages` and `categories` datasets.
- Merges the datasets into a single dataframe.
- Cleans the data and saves it to a SQLite database.

### 2. Machine Learning Pipeline

**File:** `train_classifier.py`

**Purpose:** This script creates and trains a machine learning model:

- Retrieves data from the SQLite database.
- Splits data into training and testing sets.
- Constructs and trains a model using GridSearchCV.
- Evaluates the model and saves it as a pickle file.

### 3. Flask Web Application

**Purpose:** Provides a web interface for users to:

- Input disaster messages.
- Receive classification results in multiple categories.
- View data visualizations.

## File Structure

Here’s an overview of the project directory structure:

- **`README.md`**: This document.
- **`workspace/`**:
  - **`app/`**:
    - **`run.py`**: Script to run the Flask web application.
  - **`templates/`**:
    - **`master.html`**: The main page of the web application.
    - **`go.html`**: Page displaying the classification results.
  - **`data/`**:
    - **`disaster_categories.csv`**: Dataset with category labels.
    - **`disaster_messages.csv`**: Dataset with messages.
    - **`DisasterResponse.db`**: SQLite database file.
    - **`process_data.py`**: ETL script.
  - **`models/`**:
    - **`train_classifier.py`**: Script for training the classification model.

## Usage Instructions

Follow these steps to set up and run the application:

1. **Set up the database and model**:
    ```bash
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    ```

2. **Run the web application**:
    ```bash
    python workspace/app/run.py
    ```

3. **Access the application**:
    Visit `http://0.0.0.0:3001/` in your web browser.
