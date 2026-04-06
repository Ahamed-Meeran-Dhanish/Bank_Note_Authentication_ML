# Banknote Authentication API
A complete machine learning pipeline and REST API that predicts whether a given banknote is genuine or fake based on wavelet-transformed image features.

The model is trained using a Random Forest Classifier and is served via a FastAPI backend. Data validation is strictly handled to ensure accurate predictions.

# 🚀 Features

## Machine Learning Model:

Utilizes a RandomForestClassifier trained via scikit-learn.

## High Accuracy:

The trained model achieved an evaluation accuracy of approximately 98.7% on the test split.

## FastAPI Backend: 

A lightweight, high-performance web framework serves the /predict endpoint.

## Data Validation:

Implements Pydantic base models to enforce type hints and validate incoming JSON payloads.

## Model Serialization: 

The trained classifier is serialized and exported as classifier.pkl using Python's built-in pickle library.

# 🛠️ Tech Stack

## Data Science:

Python

pandas

numpy

scikit-learn

## API & Server: 

FastAPI

uvicorn

pydantic

## Environment: 

Jupyter Notebook (for initial EDA and model training)


# 📁 Project Structure

While your exact directory might vary, the application utilizes the following core components:

## Notebook.ipynb: 

Contains the data loading (BankNote_Authentication.csv), train-test splitting, model training, and exporting steps.

## Banknote.py:

Defines the BankNote Pydantic model for API request validation.

## app.py: 

The main FastAPI application that loads the classifier.pkl model and handles incoming POST requests.

## classifier.pkl: 

The serialized Random Forest model.

# ⚙️ Installation & Setup

## Clone the repository:

git clone <your-repository-url>
cd <your-repository-directory>

## Install the required dependencies:

Make sure you have your virtual environment activated, then install the necessary packages:

pip install fastapi uvicorn scikit-learn pandas pydantic numpy

## Run the API Server:

You can start the server directly using the provided script execution:
python app.py

# 📡 API Usage

Endpoint: /predict
Method: POST

Description: Accepts banknote features and returns a classification prediction.

## Request Body (JSON):

The API expects four continuous float variables extracted from the banknote's wavelet transform:

JSON

{
  "variance": 3.6216,
  
  "skewness": 8.6661,
  
  "curtosis": -2.8073,
  
  "entropy": -0.44699
}


## Response:

The API evaluates the input array against the loaded pickle file.

If the model's prediction output is greater than 0.5, it returns "Fake Note".

Otherwise, it returns "Its a Bank Note".

JSON

{

  "prediction": "Its a Bank Note"

}
