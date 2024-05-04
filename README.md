# DATS-6303-Deep-Learning-project
Final project for the DATS 6303 Deep Learning course, spring 2024. 

The project is aimed at employing time series prediction approach for music generation using deep neural netowkrs. The models explored were GRU, LSTM (stacked and bidirectional), and Multi-head attention-based LSTM. The idea is to first represemt the music symbolically and then convert the sequences into a time series like representation. The models are trained on dataset of 5000+ folk songs melodies and after training, the model is given a melody seed to extrapolate further to a fixed number of time steps.  

# Introduction

This repository contains the culmination of our team's efforts in developing four distinct machine learning models aimed to predict melodies. Each model is housed in its own directory within the final_models folder. Additionally, we have included a Streamlit application that demonstrates the practical application of our models in an interactive web interface.

# Repository Structure

final_models/: Contains individual folders for each of the four models we developed. Each folder is named after the model it contains and includes all necessary files for that model.
streamlit_app/: Contains all files necessary to run our Streamlit application, which showcases the capabilities of our models in an interactive format.
Individual Contributions

Each team member's report detailing their contributions to the project is available within the repository. These reports provide insights into the individual efforts and roles in the development of the project.

# Getting Started
# Prerequisites:

AWS Account (for model deployment)
Python 3.x
Streamlit

For each model, you need to fork the respective folder to your system. Ensure that you upload the model data to AWS as the models rely on AWS configurations to function correctly.
Streamlit Application Setup
Navigate back to the root directory and then to the streamlit_app directory:

# Run the Streamlit application locally:

# streamlit run app.py
No external connections are required for the Streamlit application; it runs directly once forked into your system.
Deployment
AWS Deployment
Ensure that all model files are uploaded to your AWS account as the models leverage AWS services for full functionality.
Follow AWS documentation to set up model serving resources (e.g., Amazon S3 for storage, AWS Lambda for computing).
