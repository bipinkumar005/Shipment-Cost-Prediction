import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import yaml
import lightgbm 

def display_categories():
    categories_info = {
        "Artist_Reputation": ['Low_Reputation', 'High_Reputation', 'Moderate_Reputation'],
        "Material": ['Brass', 'Clay', 'Aluminium', 'Bronze', 'Wood', 'Stone', 'Marble'],
        "International": ['Yes', 'No'],
        "Express_Shipment": ['Yes', 'No'],
        "Installation_Included": ['No', 'Yes'],
        "Transport": ['Airways', 'Roadways', 'Waterways'],
        "Fragile": ['No', 'Yes'],
        "Customer_Information": ['Working Class', 'Wealthy'],
        "Remote_Location":['urban','remote'],
        "State":['Georgia', 'Pennsylvania', 'North Carolina', 'Alabama', 'Arkansas',
                'AA', 'Mississippi', 'South Carolina', 'New Jersey', 'AP',
                'Wyoming', 'California', 'Massachusetts', 'Missouri', 'Tennessee',
                'District of Columbia', 'North Dakota', 'Utah', 'Louisiana',
                'Kansas', 'Delaware', 'Illinois', 'Idaho', 'Oregon', 'Arizona',
                'Florida', 'Nebraska', 'Virginia', 'West Virginia', 'Oklahoma',
                'Connecticut', 'Alaska', 'Maine', 'Maryland', 'New Mexico',
                'Rhode Island', 'Colorado', 'Michigan', 'Vermont', 'Kentucky',
                'Montana', 'Iowa', 'Indiana', 'New Hampshire', 'Nevada', 'Ohio',
                'AE', 'Minnesota', 'Texas', 'South Dakota', 'Washington',
                'New York', 'Wisconsin', 'Hawaii']
    }

    st.title("Categorical Features")

    # Collect user input for each category
    categorical_data = {}
    for category, values in categories_info.items():
        user_input = st.selectbox(f"Select {category} category:", values)
        categorical_data[category] = user_input

    return categorical_data


def collect_input_data():
    st.title("Numerical Features")

    # Collect numerical inputs
    weight = st.number_input("Weight", min_value=0.0)
    price_of_sculpture = st.number_input("Price of Sculpture", min_value=0.0)
    base_shipping_price = st.number_input("Base Shipping Price", min_value=0.0)
    Height = st.number_input("Height", min_value=0.0)
    Width = st.number_input("Width", min_value=0.0)

    # Log-transform numerical inputs
    weight_log = np.log1p(weight + 1)
    price_of_sculpture_log = np.log1p(price_of_sculpture + 1)
    base_shipping_price_log = np.log1p(base_shipping_price + 1)
    Height_log = np.log1p(Height + 1)
    Width_log = np.log1p(Width + 1)

    # Collect categorical inputs
    categorical_data = display_categories()

    # Combine numerical and categorical data into a DataFrame
    input_data = pd.DataFrame({
        "Weight": [weight_log],
        "Price_Of_Sculpture": [price_of_sculpture_log],
        "Base_Shipping_Price": [base_shipping_price_log],
        "Height": [Height_log],
        "Width": [Width_log],
        **categorical_data
    })

    return input_data


def load_model(model_path):
    """
    Load a machine learning model from a joblib file.

    Parameters:
    - model_path (str): The file path to the joblib file containing the model.

    Returns:
    - model: The loaded machine learning model.
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading the model from {model_path}: {e}")
        return None


def make_prediction(model, input_data):
    # Replace this with the actual prediction logic based on your models
    prediction = model.predict(input_data)
    
    st.write("Predictions :")
    st.write(prediction)
    return  round(prediction[0], 2)




def preprocess_data(input_data, encoder):
    # Apply preprocessing to the input data

    # Print input data columns before transformation
    print("Input Data Columns (Before Transformation):")
    column_order = ['Artist_Reputation', 'Height', 'Width', 'Weight', 'Material',
                    'Price_Of_Sculpture', 'Base_Shipping_Price', 'International',
                    'Express_Shipment', 'Installation_Included', 'Transport', 'Fragile',
                    'Customer_Information', 'Remote_Location', 'State']

    input_data = input_data[column_order]

    categorical_columns = ['Artist_Reputation',
                           'Material',
                           'International',
                           'Express_Shipment',
                           'Installation_Included',
                           'Transport',
                           'Fragile',
                           'Customer_Information',
                           'Remote_Location',
                           'State']

    # Use the pre-trained encoder to transform the categorical columns
    one_hot_encoded_data = encoder.transform(input_data[categorical_columns])

    # Concatenate the one-hot encoded features with the non-categorical features
    input_data_transformed = np.concatenate([input_data.drop(columns=categorical_columns).values, one_hot_encoded_data], axis=1)

    return input_data_transformed



def load_preprocessor_from_file(file_path):
    # Load the preprocessor from the specified file
    preprocessor = joblib.load(file_path)
    return preprocessor

def load_yaml_parameters(file_path):
    """
    Load parameters from a YAML file.

    Parameters:
    - file_path: Path to the YAML file.

    Returns:
    - parameters: Dictionary containing parameters.
    """
    with open(file_path, 'r') as file:
        parameters = yaml.safe_load(file)
    return parameters

def load_and_display_yaml(file_path):
    """
    Load parameters from a YAML file and display them in Streamlit.

    Parameters:
    - file_path: Path to the YAML file.

    Returns:
    - None
    """

    # Load parameters from YAML file
    parameters = load_yaml_parameters(file_path)



    # Extract model-specific parameters
    model_name = parameters.get(f"Model", "Unknown")
    model_params_str = parameters.get("Model_Parameters", "")
    r2_score = parameters.get("R2_Score", "Unknown")

    # Display parameters in Streamlit
    st.sidebar.title(f"{model_name}")
    
    # Display model name and R2 score
    st.sidebar.text(f"Model: {model_name}")
    st.sidebar.text(f"R2 Score: {r2_score}")

    # Display model-specific parameters
    if model_params_str:
        st.sidebar.text("Model Parameters:")
        try:
            # Parse the model parameters string as a dictionary
            model_params = yaml.safe_load(model_params_str.replace("''", '"'))
            for param_name, param_value in model_params.items():
                st.sidebar.text(f"{param_name}: {param_value}")
        except Exception as e:
            st.sidebar.error(f"Error parsing model parameters: {e}")
            
            
def make_prediction_with_loading_spinner(model, input_data):
    with st.spinner("Making Predictions..."):
        prediction = model.predict(input_data)
    return round(np.exp(prediction[0]), 2)


def main():
    st.title("Cost Prediction")

    # Load models from the specified folder
    #Xg_boost_model = load_model(model_path="Notebook/Models/XGBoost/XGBoost_model.joblib")
    Rf_model = load_model(model_path="Notebook/Models/Random Forest/Random Forest_model.joblib")
    #gbm_model = load_model(model_path="Notebook/Models/LightGBM/LightGBM_model.joblib")
    # Assuming preprocess_object is your preprocessing object
    preprocessor = load_preprocessor_from_file(file_path="Notebook/Preprocessor/one_hot_encoder.joblib")  # Implement a function to load your preprocessor object

    # User input for prediction data
    input_data = collect_input_data()

    # Display the input_data DataFrame on the Streamlit page
    st.write("Input Data:")
    st.write(input_data)

    if st.button("Make Predictions"):
        try:
            # Preprocess the input data
            print("---------")
            print(type(preprocessor))
            input_data = preprocess_data(input_data, preprocessor)

            # Display the input_data DataFrame on the Streamlit page
            st.write("Preprocessed Data:")
            st.write(input_data)

            Predictions = []

            # Display predictions and parameters in three columns
            col1, _, _ = st.columns(3)

            # XG Boost Model
         #   with col1:
          #      st.write("XG Boost Model Prediction")
         #       prediction_xgb = make_prediction_with_loading_spinner(Xg_boost_model, input_data)
        #        st.write(f"Cost Prediction: {round(prediction_xgb,2)}")
        #        Predictions.append(round(prediction_xgb,2))
        #        load_and_display_yaml(file_path="Notebook/Models/XGBoost/XGBoost_params.yaml")

            # Random Forest Model
            with col1:
                st.write("Random Forest Model Prediction")
                prediction_rf = make_prediction_with_loading_spinner(Rf_model, input_data)
                print(" Prediction Done")
                st.write(f"Cost Prediction: {prediction_rf}")
                Predictions.append(prediction_rf)
                load_and_display_yaml(file_path="Notebook/Models/Random Forest/Random Forest_params.yaml")

            # LightGBM Model
         #   with col3:
          #      st.write("Gradient Boost Model Prediction")
          #      prediction_gbm = make_prediction_with_loading_spinner(gbm_model, input_data)
        #        st.write(f"Cost Prediction: {prediction_gbm}")
          #      Predictions.append(prediction_gbm)
         #       load_and_display_yaml(file_path="Notebook/Models/LightGBM/LightGBM_params.yaml")

            avg_prediction = np.mean(Predictions)

            st.success("Predictions Completed!")
            st.write("Predictions:")
            formatted_avg_prediction = "{:.2f}".format(avg_prediction)
            st.success(formatted_avg_prediction)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()