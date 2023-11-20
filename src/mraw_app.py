import streamlit as st
import xgboost as xgb
import numpy as np
import markdown
import os

# Load the saved models
loaded_model_child = xgb.XGBRegressor()
loaded_model_adult = xgb.XGBRegressor()
loaded_model_birth = xgb.XGBRegressor()
loaded_model_sixty = xgb.XGBRegressor()
script_dir = os.path.dirname(os.path.abspath(__file__))
child_model_path = os.path.join(script_dir, "xgb_child_mort.model")
adult_model_path = os.path.join(script_dir, "xgb_adult_mort.model")
birth_model_path = os.path.join(script_dir, "xgb_life_expect_birth.model")
sixty_model_path = os.path.join(script_dir, "xgb_life_expect_sixty.model")

loaded_model_child.load_model(child_model_path)
loaded_model_adult.load_model(adult_model_path)
loaded_model_birth.load_model(birth_model_path)
loaded_model_sixty.load_model(sixty_model_path)

# Read the markdown file for description
mraw_description = os.path.join(script_dir, "mraw_description.md")
with open(mraw_description, "r") as file:
    description_markdown = file.read()

# Read the markdown file for article
mraw_article = os.path.join(script_dir, "mraw_article.md")
with open(mraw_article, "r") as file:
    article_markdown = file.read()

# Convert markdown content to HTML for description
description_html = markdown.markdown(description_markdown)

# Convert markdown content to HTML for article
article_html = markdown.markdown(article_markdown)

# Define the prediction function
###
# Assuming num_features is the actual number of features your model expects
num_features_child = 20
num_features_adult = 16

def predict_mortality_rates(percentage_clean_water):
    # Duplicate the user input to match the number of features
    input_data_child = np.full((1, num_features_child), percentage_clean_water)
    input_data_adult = np.full((1, num_features_adult), percentage_clean_water)

    # Make predictions using all 4 models
    prediction_child = loaded_model_child.predict(input_data_child)
    prediction_adult = loaded_model_adult.predict(input_data_adult)
    prediction_birth = loaded_model_birth.predict(input_data_child)
    prediction_sixty = loaded_model_sixty.predict(input_data_child)

    return (
        f"{prediction_child[0]:.2f}",
        f"{prediction_adult[0]:.2f}",
        f"{prediction_birth[0]:.2f}",
        f"{prediction_sixty[0]:.2f}"
    )
###
# Streamlit UI
st.set_page_config(
    page_title="MRAW",
    page_icon="🚰",
)
st.markdown("<h1 style='text-align: center; color: lightskyblue;'>MRAW</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: lightskyblue;'>Mortality Rates prediction based on Access to Water</h5>", unsafe_allow_html=True)

#Center image
col1, col2, col3 = st.columns(3)
with col1:
     st.write(" ")
with col2:
     water_picture = os.path.join(script_dir, "drinking_water.jpg")
     st.image(water_picture, width= 125)
with col3:
     st.write(" ")

with st.expander("**MRAW summary**"):
    st.markdown(description_html, unsafe_allow_html=True)
    multi = '''
    Given the **Percentage of People using Clean Drinking Water in a Country**,   
    the ML models predict different **Mortality rates**.
       
    '''
    st.markdown(multi)

# User input
percentage_clean_water = st.number_input("Enter the Percentage of People using Clean Drinking Water in a Country")

# Prediction
predictions = predict_mortality_rates(percentage_clean_water)

# Display predictions
st.text("Predicted Children (under 5) Mortality rate: {:.2f}".format(float(predictions[0])))
st.text("Predicted Adult (15 and above) Mortality rate: {:.2f}".format(float(predictions[1])))
st.text("Predicted Life Expectancy at Birth (in years): {:.2f}".format(float(predictions[2])))
st.text("Predicted Life Expectancy at age 60 (in years): {:.2f}".format(float(predictions[3])))

# Display article
with st.expander("**Learn more about MRAW**"):
   st.markdown(article_html, unsafe_allow_html=True)