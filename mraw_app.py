import xgboost as xgb
import gradio as gr

# Load the saved models
loaded_model_child = xgb.XGBRegressor()
loaded_model_adult = xgb.XGBRegressor()
loaded_model_birth = xgb.XGBRegressor()
loaded_model_sixty = xgb.XGBRegressor()
loaded_model_child.load_model("child_mort_data/xgb_child_mort.model")
loaded_model_adult.load_model("adult_mort_data/xgb_adult_mort.model")
loaded_model_birth.load_model("adult_mort_data/xgb_life_expect_birth.model")
loaded_model_sixty.load_model("adult_mort_data/xgb_life_expect_sixty.model")

# Define the prediction function
def predict_mortality_rates(percentage_clean_water):
    # Make predictions using all 4 models
    prediction_child = loaded_model_child.predict([[percentage_clean_water]])
    prediction_adult = loaded_model_adult.predict([[percentage_clean_water]])
    prediction_birth = loaded_model_birth.predict([[percentage_clean_water]])
    prediction_sixty = loaded_model_sixty.predict([[percentage_clean_water]])
    
    return (
        f"{prediction_child[0]:.2f}",
        f"{prediction_adult[0]:.2f}",
        f"{prediction_birth[0]:.2f}",
        f"{prediction_sixty[0]:.2f}"
    )

# Create the Gradio interface
input_text = gr.inputs.Number(label="Percentage of People Accessing Clean Water")
output_text1 = gr.outputs.Textbox(label="Predicted Children (under 5) Mortality rate")
output_text2 = gr.outputs.Textbox(label="Predicted Adult (15 and above) Mortality rate")
output_text3 = gr.outputs.Textbox(label="Predicted Life Expectancy at Birth (in years)")
output_text4 = gr.outputs.Textbox(label="Predicted Life Expectancy at age 60 (in years)")

interface = gr.Interface(
    fn=predict_mortality_rates,
    inputs=input_text,
    outputs=[output_text1, output_text2, output_text3, output_text4],
    layout="vertical"
)

# Launch the Gradio interface
interface.launch(share=True)