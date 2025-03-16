import gradio as gr
import joblib
import ast

# Load the model
pred_model = joblib.load("XGboot_Model.pkl")

# Define the prediction function
def predict_transaction(input_string):
    try:
        # Parse the input string to a Python list
        features = ast.literal_eval(input_string)
        # Ensure the input is in the correct format
        if not isinstance(features, list) or len(features) != 1 or not isinstance(features[0], list) or len(features[0]) != 29:
            return "Error: Input must be a single list with 29 numerical features."
        # Make the prediction
        prediction = pred_model.predict(features)
        # Return the result
        return "The transaction is Fraud." if prediction[0] == 1 else "The transaction is Normal."
    except Exception as e:
        return f"Error: {str(e)}"

# Define the Gradio interface
# Updated to use gr.components for interface elements
gr_interface = gr.Interface(
    fn=predict_transaction,
    inputs=gr.components.Textbox( # Changed to gr.components.Textbox()
        lines=5,
        placeholder="Enter the feature list in the following format:\n[[-2.31, 1.95, ..., -0.42]]",
        label="Input Features"
    ),
    outputs="text",
    title="Transaction Classification",
    description="Paste the feature list (29 numerical values) in the format: [[-2.31, 1.95, ..., -0.42]]. The model will classify the transaction as Normal or Fraud."
)

# Launch the interface
gr_interface.launch()