from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

app = Flask(__name__)

# Load the trained model
try:
    risk_model = joblib.load('risk_model.joblib')
except FileNotFoundError:
    raise FileNotFoundError("The model file 'risk_model.joblib' was not found. Ensure it's in the correct directory.")

# Load unique states, counties, and disaster types for forms and validation
try:
    data = pd.read_csv('PredictionDataSet.csv')
except FileNotFoundError:
    raise FileNotFoundError("The data file 'PredictionDataSet.csv' was not found. Ensure it's in the correct directory.")

disaster_types = ['Avalanche', 'Coastal Flooding', 'Cold Wave', 'Drought', 'Earthquake',
                  'Hail', 'Heatwave', 'Hurricane', 'Icestorm', 'Landslide', 'Lightning',
                  'Riverine', 'Flooding', 'Strong Wind', 'Tornado', 'Tsunami',
                  'Volcanic Activity', 'Wildfire', 'Winter Weather']

states = sorted(data['State'].unique())
counties = sorted(data['County'].unique())

# Create a dictionary mapping states to their counties for validation
state_to_counties = {
    state.lower(): sorted(data[data['State'].str.lower() == state.lower()]['County'].unique())
    for state in states
}

# Set up your Watsonx.ai credentials
# Note: It's highly recommended to use environment variables for sensitive information
# Replace the following with secure methods to handle credentials
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com/",
    api_key="iwDOQ_4_8eOg_QH86FpoLxfCo7vXlUFb6_eGolQbgdnW",
)

client = APIClient(credentials)

model_inference = ModelInference(
    model_id="mistralai/mistral-large",
    api_client=client,
    project_id="d0eaa248-e010-412c-8cf8-ba046b28f236",
    params={
        "max_new_tokens": 1000,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "stop": [".\n", "\n\n"]
    }
)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', states=states, counties=counties,
                           disaster_types=disaster_types)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle AJAX POST requests to calculate the base risk score.
    Expects JSON data with 'state', 'county', and 'disaster'.
    Returns JSON response with 'risk_score' or 'error'.
    """
    if request.is_json:
        global state, county, disaster, state_new
        data_input = request.get_json()
        state = data_input.get('state')
        county = data_input.get('county')
        disaster = data_input.get('disaster')
        state_new = state

        # Validate input presence
        if not all([state, county, disaster]):
            return jsonify({'error': 'Missing data: state, county, and disaster are required.'}), 400

        # Normalize inputs for case-insensitive comparison
        state_normalized = state.strip().lower()
        county_normalized = county.strip().lower()

        # Validate if the state exists
        if state_normalized not in state_to_counties:
            return jsonify({'error': f"Invalid state: '{state}'. Please enter a valid state."}), 400

        # Validate if the county exists within the selected state
        counties_in_state = [c.lower() for c in state_to_counties[state_normalized]]
        if county_normalized not in counties_in_state:
            return jsonify({'error': f"Invalid county: '{county}' does not belong to '{state}' or is not in dataset."}), 400

        # Fetch the correctly cased state and county from the dataset
        # to maintain consistency with the model's expectations
        state_correct = next(s for s in states if s.lower() == state_normalized)
        county_correct = next(c for c in state_to_counties[state_normalized] if c.lower() == county_normalized)

        # Create a DataFrame for prediction
        input_data = pd.DataFrame({
            'State': [state_correct],
            'County': [county_correct],
            'DisasterType': [disaster]
        })

        # Predict base risk
        try:
            base_risk = risk_model.predict(input_data)[0]
            base_risk = round(base_risk, 2)
        except Exception as e:
            return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

        return jsonify({'risk_score': base_risk})
    else:
        return jsonify({'error': 'Request must be in JSON format.'}), 400

@app.route('/additional_info', methods=['POST'])
def additional_info():
    if request.is_json:
        data_input = request.get_json()
        state = data_input.get('state')
        county = data_input.get('county')
        disaster = data_input.get('disaster')

        # Validate input presence
        if not all([state, county, disaster]):
            return jsonify({'error': 'Missing data: state, county, and disaster are required.'}), 400

        # Create the prompt for the AI model
        prompt = (
            f"Create a highly detailed Wildlife Animal Impact and Past Disaster Damages Information section regarding {disaster} "
            f"in {county}, {state_new}. Use proper HTML formatting, including <ul> for bullet lists, "
            f"<li> for list items, and <h3> or <h4> for headings. Your response should look like "
            f"a cleanly formatted website section, with separate sections for Surrounding Wildlife Impact (even including specific animals and their endangerments) and "
            f"Past Natural Disaster Damages in terms of financial loss in the area."
        )

        try:
            # Use the watsonx.ai API to get the response
            response = model_inference.generate_text(prompt)
            return jsonify({'additional_info': response})
        except Exception as e:
            return jsonify({'error': f'Error generating additional information: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Request must be in JSON format.'}), 400


@app.route('/response', methods=['POST'])
def response():
    """
    Handle AJAX POST requests to get response guidance.
    Returns JSON response with 'response' or 'error'.
    """
    prompt = (
        f"Create a highly detailed immediate action plan for an ongoing {disaster} "
        f"in {county}, {state_new}. Use proper HTML formatting, including <ul> for bullet lists, "
        f"<li> for list items, and <h3> or <h4> for headings. Your response should look like "
        f"a cleanly formatted website section, with separate sections for Evacuation Procedures, "
        f"Emergency Contacts, Safety Measures, and A LONG LIST OF nearby disaster evacuation cities."
    )

    try:
        # Use the watsonx.ai API to get the response
        response = model_inference.generate_text(prompt)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': f'Error generating response: {str(e)}'}), 500

@app.route('/recovery', methods=['POST'])
def recovery():
    """
    Handle AJAX POST requests to get recovery guidance.
    Expects JSON data with 'prompt'.
    Returns JSON response with 'response' or 'error'.
    """
    if request.is_json:
        prompt = (
            f"Create a highly detailed recovery plan for an approaching {disaster} "
            f"in {county}, {state_new}. Use proper HTML formatting, including <ul> for bullet lists, "
            f"<li> for list items, and <h3> or <h4> for headings. Your response should look like "
            f"a cleanly formatted website section, with separate sections for Post-Disaster Damage Control, "
            f"Post-Disaster Mental and Physichal Health Steps, and Links to Helpful Websites that aid with Disaster Rehabilitation."
        )

        """data_input = request.get_json()
        prompt = data_input.get('prompt')"""

        if not prompt:
            return jsonify({'error': 'Prompt is required.'}), 400

        try:
            # Use the watsonx.ai API to get the response
            response = model_inference.generate_text(prompt)
            return jsonify({'response': response})
        except Exception as e:
            return jsonify({'error': f'Error generating response: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Request must be in JSON format.'}), 400

@app.route('/favicon.ico')
def favicon():
    """
    Serve the favicon.ico file to eliminate 404 errors for favicon requests.
    Ensure that 'favicon.ico' is placed inside the 'static' directory.
    """
    return app.send_static_file('favicon.ico')

if __name__ == '__main__':
    # Determine the port to run the app on
    port = int(os.environ.get('PORT', 5000))
    # Run the Flask app
    app.run(debug=True, port=port)
