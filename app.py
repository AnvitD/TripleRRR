from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('risk_model.joblib')

# Load unique states, counties, and disaster types for dropdowns
data = pd.read_csv('PredictionDataSet.csv')
disaster_types = ['Avalanche', 'Coastal Flooding', 'Cold Wave', 'Drought', 'Earthquake',
                  'Hail', 'Heatwave', 'Hurricane', 'Icestorm', 'Landslide', 'Lightning',
                  'Riverine', 'Flooding', 'Strong Wind', 'Tornado', 'Tsunami',
                  'Volcanic Activity', 'Wildfire', 'Winter Weather']

states = sorted(data['State'].unique())
counties = sorted(data['County'].unique())

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', states=states, counties=counties,
                           disaster_types=disaster_types)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle AJAX POST requests to calculate the base risk score.
    Expects JSON data with 'state', 'county', and 'disaster'.
    Returns JSON response with 'risk_score'.
    """
    if request.is_json:
        data = request.get_json()
        state = data.get('state')
        county = data.get('county')
        disaster = data.get('disaster')

        # Validate input
        if not all([state, county, disaster]):
            return jsonify({'error': 'Missing data: state, county, and disaster are required.'}), 400

        # Optional: Validate if the state and county exist in the dataset
        if state not in states:
            return jsonify({'error': f"Invalid state: '{state}'. Please enter a valid state."}), 400
        if county not in counties:
            return jsonify({'error': f"Invalid county: '{county}'. Please enter a valid county."}), 400

        # Create a DataFrame for prediction
        input_data = pd.DataFrame({
            'State': [state],
            'County': [county],
            'DisasterType': [disaster]
            # 'Year' is not used in the model; it's for post-prediction adjustment
        })

        # Predict base risk
        try:
            base_risk = model.predict(input_data)[0]
            base_risk = round(base_risk, 2)
        except Exception as e:
            return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

        return jsonify({'risk_score': base_risk})
    else:
        return jsonify({'error': 'Request must be in JSON format.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
