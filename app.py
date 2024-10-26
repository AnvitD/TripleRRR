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
    data = request.get_json()
    state = data.get('state')
    county = data.get('county')
    disaster = data.get('disaster')

    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'State': [state],
        'County': [county],
        'DisasterType': [disaster]
    })

    # Predict risk
    predicted_risk = model.predict(input_data)[0]
    predicted_risk = round(predicted_risk, 2)

    # Return the prediction as JSON for the JavaScript fetch call
    return jsonify({"risk_score": predicted_risk})

if __name__ == '__main__':
    app.run(debug=True)
