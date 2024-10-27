Triple R Assessment

Project Overview

The Triple R Assessment project is designed to assess, prepare, and respond to potential crises or natural disasters. Using predictive models and data visualization, this project provides users with:

	•	Base Risk Score Calculation for specific locations and disaster types.
	•	Predictive Risk Score Analysis for future risk projections.
	•	Response Guidance with nearby resources like gas stations, stores, and evacuation cities.
	•	Recovery Guidance, including recovery-related YouTube video resources.

This project uses a combination of Python and JavaScript, leveraging Flask for the backend and Leaflet for interactive maps.

Project Structure

	•	app.py: Main backend server file running the Flask application.
	•	train_model.py: Python script to train the predictive model.
	•	PredictionDataSet.csv: Dataset for training and evaluating the predictive model.
	•	index.html: Frontend HTML template with embedded JavaScript for map-based visualization.
	•	requirements.txt: Lists required Python packages.

Features

	1.	Base Risk Score Calculation: Calculates risk based on disaster type, location, and existing data.
	2.	Predictive Analysis: Projects risk over a specified future period.
	3.	Interactive Maps: Displays essential resources and evacuation cities using Leaflet and Marker Clusters.
	4.	Guidance Tabs: Provides actionable response and recovery guidance based on location and disaster type.

Installation and Setup

Prerequisites

Ensure you have the following installed:

	•	Python 3.8+
	•	pip (Python package manager)
	•	Node.js and npm (for JavaScript package dependencies if you need additional customization)

Step-by-Step Setup

	1.	Clone the Repository:
 ```
 git clone <https://github.com/AnvitD/TripleRRR.git>
 cd TripleRRR
 ```

2.	Install Python Dependencies:
Install the necessary Python packages listed in requirements.txt:

```
pip install -r requirements.txt
```

3.	Run the Flask Application:
Start the application server:

```
python app.py
```
4.	Open the Application:
Open your browser and go to http://127.0.0.1:5000 to view the Triple R Assessment web interface.

Dependencies

Backend (Python)

The backend uses the following Python libraries, specified in requirements.txt:

	•	Flask: For building the web server and handling HTTP requests.
	•	pandas: For data manipulation.
	•	numpy: For numerical computations.
	•	scikit-learn: For model training and predictive analysis.
	•	joblib: For saving/loading trained machine learning models.

Frontend (JavaScript and HTML)

The frontend relies on:

	•	Leaflet.js: For rendering interactive maps.
	•	Leaflet.MarkerCluster: For clustering map markers.
	•	Chart.js: For displaying predictive risk scores as line charts.
	•	FontAwesome: For icons used in the interface.
	•	Google Fonts: Specifically, the Roboto font for a clean and modern design.

Additional API Integration

The project utilizes APIs for map location data:

	•	Nominatim API: For obtaining latitude and longitude of user-selected locations.
	•	Overpass API: For querying nearby resources (e.g., gas stations, stores).

Usage Guide

	•	Navigate through the tabs to explore different functionalities: Recognition, Response, and Recovery.
	•	Input Location and Disaster Type in the Recognition tab to calculate the base risk score.
	•	Use Predictive Risk Slider to project risk scores for future periods.
	•	Map Resources in the Response tab show nearby resources.
	•	Search within the Response tab for evacuation cities using the search bar.
	•	Recovery Videos provide context and guidance for disaster recovery in the Recovery tab.

Future Improvements

	•	Extend predictive modeling capabilities with additional disaster types.
	•	Enhance the Response tab with real-time data from relevant government resources.
	•	Add additional customization for map display and markers.




 
