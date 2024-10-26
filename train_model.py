import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
data = pd.read_csv('PredictionDataSet.csv')

# Display the first few rows to understand the data
print("First 5 rows of the dataset:")
print(data.head())

# Define the feature columns and target columns
# Assuming the risk assessments are the numerical columns related to disasters
disaster_types = ['Avalanche', 'Coastal Flooding', 'Cold Wave', 'Drought', 'Earthquake',
                 'Hail', 'Heatwave', 'Hurricane', 'Icestorm', 'Landslide', 'Lightning',
                 'Riverine', 'Flooding', 'Strong Wind', 'Tornado', 'Tsunami',
                 'Volcanic Activity', 'Wildfire', 'Winter Weather']

# Melt the dataframe to have one row per disaster type per location
melted = data.melt(id_vars=['State', 'Abbr', 'County'], value_vars=disaster_types,
                   var_name='DisasterType', value_name='Risk')

# Drop rows with missing risk values
melted = melted.dropna(subset=['Risk'])

# Convert Risk to numeric (if not already)
melted['Risk'] = pd.to_numeric(melted['Risk'], errors='coerce')
melted = melted.dropna(subset=['Risk'])

# Features and target
X = melted[['State', 'County', 'DisasterType']]
y = melted['Risk']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: One-Hot Encode categorical variables
categorical_features = ['State', 'County', 'DisasterType']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline that first preprocesses the data and then fits the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
print("Training the model...")
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model Mean Squared Error on Test Set: {mse}")

# Save the trained model to a file
joblib.dump(pipeline, 'risk_model.joblib')
print("Model saved as risk_model.joblib")