#*******************************
#Author:
#u3253248 u3275890 Assessment 3_machine_learning_testing 20/ 10/2024
#Programming:
#*******************************

import pandas as pd

file_path = "clothes_price_prediction_data.csv"
data = pd.read_csv(file_path)

cleaned_data = data.drop_duplicates()

categorical_columns = ['Brand', 'Category', 'Color', 'Size', 'Material']

#Step 11: Data conversion to numeric values for machine learning/predictive analysis
# Converting categorical variables to numeric using get_dummies()
encoded_data = pd.get_dummies(cleaned_data, columns=categorical_columns, drop_first=True)

# Displaying a sample of the encoded dataset to verify the conversion
encoded_data_sample = encoded_data.sample(5)
print(encoded_data_sample)

# Each category is displayed as a separate column and uses binary to indicate if it belongs in that category (True/1 if it does, False/0 if it doesnt)


#Step 12: Train/test data split and standardisation/normalisation of data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Splitting the data into features (X) and target (y)
X = encoded_data.drop('Price', axis=1)
y = encoded_data['Price']

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying standardisation using StandardScaler (optional, but useful for KNN or Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Displaying the shape of the training and testing datasets to verify the split
data_split_info = {
    "X_train shape": X_train.shape,
    "X_test shape": X_test.shape,
    "y_train shape": y_train.shape,
    "y_test shape": y_test.shape
}
# Converting the data split info to a DataFrame with better formatting
data_split_df = pd.DataFrame(list(data_split_info.items()), columns=['Data Type', 'Shape'])

# Formatting the print output as a table
print("Data Split Information:\n")
print(data_split_df.to_string(index=False))


#Step 13: Investigating multiple regression algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Initializing the models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Support Vector Regressor": SVR()
}

# Dictionary to store the results
model_results = {}

# Training and evaluating each model
for name, model in models.items():
    # Training the model on the scaled training data
    model.fit(X_train_scaled, y_train)
    
    # Making predictions on the test data
    y_pred = model.predict(X_test_scaled)
    
    # Calculating performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Storing the results
    model_results[name] = {"MSE": mse, "R2 Score": r2}

# Converting the model results to a DataFrame for cleaner printing
model_results_df = pd.DataFrame(model_results).T

# Formatting the print output as a table
print("Model Performance Summary:\n")
print(model_results_df.to_string(index=True, float_format="%.6f"))


#Step 14: Selection of the best model
# Based on previous trials, Linear Regression is the best performing model.


#Step 15: Deployment of the best model in production
import joblib
import numpy as np

# Retrain the best model on 100% of the available data
# Prepare the full dataset
X_full_scaled = scaler.fit_transform(X) 
best_model_final = LinearRegression()
best_model_final.fit(X_full_scaled, y)

# Save the model, scaler, and column names as serialized files
joblib.dump(best_model_final, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns, 'Xcolumns.pkl')

# Define a function for predictions (to integrate with GUI/website)
def predict_price(inputs):
    try:
        # Load the saved model, scaler, and column names
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        X_columns = joblib.load('Xcolumns.pkl')

        # Initialize a DataFrame with all features set to 0
        input_data = pd.DataFrame(np.zeros((1, len(X_columns))), columns=X_columns)

        # Update relevant features based on input
        for feature, value in inputs.items():
            if feature in input_data.columns:
                input_data.at[0, feature] = value

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        return prediction

    except Exception as e:
        return f"Error: {str(e)}"

# Test the prediction function with new unseen data
test_input = {
    'Brand_Nike': 1, 'Category_Shoes': 1, 'Color_Red': 1,
    'Size_M': 1, 'Material_Cotton': 1
}

predicted_price = predict_price(test_input)
print(f"Predicted Price for Test Input: ${predicted_price:.2f}")
