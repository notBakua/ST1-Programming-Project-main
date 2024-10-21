#*******************************
#Author:
#u3253248 u3275890 Assessment 3_Dataset_Analysis 10/ 03/2024
#Programming:
#*******************************

#Step 1: Reading the dataset and printing sample data
import pandas as pd

# Load the dataset
file_path = "clothes_price_prediction_data.csv"
data = pd.read_csv(file_path)

# Remove duplicates
cleaned_data = data.drop_duplicates()

# Display sample data for inspection
print(cleaned_data.sample(5))


#Step 2: Problem Statement definition
# Our goal is to predict the price of clothing based on various attributes.
# Our dependent variable is Price
# Our independent variables are Brand, Catagory, Color, Size, Material


#Step 3: Visualising the distribution of Target variable (Price)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(cleaned_data['Price'], bins=20, edgecolor='black')
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Upon analysis of the of the target variable, it is fairly balanced.


#Step 4: Data exploration at a basic level
# Quantitative: Price
# Catagorical: Brand, Category, Color, Size, Material


#Step 5: Visual Exploratory Data Analysis (EDA) of data (with histogram and barcharts)

# Creating bar plots for categorical variables
categorical_columns = ['Brand', 'Category', 'Color', 'Size', 'Material']

# Plotting bar charts for each categorical column
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle('Distribution of Categorical Variables', fontsize=16)

for i, col in enumerate(categorical_columns):
    ax = axes[i // 2, i % 2]
    cleaned_data[col].value_counts().plot(kind='bar', ax=ax, edgecolor='black')
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Removing any empty subplot
fig.delaxes(axes[2, 1])
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


#Step 6: Outlier analysis
# This dataset contains no significant outliers.


#Step 7: Missing values analysis
# This dataset contains no missing values.


#Step 8: Feature selection - Visual and statistic correlation analysis for selection of best features
# No scatter plot or pearson correlation is needed because the only continuous predictor variable is Price.

# Boxplots for categorical variables against the continuous target variable (Price)
import seaborn as sns

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle('Box Plots: Categorical Variables vs Price', fontsize=16)

for i, col in enumerate(categorical_columns):
    ax = axes[i // 2, i % 2]
    sns.boxplot(x=cleaned_data[col], y=cleaned_data['Price'], ax=ax)
    ax.set_title(f'{col} vs Price')
    ax.set_xlabel(col)
    ax.set_ylabel('Price')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Removing any empty subplot
fig.delaxes(axes[2, 1])
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


#Step 9: Statistical feature selection (categorical vs. continuous) using ANOVA test
import scipy.stats as stats

# Perform ANOVA tests on each categorical variable against the target variable (Price)
anova_results = {}

# Loop through each categorical column to perform ANOVA
for col in categorical_columns:
    groups = [cleaned_data[cleaned_data[col] == level]['Price'] for level in cleaned_data[col].unique()]
    f_stat, p_value = stats.f_oneway(*groups)
    anova_results[col] = {"F-statistic": f_stat, "p-value": p_value}

# Convert the results to a DataFrame for cleaner display
anova_results_df = pd.DataFrame(anova_results).T

# Print the ANOVA results as a table
print("ANOVA Test Results:\n")
print(anova_results_df.to_string(index=True, float_format="%.6f"))

# P-values for each catagorical variable against Price
# Brand: 0.2601
# Category: 0.2258
# Color: 0.4618
# Size: 0.7099
# Material: 0.7366

# Since all p-values are > 0.05, H0 is true.
# This means that individually each catagorical variables doesnt have an impact on the price by its self.


#Step 10: Selecting final predictors/features for building machine learning/AI model

# Final Features:
# Brand: Different brands might influence price through quality and reputation.
# Category: Different clothing types (e.g., Sweater vs. Shoes) are expected to have different price ranges.
# Color: This feature might interact with others (e.g., trending colors could impact prices).
# Size: Sizes might affect pricing (e.g., certain sizes may be priced differently).
# Material: Material quality could impact the price significantly.

# Price: Continuous variable we want to predict.


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
