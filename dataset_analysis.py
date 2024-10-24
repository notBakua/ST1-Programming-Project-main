#*******************************
#Author:
#u3253248 u3275890 Assessment 3_Dataset_Analysis 20/ 10/2024
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