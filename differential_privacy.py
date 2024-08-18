import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from google.colab import files

# Define the filename
filename = 'asthma_disease_dataset.csv'

# Check if the file exists in the directory, if not prompt the user to upload
if not os.path.isfile(filename):
    print(f"{filename} not found. Please upload the preprocessed asthma disease dataset CSV file.")
    uploaded = files.upload()
    filename = next(iter(uploaded))  # Get the filename from the uploaded files

# Load the dataset
asthma_disease_data = pd.read_csv(filename)

# Define the differential privacy parameters
epsilon = 1.0  # Privacy budget parameter (increased noise)
sensitivity = 1.0  # Sensitivity of the query function

# Function to add Laplace noise
def add_laplace_noise(value, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, 1)[0]
    return value + noise

# Apply differential privacy to the selected numerical columns
columns_to_add_dp = [
    'Age', 'BMI', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
    'PollutionExposure', 'PollenExposure', 'DustExposure', 'LungFunctionFEV1', 'LungFunctionFVC'
]

for column in columns_to_add_dp:
    if column in asthma_disease_data.columns:
        asthma_disease_data[column + '_noisy'] = asthma_disease_data[column].apply(
            lambda x: add_laplace_noise(x, sensitivity, epsilon) if pd.notnull(x) else x
        )

# Save the dp dataset with differential privacy
dp_filename = 'asthma_disease_dataset_dp.csv'
asthma_disease_data.to_csv(dp_filename, index=False)
print(f"Dataset with differential privacy saved as '{dp_filename}'")

# If there's no target column, simply work with features
X_original = asthma_disease_data[columns_to_add_dp]
X_dp = asthma_disease_data[[col + '_noisy' for col in columns_to_add_dp]]

# Calculate utility metrics to compare original and DP data
mae_values = {}
rmse_values = {}
r2_values = {}

for column in columns_to_add_dp:
    mae = mean_absolute_error(X_original[column], X_dp[column + '_noisy'])
    rmse = np.sqrt(mean_squared_error(X_original[column], X_dp[column + '_noisy']))
    r2 = r2_score(X_original[column], X_dp[column + '_noisy'])
    
    mae_values[column] = mae
    rmse_values[column] = rmse
    r2_values[column] = r2

print("\nUtility Metrics:")
print("Mean Absolute Error (MAE):", mae_values)
print("Root Mean Squared Error (RMSE):", rmse_values)
print("R-squared (R²):", r2_values)

# 1. Plot density comparisons for key features
for column in columns_to_add_dp:
    plt.figure(figsize=(12, 6))
    sns.kdeplot(X_original[column], label=f'Original {column}', color='blue')
    sns.kdeplot(X_dp[column+'_noisy'], label=f'DP {column}', color='orange')
    plt.title(f'Density Plot Comparison - {column}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 2. Bar Plot of MAE and RMSE - Side by Side
features = list(mae_values.keys())
x = np.arange(len(features))  # the label locations
width = 0.35  # the width of the bars

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, mae_values.values(), width, label='MAE', color='blue')
plt.bar(x + width/2, rmse_values.values(), width, label='RMSE', color='orange')
plt.xlabel('Features')
plt.ylabel('Error Value')
plt.title('Comparison of MAE and RMSE between Original and DP Datasets')
plt.xticks(x, features, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Line Plot of R-squared (R²) Values
plt.figure(figsize=(12, 6))
plt.plot(features, r2_values.values(), marker='o', color='green', label='R²')
plt.xlabel('Features')
plt.ylabel('R² Value')
plt.title('R² Comparison between Original and DP Datasets')
plt.xticks(rotation=45)
plt.axhline(y=1, color='gray', linestyle='--', label='Perfect Preservation')
plt.legend()
plt.tight_layout()
plt.show()
