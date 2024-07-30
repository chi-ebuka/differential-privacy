import pandas as pd
import numpy as np
import os
from google.colab import files
import matplotlib.pyplot as plt

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

# Function to plot histograms side by side and combined
def plot_histograms(original, noisy, column_name):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    axes[0].hist(original, bins=20, alpha=0.7, label=f'Original {column_name}', color='blue')
    axes[0].set_title(f'Original {column_name}')
    axes[0].set_xlabel(column_name)
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(noisy, bins=20, alpha=0.7, label=f'Noisy {column_name}', color='orange')
    axes[1].set_title(f'Noisy {column_name}')
    axes[1].set_xlabel(column_name)
    axes[1].set_ylabel('Frequency')
    
    plt.suptitle(f'Comparison of Original and Noisy {column_name} Histograms (Side by Side)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    # Save the figure
    fig.savefig(f'{column_name}_histogram_side_by_side.png')
    
    # Combined histogram
    plt.figure(figsize=(10, 5))
    plt.hist(original, bins=20, alpha=0.5, label=f'Original {column_name}', color='blue')
    plt.hist(noisy, bins=20, alpha=0.5, label=f'Noisy {column_name}', color='orange')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title(f'Combined Histogram of Original and Noisy {column_name}')
    plt.show()
    # Save the figure
    plt.savefig(f'{column_name}_histogram_combined.png')

# Plot histograms for all columns with differential privacy
for column in columns_to_add_dp:
    plot_histograms(asthma_disease_data[column], asthma_disease_data[column + '_noisy'], column)

# List all files in the current directory
files = os.listdir('.')
print("Files in the current directory:")
for file in files:
    print(file)

# Check if the dp file exists and provide download link
if os.path.isfile(dp_filename):
    print(f"{dp_filename} found.")
    # Download the file
    files.download(dp_filename)
else:
    print(f"{dp_filename} not found.")
