import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load clustered data (Reduce dataset size for testing purposes)
logging.info("Loading dataset...")
df = pd.read_csv(os.path.join('data', 'clustered_data.csv'))

# Reduce the dataset size (for testing)
df = df.sample(n=1000, random_state=42)  # Taking a smaller sample for faster processing

# Prepare features and labels
logging.info("Preparing features and labels...")
X = df[['Quantity', 'UnitPrice', 'Hour', 'Weekday']].fillna(0)
y = df['Cluster'].fillna(0).astype(int)

# Train-test split
logging.info("Splitting dataset into train and test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models with reduced complexity for faster training
models = {
    'svm': SVC(C=1.0, kernel='linear'),  # Reduced complexity
    'rf': RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42),  # Reduced complexity
    'knn': KNeighborsClassifier(n_neighbors=3)  # Reduced complexity
}

# Output directories
os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)
os.makedirs('metrics/evaluation_plots', exist_ok=True)

metrics_output = {}

# Train, save each model, and evaluate
for name, model in models.items():
    logging.info(f"Training {name.capitalize()} model...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    logging.info(f"{name.capitalize()} model trained in {time.time() - start_time:.2f} seconds.")
    
    preds = model.predict(X_test)

    # Save model
    logging.info(f"Saving {name} model...")
    joblib.dump(model, os.path.join('models', f'{name}_model.pkl'))

    # Save metrics
    report = classification_report(y_test, preds, output_dict=True)
    metrics_output[name] = report

    # Plot and save accuracy for each model
    logging.info(f"Plotting accuracy for {name} model...")
    plt.figure(figsize=(6,6))

    # Ensure the accuracy is passed as a 2D array for sns.heatmap
    accuracy_value = [[report['accuracy']]]  # Convert to 2D array

    sns.heatmap(accuracy_value, annot=True, cmap='Blues', fmt='.2f', cbar=False)
    plt.title(f'{name.capitalize()} Model Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join('metrics/evaluation_plots', f'{name}_accuracy.png'))
    plt.show()

# Save metrics to JSON
logging.info("Saving metrics to JSON file...")
with open(os.path.join('metrics', 'classification_report.json'), 'w') as f:
    json.dump(metrics_output, f, indent=4)

logging.info("[✓] Models and metrics saved.")
