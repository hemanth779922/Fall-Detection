import csv
import random
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import config
import os

def create_mock_dataset(filename, num_samples=2000):
    """
    Creates a simulated dataset if no real data is available.
    Class 0: Standing
    Class 1: Sitting
    Class 2: Fall
    Class 3: Sleeping (Lying down)
    """
    print(f"Generating mock dataset of {num_samples} samples...")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 5 Features + Label
        writer.writerow(['tilt_angle', 'hw_ratio', 'vertical_hip_pos', 'hip_speed', 'total_movement', 'label'])
        
        # We will split samples evenly across 4 classes
        samples_per_class = num_samples // 4
        
        for _ in range(samples_per_class):
            # Class 0: Standing
            # Small tilt, high ratio, hips high up (low value), slow/no descent, low total movement
            writer.writerow([
                random.uniform(0, 20),           # tilt_angle
                random.uniform(2.5, 4.0),        # hw_ratio
                random.uniform(0.1, 0.45),       # vertical_hip_pos
                random.uniform(-0.02, 0.02),     # hip_speed
                random.uniform(0.0, 0.05),       # total_movement
                0                                # Label: Standing
            ])
            
            # Class 1: Sitting
            # Medium tilt, medium ratio, hips midway, slow/no descent, low total movement
            writer.writerow([
                random.uniform(10, 45),          # tilt_angle
                random.uniform(1.0, 2.5),        # hw_ratio
                random.uniform(0.4, 0.7),        # vertical_hip_pos
                random.uniform(-0.02, 0.02),     # hip_speed
                random.uniform(0.0, 0.05),       # total_movement
                1                                # Label: Sitting
            ])
            
            # Class 2: Fall
            # High tilt, low ratio, hips very low, FAST descent, high total movement during fall
            writer.writerow([
                random.uniform(60, 90),          # tilt_angle
                random.uniform(0.2, 1.0),        # hw_ratio
                random.uniform(0.7, 1.0),        # vertical_hip_pos
                random.uniform(0.15, 0.5),       # hip_speed
                random.uniform(0.1, 0.4),        # total_movement
                2                                # Label: Fall
            ])

            # Class 3: Sleeping (Lying down)
            # High tilt, low ratio, hips very low, NO recent descent, low/med total movement (tossing)
            writer.writerow([
                random.uniform(70, 90),          # tilt_angle
                random.uniform(0.2, 1.0),        # hw_ratio
                random.uniform(0.7, 1.0),        # vertical_hip_pos
                random.uniform(-0.02, 0.02),     # hip_speed
                random.uniform(0.0, 0.15),       # total_movement (some tossing allowed)
                3                                # Label: Sleeping
            ])
            
    print(f"Generated {filename}")

def load_data(filename):
    """
    Loads custom CSV supervised dataset.
    Returns Numpy arrays of Features (X) and Labels (y).
    """
    X = []
    y = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip CSV header
        for row in reader:
            features = [float(val) for val in row[:-1]]
            label = int(row[-1])
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

def train_model():
    """
    Main ML Pipeline training execution.
    """
    print("=== Training Fall Detection ML Model ===")
    
    # Check if dataset exists, if not generate one!
    if not os.path.exists(config.DATASET_PATH):
        print(f"Dataset not found at {config.DATASET_PATH}.")
        create_mock_dataset(config.DATASET_PATH, 1000)
        
    print("1. Loading dataset...")
    X, y = load_data(config.DATASET_PATH)
    print(f"Dataset loaded: {len(X)} records.")

    # 2. Split data into training (80%) and testing sets (20%)
    print("2. Splitting data into training and test datasets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Create the Machine Learning Model
    print("3. Initializing Random Forest Classifier...")
    # RandomForest resolves non-linear dependencies well and prevents overfitting.
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    # 4. Train the model using the .fit methodology
    print("4. Training the Machine Learning model...")
    model.fit(X_train, y_train)
    
    # 5. Model Inference Evaluation / Quality Check
    print("5. Evaluating model performance on validation split...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("="*40)
    print("Detailed Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Standing (0)', 'Sitting (1)', 'Fall (2)', 'Sleeping (3)']))
    
    # 6. Save the trained ML Model to a pickle (.pkl) format object using joblib
    print(f"\n6. Saving the trained model framework to '{config.MODEL_PATH}'...")
    joblib.dump(model, config.MODEL_PATH)
    print("Done! The ML Model is now ready for Real-Time Predictions (`python main.py`).")

if __name__ == "__main__":
    train_model()
