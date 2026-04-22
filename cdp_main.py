# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# import datacheck function
import Datacheck as dc

# import models from libraries
from sklearn.neural_network import MLPClassifier        #mlp
from xgboost import XGBClassifier                       #xgboost
from lightgbm import LGBMClassifier                     #lightgbm
from sklearn.neighbors import KNeighborsClassifier      #knn
from sklearn.linear_model import LogisticRegression     #logistic regression
from sklearn.ensemble import RandomForestClassifier     #random forest

# import metric and evaluation functions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# import convergence warning ignore method
import warnings
from sklearn.exceptions import ConvergenceWarning

# --- Load dataset ---
file_path = 'heart.csv'
data = pd.read_csv(file_path)

# --- Inspect dataset ---
dc.datacheck(data)

# --- Preprocessing ---
# Identify numeric and feature groups by heading name
numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'OLDpeak']
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Use binary encoding for features with only 2 possible options
if 'Sex' in data.columns:
    data['Sex'] = data['Sex'].map({'M':1, 'F':0})
if 'ExerciseAngina' in data.columns:
    data['ExerciseAngina'] = data['ExerciseAngina'].map({'Y':1, 'N':0})

# One-hot encoding for categorical features
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Clean column names
data.columns = data.columns.str.strip()

# Separate features from labels
features = data.drop('HeartDisease', axis=1)
targets = data['HeartDisease']

# Split data
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=5)

# Standardization
features_to_scale = [col for col in numeric_features if col in x_train.columns]

x_train[features_to_scale] = StandardScaler().fit_transform(x_train[features_to_scale])
x_test[features_to_scale] = StandardScaler().fit_transform(x_test[features_to_scale])


print("\n --- Preprocessing Completed ---")
print("Data split:")
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape:  {x_test.shape}")
print("-" * 40)

# --- Model Configuration and Storing---
# Note: convergence warning is being ignored for MLPClassifier to make final output look cleaner.
warnings.filterwarnings("ignore", category=ConvergenceWarning) # Note: comment out filterwarnings for model testing and tuning
random_state_seed = 25
models_dict = {
    "MLP Network" : MLPClassifier(
        hidden_layer_sizes=(16, 4),
        activation="relu",
        solver="adam",
        random_state=random_state_seed,
        alpha=0.0001,
        max_iter=200,
        learning_rate_init=0.0005
    ),

    "LightGBM" : LGBMClassifier(
        n_estimators=55,
        learning_rate=0.1,
        max_depth=3,
        num_leaves=8,
        random_state=random_state_seed,
        min_child_samples=3,
        colsample_bytree=0.8,
        force_row_wise=True,
        max_bin=90,
        verbose=-1
    ),

    "Random Forest" : RandomForestClassifier(
        n_estimators=14,
        max_depth=3,
        min_samples_split=32,
        random_state=random_state_seed,
        n_jobs=-1
    ),

    "Logistic Regression" : LogisticRegression(
        max_iter=1000,
        random_state=random_state_seed
    ),

    "KNN" : KNeighborsClassifier(
        n_neighbors=7,
        p=1  
    ),
    
    "XGBoost" : XGBClassifier(
        learning_rate=0.06,
        max_depth=3,
        random_state=random_state_seed,
        eval_metric="logloss",
        subsample=0.2,
        max_bin=225
    )
}


# --- Model Training and Evaluation ---
print("\n --- Model Training and Evaluation ---")
model_results = []

for model_name, model in models_dict.items():
    start_time = time.time()
    print(f"Now Training: {model_name}...", end="   ")

    # Train model(s)
    model.fit(x_train, y_train)
    model_training_time = time.time() - start_time
    print(f"Training Time: {model_training_time:.2f} seconds")

    # Evaluate model(s)
    y_pred = model.predict(x_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)       # measures model's ability to find all positive samples
    score = f1_score(y_test, y_pred)            # measures the percisiona and recall. 1 = best, 0 = worst

    # Store results
    model_results.append({
        "Model" : model_name,
        "Accuracy" : accuracy,
        "Precision" : precision,
        "Recall" : recall,
        "F1 Score" : score,
        "Training Time" : model_training_time
    })
    
    # Print results for current model
    print(f"\n --- {model_name} Results ---")
    print(f"   - Accuracy:              {accuracy*100:.4f}%")
    print(f"   - Precision:             {precision*100:.4f}%")
    print(f"   - Sensetivity (Recall):  {recall*100:.4f}%")

    print("\n" + "-" * 40 + "\n")


# --- Model Performance Visualization ---

# Graph accuracy, precision, and recall for each model
x = np.arange(len(model_results))
width = 0.25
multiplier = 0

fig, ax = plt.subplots(layout='constrained', figsize=(15, 6))
for attribute in ["Accuracy", "Precision", "Recall"]:
    values = [result[attribute] * 100 for result in model_results]
    rects = ax.bar(x + width * multiplier, values, width, label=attribute)
    ax.bar_label(rects, padding=5, fmt='{:.1f}%', fontweight='bold')
    multiplier += 1
ax.set_ylabel('Scores (%)')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + width, [result["Model"] for result in model_results])
ax.legend(loc='upper right', ncols=3)
ax.set_ylim(0, 110)
ax.grid(axis='y', linestyle='--', alpha=0.9)


# Graph training time for each model
x = np.arange(len(model_results))
training_times = [result["Training Time"] for result in model_results]
fig, ax = plt.subplots(layout='constrained', figsize=(15, 6))
rects = ax.bar(x, training_times, width=0.5, color='skyblue')
ax.bar_label(rects, padding=5, fmt='{:.3f}s', fontweight='bold')
ax.set_ylabel('Training Time (seconds)')
ax.set_title('Model Training Time Comparison')
ax.set_xticks(x, [result["Model"] for result in model_results])
ax.grid(axis='y', linestyle='--', alpha=0.9)
plt.show()

