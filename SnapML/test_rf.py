import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load iris dataset
iris_data = sns.load_dataset("iris")
print("Iris dataset loaded successfully")
print(f"Dataset shape: {iris_data.shape}")

# Split the data
X = iris_data.drop(columns=['species'])
y = iris_data['species']
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.1, random_state=42)

# Clean data - drop NA values (same as your dropNa function)
def dropNa(train_X, test_X, train_Y, test_Y):
    train_X = train_X.dropna()
    train_Y = train_Y.loc[train_X.index.values.tolist()]
    test_X = test_X.dropna()
    test_Y = test_Y.loc[test_X.index.values.tolist()]
    return train_X, test_X, train_Y, test_Y

train_X, test_X, train_Y, test_Y = dropNa(train_X, test_X, train_Y, test_Y)

# Train a random forest model (equivalent to create_model('rf') in PyCaret)
RandomForest_ML = RandomForestClassifier(n_estimators=100, random_state=42)
RandomForest_ML.fit(train_X, train_Y)

# Make predictions (equivalent to predict_model() in PyCaret)
predictions = RandomForest_ML.predict(test_X)
proba = RandomForest_ML.predict_proba(test_X)

# Create output dataframe similar to PyCaret's predict_model output
output = test_X.copy()
output['prediction_label'] = predictions
output['prediction_score'] = np.max(proba, axis=1)
output['target'] = test_Y.values

# Display results
print("\nModel trained successfully")
print(f"Accuracy: {accuracy_score(test_Y, predictions):.4f}")
print("\nClassification Report:")
print(classification_report(test_Y, predictions))

print("\nSample predictions (first 5 rows):")
print(output.head()) 