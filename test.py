import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load the CSV file for the training data
train_df = pd.read_csv('C:\\Users\\sinha\\Desktop\\VB\\data\\mfccs.csv')

# Normalize the feature values
X_train = train_df.drop('speaker', axis=1)
y_train = train_df['speaker']

X_train_norm = (X_train - X_train.mean()) / X_train.std()

# Train the MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(30, 20, 1), random_state=1, max_iter=300)
clf.fit(X_train_norm, y_train)

# Load the CSV file for the test data
test_df = pd.read_csv('C:\\Users\\sinha\\Desktop\\VB\\data\\mfccs1.csv')

# Normalize the feature values for the test data
X_test = test_df.drop('speaker', axis=1)
X_test_norm = (X_test - X_train.mean()) / X_train.std()

# Make predictions on the test data
y_pred = clf.predict(X_test_norm)

# Evaluate the model on the test data
print("Accuracy:", accuracy_score(test_df['speaker'], y_pred))

# Print authentication results
if np.all(y_pred == test_df['speaker']):
    print("Authenticated")
else:
    print("Denied")