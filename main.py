import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from scipy import stats


# Load the dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('Fraud.csv')

# Preprocess the dataset
data.dropna(inplace=True)
z_scores = np.abs(stats.zscore(data[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]))
data = data[(z_scores < 3).all(axis=1)]

# Prepare features and target variable
X = data[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']].values
y = data['isFraud'].valuesg
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create and train the Isolation Forest model
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_train_scaled)

def predict_fraud(transaction):
    scaled_transaction = scaler.transform([transaction])
    is_fraud = model.predict(scaled_transaction)
    if is_fraud == -1:
        return "Fraud"
    else:
        return "Not Fraud"

# Example transaction input (replace with user input)
example_transaction = [9839.64, 170136, 160296.36, 0, 0]  # Modify values as needed

result = predict_fraud(example_transaction)
print("Transaction is:", result)
