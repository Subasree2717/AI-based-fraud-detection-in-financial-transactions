import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load historical transaction data
# Replace 'data.csv' with your dataset path
data = pd.read_csv('data.csv')

# Example columns: 'amount', 'location', 'merchant', 'user_id', 'is_fraud'
# Modify based on your dataset

# Feature columns and target variable
features = ['amount', 'location', 'merchant', 'user_id']
target = 'is_fraud'

X = data[features]
y = data[target]

# Preprocessing: Convert categorical data to numerical if needed
X = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'fraud_detection_model.pkl')

# Function to detect fraudulent transactions
def detect_fraud(transaction_data):
    # Load the model
    model = joblib.load('fraud_detection_model.pkl')
    
    # Convert incoming transaction data to DataFrame
    transaction_df = pd.DataFrame([transaction_data], columns=features)
    transaction_df = pd.get_dummies(transaction_df)

    # Predict fraud
    is_fraud = model.predict(transaction_df)[0]
    
    return is_fraud

# Example usage
transaction = {
    'amount': 250,
    'location': 'New York',
    'merchant': 'Electronics Store',
    'user_id': 12345
}

is_fraud = detect_fraud(transaction)
print('Fraudulent Transaction' if is_fraud else 'Legitimate Transaction')
