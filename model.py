import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
def load_data():
    df = pd.read_csv('diabetes.csv')
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y

# Train and save the model
def train_model():
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Model accuracy: {accuracy_score(y_test, y_pred)}")
    
    # Save model
    joblib.dump(model, 'diabetes_model.joblib')
    print("Model saved as diabetes_model.joblib")

if __name__ == '__main__':
    train_model()