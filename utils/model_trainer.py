import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        self.trained_models = {}
        self.model_metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None
        
    def prepare_data(self, df):
        """Prepare data for training"""
        df_encoded = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['Brand', 'Processor', 'GPU']
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
        
        # Features and target
        X = df_encoded.drop(['Price_USD'], axis=1)
        y = df_encoded['Price_USD']
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def train_models(self, df):
        """Train all models and evaluate performance"""
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'R2 Score': r2_score(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred)
            }
            
            self.trained_models[name] = model
            self.model_metrics[name] = metrics
        
        # Identify best model
        self.best_model_name = max(self.model_metrics.keys(), 
                                 key=lambda k: self.model_metrics[k]['R2 Score'])
        self.best_model = self.trained_models[self.best_model_name]
        
        # Save models
        self.save_models()
        
        return self.trained_models, self.model_metrics, X_test, y_test
    
    def save_models(self):
        """Save the best model and feature columns"""
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Save best model
        joblib.dump(self.best_model, os.path.join(models_dir, 'best_model.pkl'))
        joblib.dump(self.feature_columns, os.path.join(models_dir, 'feature_columns.pkl'))
        
        # Save all models and metrics
        joblib.dump(self.trained_models, os.path.join(models_dir, 'all_models.pkl'))
        joblib.dump(self.model_metrics, os.path.join(models_dir, 'model_metrics.pkl'))
    
    def load_models(self):
        """Load saved models"""
        models_dir = 'models'
        try:
            self.best_model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
            self.feature_columns = joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
            self.trained_models = joblib.load(os.path.join(models_dir, 'all_models.pkl'))
            self.model_metrics = joblib.load(os.path.join(models_dir, 'model_metrics.pkl'))
            self.best_model_name = max(self.model_metrics.keys(), 
                                     key=lambda k: self.model_metrics[k]['R2 Score'])
            return True
        except FileNotFoundError:
            return False
    
    def predict(self, input_data):
        """Make prediction using the best model"""
        if self.best_model is None:
            raise ValueError("No trained model available. Please train models first.")
        
        # Create dataframe and encode
        input_df = pd.DataFrame([input_data])
        categorical_cols = ['Brand', 'Processor', 'GPU']
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
        
        # Ensure all columns are present
        for col in self.feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reorder columns to match training data
        input_encoded = input_encoded[self.feature_columns]
        
        return self.best_model.predict(input_encoded)[0]

# Global functions for easy importing in Streamlit pages
def train_and_evaluate_model(df, model_type="Random Forest", test_size=0.2):
    """
    Train and evaluate a specific model type.
    
    Args:
        df: Input dataframe
        model_type: Type of model to train ("Linear Regression", "Random Forest", "XGBoost")
        test_size: Fraction of data to use for testing
    
    Returns:
        metrics: Dictionary with model performance metrics
        X_test: Test features
        y_test: Test targets  
        trained_model: The trained model object
    """
    # Prepare data
    df_encoded = df.copy()
    categorical_cols = ['Brand', 'Processor', 'GPU']
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
    
    X = df_encoded.drop(['Price_USD'], axis=1)
    y = df_encoded['Price_USD']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Select and train model
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=100, random_state=42)
        except ImportError:
            # Fallback to Random Forest if XGBoost not available
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model_type = "Random Forest (XGBoost not available)"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred)
    }
    
    # Save model and feature columns
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, 'best_model.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(models_dir, 'feature_columns.pkl'))
    
    return metrics, X_test, y_test, model

def get_feature_importance(model, feature_columns, model_type):
    """
    Get feature importance from a trained model.
    
    Args:
        model: Trained model object
        feature_columns: List of feature column names
        model_type: Type of model
    
    Returns:
        DataFrame with feature importance or None if not available
    """
    if hasattr(model, 'feature_importances_'):
        # For tree-based models like Random Forest
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        return importance_df
    elif hasattr(model, 'coef_'):
        # For linear models
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': np.abs(model.coef_)
        }).sort_values('Importance', ascending=False)
        return importance_df
    else:
        return None

def load_model_and_feature_columns(model_path=None, feature_columns_path=None):
    """
    Load saved model and feature columns.
    
    Args:
        model_path: Path to saved model file
        feature_columns_path: Path to saved feature columns file
    
    Returns:
        model: Loaded model object
        feature_columns: List of feature column names
    """
    if model_path is None:
        model_path = os.path.join('models', 'best_model.pkl')
    if feature_columns_path is None:
        feature_columns_path = os.path.join('models', 'feature_columns.pkl')
    
    model = joblib.load(model_path)
    feature_columns = joblib.load(feature_columns_path)
    
    return model, feature_columns

def predict_price(model, feature_columns, input_features):
    """
    Predict laptop price using a trained model.
    
    Args:
        model: Trained model object
        feature_columns: List of feature column names expected by the model
        input_features: Dictionary with input feature values
    
    Returns:
        predicted_price: Predicted price as float
    """
    # Create dataframe from input
    input_df = pd.DataFrame([input_features])
    
    # Encode categorical variables
    categorical_cols = ['Brand', 'Processor', 'GPU']
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    # Ensure all required columns are present
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded[feature_columns]
    
    # Make prediction
    prediction = model.predict(input_encoded)[0]
    
    return prediction
