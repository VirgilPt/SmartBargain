import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import joblib

class VintedDealPredictor:
    def __init__(self, data):
        self.df = pd.DataFrame(data)
        
    def preprocess_data(self):
        # Handle missing values more robustly
        for col in ['UserRating', 'ItemStatus', 'NbEvalUser', 'NbFavourite']:
            self.df[col].fillna(self.df[col].median(), inplace=True)
        
                
        # Prepare features and target
        features_cols = ['Brand', 'Type', 'Price', 'NbFavourite', 
                         'UserRating', 'NbEvalUser', 'ItemStatus']
        X = self.df[features_cols]
        y = self.df['bonne_affaire']
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        #Save the scaler
        joblib.dump(self.scaler, 'scaler.pkl')
        
        return X_scaled, y
    
    def create_model(self, input_dim):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005), 
            loss='binary_crossentropy', 
            metrics=['accuracy', 
                     tf.keras.metrics.Precision(), 
                     tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train_model(self):
        # Preprocess data
        X, y = self.preprocess_data()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Compute class weights to handle imbalance
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # Create and train the model
        model = self.create_model(X.shape[1])
        
        # Early stopping with model checkpoint
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            restore_best_weights=True,
            min_delta=0.001
        )
        
        # Fit the model with class weights
        history = model.fit(
            X_train, y_train, 
            validation_split=0.2,
            epochs=500,
            batch_size=32,
            class_weight=class_weight_dict,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate the model with additional metrics
        results = model.evaluate(X_test, y_test, verbose=1)
        print("\nModel Evaluation:")
        print(f"Test Loss: {results[0]:.4f}")
        print(f"Test Accuracy: {results[1]:.4f}")
        print(f"Precision: {results[2]:.4f}")
        print(f"Recall: {results[3]:.4f}")
        
        # Visualize training history
        self.plot_training_history(history)
        
        return model
    
    def plot_training_history(self, history):
        """
        Plot training and validation metrics
        """
        plt.figure(figsize=(15, 5))
        
        # Loss subplot
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy subplot
        plt.subplot(1, 3, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Precision-Recall subplot
        plt.subplot(1, 3, 3)
        plt.plot(history.history['precision'], label='Training Precision')
        plt.plot(history.history['val_precision'], label='Validation Precision')
        plt.title('Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict_deal(self, new_item_data):
        # Prepare new item data
        new_item_df = pd.DataFrame([new_item_data])
        
        # Select and prepare features
        features_cols = ['Brand', 'Type', 'Price', 'NbFavourite', 
                 'UserRating', 'NbEvalUser', 'ItemStatus']
        X_new = new_item_df[features_cols]
        
        # Scale features
        self.scaler = joblib.load('scaler.pkl')
        X_new_scaled = self.scaler.transform(X_new)
        
        # Predict
        prediction = self.model.predict(X_new_scaled)
        
        return prediction[0][0]
    
    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def load_encoders(self, encoder_paths):
        self.label_encoders = {}
        for col, path in encoder_paths.items():
            le = joblib.load(path)
            self.label_encoders[col] = le

        # Update encoders to handle unknown values
        for col, le in self.label_encoders.items():
            le_classes = set(le.classes_)
            def safe_transform(value):
                if value not in le_classes:
                    le_classes.add(value)
                    le.classes_ = np.array(sorted(le_classes))
                return le.transform([value])[0]
            le.safe_transform = safe_transform


if __name__ == "__main__":
    # Example usage
    file_path = 'data/merged_labelise.csv'
    df = pd.read_csv(file_path)

    # Create and train the model
    predictor = VintedDealPredictor(df)
    predictor.model = predictor.train_model()

    # Save the model
    predictor.model.save('deal_predictor_model.keras')