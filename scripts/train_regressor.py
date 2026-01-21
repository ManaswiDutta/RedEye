import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

EMB_PATH = "data/embeddings/embeddings.npy"
LABELS_PATH = "data/embeddings/labels.npy"
MODEL_OUT = "models/regressor.pkl"

os.makedirs("models", exist_ok=True)

def main():
    X = np.load(EMB_PATH)
    y = np.load(LABELS_PATH)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # LightGBM regression model
    model = lgb.LGBMRegressor(
        n_estimators=50,
        num_leaves=8,
        max_depth=3,
        learning_rate=0.1,
        min_data_in_leaf=5,
        random_state=42
    )


    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("Sample predictions:", preds[:10])
    print("Sample ground truth:", y_test[:10])

    # Save model
    joblib.dump(model, MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
