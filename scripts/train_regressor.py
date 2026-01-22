import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import os

EMB_PATH = "data/embeddings/embeddings.npy"
LABELS_PATH = "data/embeddings/labels.npy"
MODEL_OUT = "models/regressor.pkl"

os.makedirs("models", exist_ok=True)

def main():
    X = np.load(EMB_PATH)
    y = np.load(LABELS_PATH) # (N, 3) -> [ups, downs, score]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # LightGBM regression model wrapped in MultiOutputRegressor
    base_model = lgb.LGBMRegressor(
        n_estimators=50,
        num_leaves=8,
        max_depth=3,
        learning_rate=0.1,
        min_data_in_leaf=5,
        random_state=42,
        verbosity=-1
    )
    
    model = MultiOutputRegressor(base_model)

    print("Training model for [ups, downs, score]...")
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    
    targets = ["ups", "downs", "score"]
    for i, name in enumerate(targets):
        mae = mean_absolute_error(y_test[:, i], preds[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], preds[:, i]))
        print(f"\nTarget: {name}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")

    print("\nSample predictions [ups, downs, score]:\n", preds[:5])
    print("Sample ground truth [ups, downs, score]:\n", y_test[:5])

    # Save model
    joblib.dump(model, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
