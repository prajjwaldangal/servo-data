import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score
from catboost import CatBoostRegressor


def save_model(model, path):
    """Save a model to disk.

    Tries CatBoost's native save first; falls back to joblib for other models.
    """
    try:
        # CatBoost models have save_model
        model.save_model(path)
        print(f"Model saved with CatBoost.save_model -> {path}")
    except Exception:
        # generic fallback
        joblib.dump(model, path)
        print(f"Model saved with joblib.dump -> {path}")


def compute_f1(y_true, y_pred, threshold=None, average='binary'):
    """Compute and print an F1 score when classification is sensible.

    Behavior:
    - If `threshold` is provided: binarize both true and predicted values using it.
    - Else, if y_true has <= 10 unique integer-ish values, round predictions to nearest int
      and compute multiclass F1 (uses `average='macro'` for >2 classes).
    - Otherwise, skip and print that F1 is not applicable for continuous targets.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # If a threshold is provided, binarize
    if threshold is not None:
        y_true_bin = (y_true >= threshold).astype(int)
        y_pred_bin = (y_pred >= threshold).astype(int)
        score = f1_score(y_true_bin, y_pred_bin, average=average)
        print(f"F1 (threshold={threshold}, average={average}): {score:.4f}")
        return score

    # Check if the target looks categorical (few unique values)
    unique_vals = np.unique(y_true[~np.isnan(y_true)])
    if unique_vals.size <= 10 and np.all(np.mod(unique_vals, 1) == 0):
        # treat as multiclass by rounding predictions
        y_pred_round = np.rint(y_pred).astype(int)
        n_classes = unique_vals.size
        avg = 'binary' if n_classes == 2 else 'macro'
        score = f1_score(y_true.astype(int), y_pred_round, average=avg)
        print(f"F1 (treated as {'binary' if n_classes==2 else 'multiclass'}, average={avg}): {score:.4f}")
        return score

    print("F1 score not applicable: target appears continuous. Provide a threshold or use a classification target.")
    return None


def main():
    df = pd.read_csv("servo.data")   # or your DataFrame
    target = "motor_class"
    X = df.drop(columns=[target])
    y = df[target]

    # identify categorical columns (pandas dtype or list)
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric="RMSE",
        early_stopping_rounds=50,
        verbose=100
    )

    model.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_val, y_val))
    pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, pred, squared=False)
    print("RMSE:", rmse)

    # Try to compute F1 when it makes sense. By default we don't pass a threshold so
    # the helper will decide whether F1 is applicable.
    compute_f1(y_val, pred)

    # Save the trained model
    save_model(model, "catboost_model.cbm")


if __name__ == '__main__':
    main()