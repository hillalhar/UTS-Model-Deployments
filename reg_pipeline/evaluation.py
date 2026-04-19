from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_regression(y_true, y_pred):
    metrics = {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "r2": round(r2_score(y_true, y_pred), 4) #
    }
    
    report = f"MAE: {metrics['mae']} | RMSE: {metrics['rmse']} | R2 Score: {metrics['r2']}"
    return metrics, report