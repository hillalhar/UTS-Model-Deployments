import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    filepath=path
    df = pd.read_csv(filepath)
    
    target_col = "placement_status"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # stratify untuk keep rasio imbalance di train test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[*] Data berhasil di-split. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

