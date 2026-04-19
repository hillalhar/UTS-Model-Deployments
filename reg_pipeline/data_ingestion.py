import pandas as pd
from sklearn.model_selection import train_test_split

def load_data_reg(filepath):
    df = pd.read_csv(filepath)
    
    # Filter hanya mahasiswa placement = 1
    df_placed = df[df["placement_status"] == 1].copy() 
    
    X = df_placed.drop(columns=["placement_status", "salary_package_lpa"])
    y = df_placed["salary_package_lpa"] #
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

