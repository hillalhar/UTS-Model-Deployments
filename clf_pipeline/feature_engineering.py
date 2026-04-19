import pandas as pd
from sklearn.preprocessing import FunctionTransformer

def feature_eng(X):
    X_out = X.copy()
    
    # Create new feature
    X_out["academic_score"] = (X_out["ssc_percentage"] + X_out["hsc_percentage"] + X_out["degree_percentage"]) / 3
    
    X_out["skill_index"] = (X_out["technical_skill_score"] * 0.4 + 
                            X_out["soft_skill_score"] * 0.4 + 
                            X_out["certifications"] * 0.2)
    
    X_out["experience_index"] = (X_out["work_experience_months"] * 0.5 + 
                                 X_out["internship_count"] * 0.3 + 
                                 X_out["live_projects"] * 0.2)
    
    # cols to drop sesuai dengan EDA
    cols_to_drop = [
        "ssc_percentage", "hsc_percentage", "degree_percentage", "student_id", 
        "technical_skill_score", "soft_skill_score", "certifications",
        "work_experience_months", "internship_count", "live_projects"
    ]
    
    X_out = X_out.drop(columns=[c for c in cols_to_drop if c in X_out.columns])
    return X_out

