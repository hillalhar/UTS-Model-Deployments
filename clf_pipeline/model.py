from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

def classifier_model(n_estimators=200, class_weight='balanced', max_depth=4, random_state=42):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        max_depth=max_depth,
        random_state=random_state,
    )