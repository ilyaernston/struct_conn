"""
Train and evaluate an SVM on persistence‐image features
to see how well they predict any metadata grouping.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

def load_and_merge(
        features_path: str, 
        metadata_path: str
        ) -> pd.DataFrame:
    '''Read features & metadata, merge on 'subject_id'

    Args:
        features_path (str): path to per-subject PI features CSV
        metadata_path (str): path to per-subject matadata CSV

    Returns:
        pd.DataFrame: df in SVM-ready format
    '''
    feats = pd.read_csv(features_path)
    meta  = pd.read_csv(metadata_path)
    df    = pd.merge(feats, meta, on='subject_id', how='inner')
    print(f"Merged: {df.shape[0]} subjects, {df.shape[1]} columns.")
    return df

def remove_uniques(df: pd.DataFrame, cols_to_check=None) -> pd.DataFrame:
    # work on a copy
    df = df.copy()

    if cols_to_check is None:
        for col in df.columns:
            if col == 'subject_id':
                continue

            # 1. find all values that occur exactly once
            counts = df[col].value_counts()
            singles = counts[counts == 1].index.tolist()
            
            if not singles:
                continue
            
            if len(singles) == 1:
                # exactly one singleton → drop its row
                unique_val = singles[0]
                df = df[df[col] != unique_val]
            else:
                # multiple singletons → replace them with 'other'
                mask = df[col].isin(singles)
                df.loc[mask, col] = 'other'
    else:
        for col in cols_to_check:
            # 1. find all values that occur exactly once
            counts = df[col].value_counts()
            singles = counts[counts == 1].index.tolist()
            
            if not singles:
                continue
            
            if len(singles) == 1:
                # exactly one singleton → drop its row
                unique_val = singles[0]
                df = df[df[col] != unique_val]
            else:
                # multiple singletons → replace them with 'other'
                mask = df[col].isin(singles)
                df.loc[mask, col] = 'other'
    
    return df

def run_svm(df: pd.DataFrame, label_col: str):
    """
    Train & evaluate an SVM on df[label_col].
    Falls back to non-stratified split if any class has < 2 samples.
    """
    # prepare X, y
    X = df.drop(columns=['subject_id', label_col])
    y = df[label_col].astype(str)

    # 1) cross‐val (still OK even if singleton classes exist,
    #    though you may get warnings)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc',    SVC(kernel='rbf', C=1.0, class_weight='balanced'))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        print(f"\n[{label_col}] 5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    except ValueError as e:
        print(f"\n[!] Cross‐val failed: {e}\n   You may have singleton classes.")

    # 2) hold-out split
    counts = y.value_counts()
    if (counts < 2).any():
        print(f"\n[!] Found singleton class(es):\n{counts[counts<2]}")
        print("    → Skipping stratified split, doing a plain random split.\n")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.3,
            random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.3,
            stratify=y,
            random_state=42
        )

    # 3) fit & report
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"\n[{label_col}] Hold-out test results:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

def logistic_regression(features, metadata):
    
    X_train, X_test, y_train, y_test = train_test_split(features, metadata, test_size=0.40, random_state=42)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr.score(X_test, y_test)

def main():
    
    features_path = '/Users/elijah/Desktop/thesis/test_PI_1_allsub/PI_features.csv'
    metadata_path = '/Users/elijah/Desktop/thesis/tests_2/all_labels.csv'

    # Load
    feats_df = pd.read_csv(features_path)

    meta_df = pd.read_csv(metadata_path)
    # prepend 'sub-' and rename
    meta_df['subject_id'] = 'sub-' + meta_df['subject_id'].astype(str)
    cols_to_check = ['Gender Indentity', 'Country of Birth', 'Place of Residense', 'Living environment', 
                     'Marital Status', 'Native Language', 'EthnicalIdentity', 'Religion', 'BloodSuger', 
                     'Medical Treatment', 'SupportHistory', 'Education']
    nominative_cols = ['Gender', 'DominantHand', 'Gender Indentity', 'Sexual Orientation', 'Country of Birth',
                         'Place of Residense', 'Living environment', 'Marital Status', 'Twins', 'Native Language', 
                         'EthnicalIdentity', 'PoliticalOrientation', 'EconomicalApproach', 'Religion',
                         'Depression', 'Anxiety', 'AttentionDisorders', 'CommunicationDisorders', 'VisualAid', 
                         'HearingAid', 'LongCovid', 'HolocaustLineage', 'SufferingReporting', 'Education', 
                         'Native Language Family', 'Number of Languages']
    meta_df = remove_uniques(meta_df)
    for col in meta_df.columns:
        print(meta_df[col].value_counts())

    # Show available metadata columns
    print("\nAvailable metadata columns:")
    valid_cols = [
        col
        for col in meta_df.columns
        if not (meta_df[col].value_counts() == 1).any()
    ]
    for col in valid_cols and nominative_cols:
        print("  •", col)

    # Ask which one to use
    label = input("\nEnter the exact column name to use as the label: ").strip()
    if label not in meta_df.columns:
        print(f"Error: '{label}' is not a column in your metadata.")
        return

    # Merge features + chosen label
    df = pd.merge(
        feats_df,
        meta_df[['subject_id', label]],
        on='subject_id',
        how='inner'
    )
    print(f"\nMerged DataFrame: {df.shape[0]} subjects × {df.shape[1]} columns.")
    df = df.dropna(subset=[label])
    print(f"\nShape after dropping NaNs: {df.shape[0]} subjects × {df.shape[1]} columns.")

    # Run SVM
    run_svm(df, label)

if __name__ == "__main__":
    main()
