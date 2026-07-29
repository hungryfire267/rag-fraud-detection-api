import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parents[2]


class CleanData: 
    def __init__(self, data_paths_dict, missing_threshold, fi_threshold): 
        self.transaction_df = pd.read_csv(data_paths_dict["transaction"])
        self.identity_df = pd.read_csv(data_paths_dict["identity"])
        
        self.missing_threshold = missing_threshold
        self.fi_threshold = fi_threshold
        
        
    def merge_data(self): 
        df = self.transaction_df.merge(
            right=self.identity_df, left_on="TransactionID", right_on="TransactionID", how="left"
        )
        return df
    
    def drop_dead_columns(self, df): 
        missing = df.isna().mean() 
        dead = missing[missing > self.missing_threshold].index.tolist()
        const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        return df.drop(columns=set(dead + const))
    
    def missingness_pvalues(self, df, target_col="isFraud", alpha=0.05):
        results = []
        for col in df.columns:
            if col == target_col:
                continue
            flag = df[col].isna().astype(int)
            if flag.nunique() < 2:
                continue  # no missingness at all, nothing to test

            table = pd.crosstab(flag, df[target_col])
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue

            chi2, p, dof, expected = chi2_contingency(table)
            phi = flag.corr(df[target_col])
            results.append({"column": col, "p_value": p, "phi": phi, "missing_rate": flag.mean()})

        result_df = pd.DataFrame(results).sort_values("p_value")
        result_df["significant"] = result_df["p_value"] < alpha
        
        condition_significance = result_df["significant"] == True 
        condition_phi = result_df["phi"].abs() >= 0.025
        relevant_result_df = result_df[condition_significance & condition_phi].copy()
        relevant_result_df_cols = relevant_result_df["column"].tolist()
        if "isFraud" not in relevant_result_df_cols: 
            relevant_result_df_cols.append("isFraud")  
        return result_df, df[relevant_result_df_cols]
    
    def get_feature_importance(self, df, target_col="isFraud"): 
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        cat_cols = X.select_dtypes(include="object").columns
        for col in cat_cols:
            X[col] = X[col].astype("category")
        
        xgboost_model = XGBClassifier(
            n_estimators=200, 
            max_depth=4, 
            enable_categorical=True,
            random_state=42
        )
        xgboost_model.fit(X, y)
        importance = pd.Series(xgboost_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        no_features_importance = len(importance.index)
        
        nonzero_importance = importance[importance > 0]
        no_features_nonzero_importance = len(nonzero_importance)
        zero_importance = no_features_importance - no_features_nonzero_importance
        print(f"There are {zero_importance} features of zero importance")
        
        cum_importance = nonzero_importance.cumsum() / nonzero_importance.sum()
        keep_cols = cum_importance[cum_importance <= self.fi_threshold].index.tolist()
        # add the feature that crosses the 95% line too
        if len(keep_cols) < len(cum_importance):
            keep_cols.append(cum_importance.index[len(keep_cols)])
        print(f"Keeping {len(keep_cols)} features at {self.fi_threshold * 100}% cumulative importance")
        
        if "isFraud" not in keep_cols: 
            keep_cols.append("isFraud")
        return df[keep_cols]
    
    def get_test_auc(self, df, target_col="isFraud"): 
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        cat_cols = X.select_dtypes(include="object").columns
        for col in cat_cols:
            X[col] = X[col].astype("category")
        
        stratified_kfold = StratifiedKFold(
            n_splits=10,
            shuffle=True,
            random_state=42
        )
        
        auc_scores = []
        
        for train_idx, test_idx in stratified_kfold.split(X, y): 
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                enable_categorical=True,
                random_state=42
            )

            model.fit(X_train, y_train)

            proba = model.predict_proba(X_test)[:, 1]
            auc_scores.append(roc_auc_score(y_test, proba))

        return np.mean(auc_scores)

    def run_data(self): 
        df = self.merge_data()
        no_merged_obs, no_merged_features = df.shape
        print(f"Our merged dataset has {no_merged_obs} observations and {no_merged_features} features")
        df = self.drop_dead_columns(df)
        no_dropped_obs, no_dropped_features = df.shape
        print(f"Our dropped dataset has {no_dropped_obs} observations and {no_dropped_features} features")
        diff_dropped_features = no_merged_features - no_dropped_features
        print(f"From the dropped dataset, we have dropped {diff_dropped_features} from the merged dataset with {self.missing_threshold} missing threshold.")
        missing_pvalues_result_df, df = self.missingness_pvalues(df)
        no_pvalue_obs, no_pvalue_features = df.shape
        print(f"Our reduced dataset from excluding insignificant pvalues has {no_pvalue_obs} observations and {no_pvalue_features} features")
    
        df = self.get_feature_importance(df)
        print(df.columns)
        return missing_pvalues_result_df
        
    
if __name__ == "__main__": 
    data_paths_dict = {
        "identity": os.path.join(BASE_DIR, "data/raw/identity.csv"),
        "transaction": os.path.join(BASE_DIR, "data/raw/transaction.csv")
    }
    
    clean_pipeline = CleanData(data_paths_dict, 0.98).run_data()
    
    
    
    
    