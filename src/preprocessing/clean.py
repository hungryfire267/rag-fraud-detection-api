import os
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency

BASE_DIR = Path(__file__).resolve().parents[2]


class CleanData: 
    def __init__(self, data_paths_dict, missing_threshold): 
        self.transaction_df = pd.read_csv(data_paths_dict["transaction"])
        self.identity_df = pd.read_csv(data_paths_dict["identity"])
        
        self.missing_threshold = missing_threshold
        
        
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
            results.append({"column": col, "p_value": p, "missing_rate": flag.mean()})

        result_df = pd.DataFrame(results).sort_values("p_value")
        result_df["significant"] = result_df["p_value"] < alpha
        return result_df

    def run_data(self): 
        df = self.merge_data()
        no_merged_obs, no_merged_features = df.shape
        print(f"Our merged dataset has {no_merged_obs} observations and {no_merged_features} features")
        df = self.drop_dead_columns(df)
        no_dropped_obs, no_dropped_features = df.shape
        print(f"Our dataset with dropped dataset has {no_merged_obs} observations and {no_merged_features} features")
        df = self.missingness_pvalues(df)
        print(df.shape)
        
    
if __name__ == "__main__": 
    data_paths_dict = {
        "identity": os.path.join(BASE_DIR, "data/raw/identity.csv"),
        "transaction": os.path.join(BASE_DIR, "data/raw/transaction.csv")
    }
    
    clean_pipeline = CleanData(data_paths_dict, 0.98).run_data()
    
    
    
    