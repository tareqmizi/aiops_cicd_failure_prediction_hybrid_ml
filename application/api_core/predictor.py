import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from xgboost import XGBClassifier
from typing import Dict, Any, List, Tuple

# Paths to the model files (assumed to be in the same folder or parent folder)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Helper to find valid model file
def get_valid_path(filename):
    local_p = os.path.join(CURRENT_DIR, filename)
    parent_p = os.path.join(os.path.dirname(CURRENT_DIR), filename)
    if os.path.exists(local_p) and os.path.getsize(local_p) > 0:
        return local_p
    elif os.path.exists(parent_p) and os.path.getsize(parent_p) > 0:
        return parent_p
    return local_p

XGB_PATH = get_valid_path("aiops_xgboost_model.json")
LSTM_PATH = get_valid_path("aiops_lstm_model.keras")
SCALER_PATH = get_valid_path("aiops_lstm_scaler.pkl")

XGB_FEATS = [
    "author_commit_span_days", "author_historical_fail_rate", "author_build_count_30d",
    "is_multi_project_author", "repo_switch_velocity", "is_unreviewed_pr_merge",
    "pr_comment_density", "is_off_hour_commit", "is_weekend_commit",
    "hour_sin", "hour_cos", "churn_ratio", "log_git_diff_src_churn",
    "log_git_diff_test_churn", "test_coverage_ratio", "test_density_kloc",
    "test_lines_deleted", "is_main_branch", "branch_contributor_count",
    "commits_in_push", "log_gh_sloc", "log_gh_team_size",
]

SEQ_FEATS = [
    "consecutive_failures", "fail_streak_count", "rolling_fail_rate_5",
    "time_gap_seconds", "time_since_last_failure",
    "log_git_diff_src_churn", "churn_velocity", "author_switch_flag"
]

class AIOpsPredictor:
    def __init__(self):
        print("Loading models and scaler...")
        # 1. Load XGBoost
        self.xgb_model = XGBClassifier()
        if os.path.exists(XGB_PATH):
            self.xgb_model.load_model(XGB_PATH)
            print(f"Loaded XGBoost model from {XGB_PATH}")
        else:
            print(f"Warning: XGBoost model not found at {XGB_PATH}")

        # 2. Load LSTM (wrapped in try/except to prevent crash if CUDA/Tensorflow issues arise)
        self.lstm_model = None
        if os.path.exists(LSTM_PATH):
            try:
                self.lstm_model = tf.keras.models.load_model(LSTM_PATH)
                print(f"Loaded LSTM model from {LSTM_PATH}")
            except Exception as e:
                print(f"Error loading LSTM model: {e}")
        else:
            print(f"Warning: LSTM model not found at {LSTM_PATH}")

        # 3. Load Scaler
        self.scaler = None
        if os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)
            print(f"Loaded scaler from {SCALER_PATH}")
        else:
            print(f"Warning: Scaler not found at {SCALER_PATH}")

    def engineer_features(self, current: Dict[str, Any], history: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Takes current build dictionary and history of past builds (up to 10),
        computes engineered features for XGBoost and sequential features for LSTM.
        """
        # Convert to Pandas for convenient computation
        all_builds = history + [current]
        df = pd.DataFrame(all_builds)
        
        # Ensure timestamp is parsed
        df["gh_build_started_at"] = pd.to_datetime(df.get("gh_build_started_at", pd.Series(pd.Timestamp.now(), index=df.index)))
        df.sort_values("gh_build_started_at", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Ensure target 'y' exists (passed builds: status!='failed' and status!='errored', cancel filtered by backend)
        if "y" not in df.columns:
            df["y"] = df.get("tr_status", pd.Series("passed", index=df.index)).isin(["failed", "errored"]).astype(int)

        # ── Feature Engineering for XGBoost (Current Index = len(df)-1) ──
        # Fill missing values with reasonable defaults
        df["git_diff_src_churn"] = df.get("git_diff_src_churn", pd.Series(0.0, index=df.index)).fillna(0).astype("float32")
        df["git_diff_test_churn"] = df.get("git_diff_test_churn", pd.Series(0.0, index=df.index)).fillna(0).astype("float32")
        df["gh_num_pr_comments"] = df.get("gh_num_pr_comments", pd.Series(0.0, index=df.index)).fillna(0).astype("float32")
        df["gh_is_pr"] = df.get("gh_is_pr", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        df["gh_sloc"] = df.get("gh_sloc", pd.Series(100.0, index=df.index)).fillna(100).replace(0, 100).astype("float32")
        df["gh_team_size"] = df.get("gh_team_size", pd.Series(1.0, index=df.index)).fillna(1).astype("float32")
        df["gh_test_cases_per_kloc"] = df.get("gh_test_cases_per_kloc", pd.Series(0.0, index=df.index)).fillna(0).astype("float32")
        df["gh_diff_tests_deleted"] = df.get("gh_diff_tests_deleted", pd.Series(0.0, index=df.index)).fillna(0).astype("float32")
        df["gh_num_commits_in_push"] = df.get("gh_num_commits_in_push", pd.Series(1.0, index=df.index)).fillna(1).astype("float32")

        # Computed features
        ts = df["gh_build_started_at"]
        df["author_commit_span_days"] = (ts - ts.min()).dt.total_seconds() / 86400.0
        
        # Author fail rate & build count (proxied by project name history)
        df["author_historical_fail_rate"] = df["y"].shift(1).rolling(30, min_periods=1).mean().fillna(0.0)
        df["author_build_count_30d"] = df["y"].shift(1).rolling(30, min_periods=1).count().fillna(0.0)
        df["is_multi_project_author"] = 0
        df["repo_switch_velocity"] = 0.0

        # PR features
        df["is_unreviewed_pr_merge"] = (df["gh_is_pr"] & (df["gh_num_pr_comments"] == 0) & (df["git_diff_src_churn"] > 300)).astype(int)
        df["pr_comment_density"] = df["gh_num_pr_comments"] / (df["git_diff_src_churn"] + 1)

        # Temporal features
        hour = ts.dt.hour
        dow = ts.dt.dayofweek
        df["is_off_hour_commit"] = ((hour >= 22) | (hour < 6)).astype(int)
        df["is_weekend_commit"] = (dow >= 5).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        # Churn ratios
        df["churn_ratio"] = df["git_diff_src_churn"] / df["gh_sloc"]
        df["log_git_diff_src_churn"] = np.log10(df["git_diff_src_churn"].clip(lower=0) + 1)
        df["log_git_diff_test_churn"] = np.log10(df["git_diff_test_churn"].clip(lower=0) + 1)
        df["test_coverage_ratio"] = df["git_diff_test_churn"] / (df["git_diff_src_churn"] + 1)

        # Test features
        df["test_density_kloc"] = df["gh_test_cases_per_kloc"]
        df["test_lines_deleted"] = df["gh_diff_tests_deleted"]

        # Branch contributor and name
        branches = ["main", "master", "production", "release"]
        df["is_main_branch"] = df.get("git_branch", pd.Series("main", index=df.index)).str.lower().isin(branches).astype(int)
        df["branch_contributor_count"] = df.get("branch_contributor_count", pd.Series(1.0, index=df.index)).fillna(1.0)
        
        # Scale parameters
        df["commits_in_push"] = df["gh_num_commits_in_push"]
        df["log_gh_sloc"] = np.log10(df["gh_sloc"].clip(lower=0) + 1)
        df["log_gh_team_size"] = np.log10(df["gh_team_size"].clip(lower=0) + 1)

        # Extract current build engineered features
        current_xgb_feats = df.iloc[-1][XGB_FEATS].to_numpy(dtype="float32").reshape(1, -1)

        # ── Feature Engineering for LSTM (Sequences over recent history) ──
        # We need to construct a window of 10 builds
        # If there are fewer than 10 builds, we pad with zeros
        df["consecutive_failures"] = df["y"].shift(1).fillna(0.0)
        _sr = (df["y"].shift(1).fillna(0.0) == 0.0).astype(int).cumsum()
        df["fail_streak_count"] = df["y"].shift(1).fillna(0.0).groupby(_sr).cumsum()
        df["rolling_fail_rate_5"] = df["y"].shift(1).rolling(5, min_periods=1).mean().fillna(0.0)
        
        ts_sec = ts.astype("int64") // 10**9
        df["time_gap_seconds"] = ts_sec.diff().fillna(0.0).clip(lower=0)
        _ft = ts_sec.where(df["y"].shift(1) == 1)
        df["time_since_last_failure"] = (ts_sec - _ft.ffill().fillna(ts_sec.iloc[0])).clip(lower=0)
        
        df["log_git_diff_src_churn"] = np.log10(df["git_diff_src_churn"].clip(lower=0) + 1)
        df["churn_velocity"] = df["git_diff_src_churn"].diff().fillna(0.0)
        df["author_switch_flag"] = (df.get("git_trigger_commit", pd.Series("", index=df.index)) != 
                                    df.get("git_trigger_commit", pd.Series("", index=df.index)).shift(1)).astype(int)

        # Slice last 10 rows
        sub_df = df[SEQ_FEATS].copy()
        
        # Scale using StandardScaler fit from notebook
        scaled_seq = self.scaler.transform(sub_df.values) if self.scaler else sub_df.values
        
        # Construct window (10, 8)
        w = 10
        n_features = len(SEQ_FEATS)
        if len(scaled_seq) >= w:
            current_lstm_seq = scaled_seq[-w:]
        else:
            # Pad beginning with zeros
            pad_len = w - len(scaled_seq)
            current_lstm_seq = np.zeros((w, n_features), dtype="float32")
            current_lstm_seq[pad_len:] = scaled_seq
            
        current_lstm_seq = current_lstm_seq.reshape(1, w, n_features).astype("float32")

        return current_xgb_feats, current_lstm_seq

    def predict(self, current: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs XGBoost & LSTM models, computes ensemble risk score.
        """
        # Run feature engineering
        xgb_input, lstm_input = self.engineer_features(current, history)
        
        # 1. Run XGBoost Prediction
        xgb_proba = 0.5
        try:
            xgb_proba = float(self.xgb_model.predict_proba(xgb_input)[0][1])
        except Exception as e:
            print(f"XGBoost predict error: {e}")

        # 2. Run LSTM Prediction
        lstm_proba = 0.5
        if self.lstm_model is not None:
            try:
                lstm_proba = float(self.lstm_model.predict(lstm_input, verbose=0)[0][0])
            except Exception as e:
                print(f"LSTM predict error: {e}")
        else:
            # Fallback mock/correlation score if LSTM loading was skipped
            lstm_proba = xgb_proba * 0.9 + 0.05
            
        # 3. Ensemble
        ensemble_score = 0.5 * xgb_proba + 0.5 * lstm_proba
        
        # Determine risk category
        if ensemble_score < 0.35:
            category = "Low"
        elif ensemble_score < 0.70:
            category = "Medium"
        else:
            category = "High"

        return {
            "xgboost_probability": round(xgb_proba, 4),
            "lstm_probability": round(lstm_proba, 4),
            "ensemble_risk_score": round(ensemble_score, 4),
            "risk_category": category
        }
