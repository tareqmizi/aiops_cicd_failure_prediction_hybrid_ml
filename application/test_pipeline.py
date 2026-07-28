import pandas as pd
import json
import asyncio
from api_core.predictor import AIOpsPredictor
from api_core.rca import analyze_logs_with_ollama

async def main():
    print("=== AIOps Pipeline Verification Script ===")
    
    # 1. Initialize Predictor (loads models and scaler)
    predictor = AIOpsPredictor()
    
    # 2. Load sample data
    print("\nLoading sample_test_data.csv...")
    try:
        df = pd.read_csv("sample_test_data.csv")
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return
        
    print(f"Loaded {len(df)} rows. Finding a project with enough history...")
    
    # Group by project name to find projects with >= 11 builds
    counts = df["gh_project_name"].value_counts()
    eligible = counts[counts >= 11].index.tolist()
    
    if not eligible:
        print("Error: No project has 11 or more builds in the sample dataset.")
        return
        
    target_project = eligible[0]
    print(f"Selected project: '{target_project}' with {counts[target_project]} builds.")
    
    # Get rows for this project and sort by timestamp
    proj_df = df[df["gh_project_name"] == target_project].copy()
    proj_df["gh_build_started_at"] = pd.to_datetime(proj_df["gh_build_started_at"])
    proj_df = proj_df.sort_values("gh_build_started_at").reset_index(drop=True)
    
    # Slice first 11 rows: 10 for history, 1 for current prediction
    subset = proj_df.iloc[:11].to_dict(orient="records")
    history = subset[:10]
    current = subset[10]
    
    print("\nSample current commit metadata:")
    print(json.dumps({k: current[k] for k in ["gh_project_name", "git_branch", "git_trigger_commit", "gh_build_started_at", "git_diff_src_churn"] if k in current}, indent=2, default=str))
    
    # 3. Test Prediction
    print("\nRunning feature engineering and model prediction...")
    try:
        results = predictor.predict(current, history)
        print("\nPrediction Results:")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()

    # 4. Test Log Analysis (RCA)
    print("\nRunning log root cause analysis (Ollama check)...")
    mock_log = """
    [INFO] Parsing build configuration...
    [INFO] Fetching dependencies...
    [ERROR] npm ERR! code ECONNREFUSED
    [ERROR] npm ERR! syscall connect
    [ERROR] npm ERR! address 127.0.0.1
    [ERROR] npm ERR! port 8081
    [ERROR] npm ERR! network ECONNREFUSED: connection refused to 127.0.0.1:8081
    [ERROR] Build step failed with status 1.
    """
    
    rca_res = await analyze_logs_with_ollama(mock_log)
    print("\nRCA Results (from LLM or rule-based fallback):")
    print(json.dumps(rca_res, indent=2))
    
    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    asyncio.run(main())
