import pandas as pd
import wandb
import json
import re
from datetime import datetime, timedelta

# Optional: Filter out the noisy Pydantic warnings
import warnings
# Filter all warnings from pydantic module (catches UnsupportedFieldAttributeWarning and others)
warnings.filterwarnings("ignore", module="pydantic")

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("sjxu_gamma/fluxsae")

data_list = []

for run in runs:
    # 1. Filter: Only process runs with key phrase in the name
    if '_1_ff' in run.name:
        try:
            # Access the summary dictionary safely
            # Handle different possible types of run.summary
            summary = None
            
            if isinstance(run.summary, dict):
                summary = run.summary
            elif hasattr(run.summary, '_json_dict'):
                # Check if _json_dict is a string (JSON) or already a dict
                json_dict = run.summary._json_dict
                if isinstance(json_dict, str):
                    summary = json.loads(json_dict)
                elif isinstance(json_dict, dict):
                    summary = json_dict
                else:
                    # Try to convert it
                    summary = dict(run.summary) if hasattr(run.summary, 'keys') else {}
            else:
                # Try to convert to dict
                summary = dict(run.summary) if hasattr(run.summary, 'keys') else {}
            
            # 2. Safety Check: skip this run if summary is empty or None
            if not summary or not isinstance(summary, dict):
                continue
            
            # Start a dictionary for this row with the run name
            row = {'name': run.name}
            
            # Extract expansion value from run name (e.g., "exp16" -> 16, "exp0.5" -> 0.5)
            exp_match = re.search(r'exp(\d+\.?\d*)', run.name)
            if exp_match:
                exp_value = exp_match.group(1)
                # Convert to float if it contains a decimal, otherwise int
                row['exp'] = float(exp_value) if '.' in exp_value else int(exp_value)
            else:
                row['exp'] = None
            
            # Add created timestamp
            row['created'] = run.created_at if hasattr(run, 'created_at') else None

            # 3. Extract explained_variance metrics from history (average of last 10 points)
            # Get all explained_variance keys from summary to know what to look for
            explained_variance_keys = [key for key in summary.keys() if 'explained_variance' in key]
            
            # For each explained_variance metric, get the average of last 10 points
            for key in explained_variance_keys:
                try:
                    # Get the history for this specific metric
                    # The metric name in history might be just the key or might need adjustment
                    history = run.history(keys=[key], pandas=True)
                    
                    if history is not None and not history.empty and key in history.columns:
                        # Get the last 10 points (or all if less than 10)
                        last_points = history[key].dropna().tail(10)
                        if len(last_points) > 0:
                            # Calculate average of last 10 points
                            row[key] = last_points.mean()
                        else:
                            # Fallback to summary value if no history data
                            row[key] = summary.get(key)
                    else:
                        # Fallback to summary value if history is not available
                        row[key] = summary.get(key)
                except Exception as e:
                    # If history fetch fails, fallback to summary value
                    row[key] = summary.get(key)
            
            # Only append if we actually found relevant columns (optional, but keeps it clean)
            if len(row) > 1:        
                data_list.append(row)
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as e:
            # Skip runs with problematic summaries
            print(f"Skipping run {run.name} due to error: {e}")
            continue

# Create the DataFrame
runs_df = pd.DataFrame(data_list)

# Ensure exp column is numeric (handle None values)
if 'exp' in runs_df.columns:
    # Convert to numeric, coercing errors to NaN
    runs_df['exp'] = pd.to_numeric(runs_df['exp'], errors='coerce')

# Convert created to datetime for proper sorting
if 'created' in runs_df.columns:
    runs_df['created'] = pd.to_datetime(runs_df['created'], errors='coerce')

# Sort by exp first (ascending), then move rows older than 5 hours to the bottom
if 'exp' in runs_df.columns and 'created' in runs_df.columns:
    # Calculate 5 hours ago from now
    # Handle timezone-aware timestamps
    if runs_df['created'].notna().any():
        sample_created = runs_df['created'].dropna().iloc[0]
        if hasattr(sample_created, 'tz') and sample_created.tz is not None:
            # If created is timezone-aware, use UTC for comparison
            five_hours_ago = pd.Timestamp.now(tz='UTC') - timedelta(hours=5)
        else:
            # If created is naive, use naive timestamp
            five_hours_ago = pd.Timestamp.now() - timedelta(hours=5)
    else:
        five_hours_ago = pd.Timestamp.now() - timedelta(hours=5)
    
    # Create a column to mark rows older than 5 hours (0 = recent, 1 = old)
    runs_df['_is_old'] = (runs_df['created'] < five_hours_ago).astype(int)
    
    # Sort: exp ascending, then _is_old (recent first), then created descending
    runs_df = runs_df.sort_values(
        by=['exp', '_is_old', 'created'],
        ascending=[True, True, False],  # exp ascending, recent first, created descending
        na_position='last'
    ).reset_index(drop=True)
    
    # Drop the temporary column
    runs_df = runs_df.drop(columns=['_is_old'])
elif 'exp' in runs_df.columns:
    # If no created column, just sort by exp
    runs_df = runs_df.sort_values(
        by='exp',
        ascending=True,
        na_position='last'
    ).reset_index(drop=True)
elif 'created' in runs_df.columns:
    # If no exp column, just sort by created descending
    runs_df = runs_df.sort_values(
        by='created',
        ascending=False,
        na_position='last'
    ).reset_index(drop=True)

# 4. Print the result in tab-separated format for easy copy-paste into Google Sheets
print(runs_df.to_csv(sep='\t', index=False))