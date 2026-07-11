import io
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json

app = FastAPI(
    title="AI Data Analyst API",
    description="Production-grade API backend for automated data cleaning, profiling, and ML insights.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Welcome to the AI Data Analyst Core API v1",
        "docs_url": "/docs"
    }

@app.post("/api/v1/upload", status_code=status.HTTP_200_OK)
async def upload_and_clean_file(file: UploadFile = File(...)):
    # 1. Validate file format extension
    filename = file.filename.lower()
    if not (filename.endswith('.csv') or filename.endswith('.xlsx')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid file format. Only .csv and .xlsx files are supported."
        )
    
    try:
        # 2. Read contents into memory
        contents = await file.read()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
            
        # Record baseline data profile metrics
        initial_rows, initial_cols = df.shape
        
        # 3. Automated Data Cleaning Process
        # Rule A: Deduplication
        df.drop_duplicates(inplace=True)
        
        # Rule B: Missing values strategy
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Fill missing numbers with column Mean
                df[col] = df[col].fillna(df[col].mean())
            else:
                # Fill missing strings/objects with "Unknown"
                df[col] = df[col].fillna("Unknown")
                
        final_rows, final_cols = df.shape
        
        # 4. Generate Metadata Summary Statistics
        summary_stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                summary_stats[col] = {
                    "type": "numeric",
                    "mean": float(df[col].mean()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                }
            else:
                summary_stats[col] = {
                    "type": "categorical",
                    "unique_values": int(df[col].nunique())
                }
                
        # Prepare small preview sample safely converted to native JSON types
        cleaned_sample = json.loads(df.head(10).to_json(orient="records"))
        
        return {
            "metadata": {
                "filename": file.filename,
                "initial_shape": [initial_rows, initial_cols],
                "cleaned_shape": [final_rows, final_cols],
                "duplicates_removed": int(initial_rows - final_rows)
            },
            "summary_statistics": summary_stats,
            "data_preview": cleaned_sample
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the dataset: {str(e)}"
        )
