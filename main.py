from fastapi import FastAPI, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from os.path import join
import pandas as pd
from ml_model import Regressors

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://datarminism.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/regressor")
async def create_upload_file(file: UploadFile = Form(), request: str = Form()):
    data = await file.read()
    path = join(os.getcwd(), 'data', file.filename)
    with open(path, 'wb') as f:
        f.write(data)
    if "csv" in file.filename:
        df = pd.read_csv(join(os.getcwd(), 'data', file.filename))
    elif "xlsx" in file.filename:
        df = pd.read_excel(join(os.getcwd(), 'data', file.filename))
    else:
        return HTTPException(status_code=422, detail="Wrong file extension!")
    # print(df.head())
    regressors = Regressors(df.loc[1:, :])
    try:
        x_data = [float(x) for x in request.strip().split(", ")]
        result = regressors.train(x_data)
        return result
    except:
        return HTTPException(status_code=422, detail="Wrong data input format!")
    finally:
        os.remove(join(path))






