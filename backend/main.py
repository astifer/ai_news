from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware

import pandas as pd
from io import BytesIO

from backend.models import Model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {
    1: "cointegrated/LaBSE-en-ru",
    2: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",

}

model = Model(name=models[2], path_to_weights='backend/weights/bert_clf', cuda=False)


@app.get('/')
def read_root():
    return {'message': 'Kurenkov AI api. See more at /docs'}


@app.post("/csv", response_class=StreamingResponse)
async def process_csv(file: UploadFile = File(...)):
    """
        CSV файл с кодировкой UTF-8, в котором существует колонка - text
    """
    df = pd.read_csv(file.file, quotechar='"', escapechar='\\')
    news = df['text'].tolist()

    processed_data = model.process_data(news)

    output_data = BytesIO()
    processed_data.to_csv(output_data, index=False)
    output_data.seek(0)

    return StreamingResponse(output_data, headers={
        'Content-Disposition': 'attachment; filename="processed.csv"',
        'charset': 'utf-8'
    })


@app.post("/excel", response_class=StreamingResponse)
async def excel(file: UploadFile = File(...)):
    """
        Excel файл с кодировкой UTF-8, в котором существует колонка - text
    """
    content = await file.read()
    df = pd.read_excel(content)
    news = df['text'].values.tolist()

    processed_data = model.process_data(news)

    output_data = BytesIO()
    processed_data.to_excel(output_data, index=False)
    output_data.seek(0)

    return StreamingResponse(output_data,
                             media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@app.post('/update_device')
async def update_device(is_cuda: bool):
    """
    Переключение gpu/cpu. По дефолту установлена работа на gpu
    """
    global model
    info = ''

    try:
        model = Model(name=models[2], path_to_weights='backend/weights/bert_clf', cuda=is_cuda)
        info = 'succesfully switched to new device'
    except:
        info = 'Failed to switch device'
    finally:
        info += f'. Device: {model._DEVICE}'
        return {'status': str(info)}
