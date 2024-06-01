from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Criar uma instância do FastAPI
app = FastAPI()

# Criar uma classe que terá os dados do request body para a API
class request_body(BaseModel):
    horas_estudo: float

# Carregar o modelo de regressão linear para fazer as previsões
modelo_pontuacao = joblib.load('models/modelo_linear_regression.pkl')

def predict_pontuacao(data: request_body):
    # Preparar os dados para predição
    input_feature = [[data.horas_estudo]]

    # Fazer a predição
    y_pred = modelo_pontuacao.predict(input_feature)[0].astype(float)

    return {'pontuacao': y_pred.tolist()}

# Criar uma rota para a API
@app.post('/predict')
def predict(data: request_body):
    return predict_pontuacao(data)