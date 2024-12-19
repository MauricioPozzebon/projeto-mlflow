import mlflow
from mlflow.models import infer_signature

import requests
import pandas as pd
import numpy as np
import json
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# URL do endpoint
url = "https://geradordevendas.fly.dev/historico/0/200"

# Requisição GET
response = requests.get(url)

if response.status_code == 200:
    data = response.json()

responses = {}
data = {}

for i in range(1, 40):
    responses[i] = requests.get(url)
    data[i] = responses[i].json()

data_ = {}

for i in range (1,40):
    data_[i] = data[i]

# Inicializa dicionários vazios
rows_ = {}
df_vendas_ = {}

for i in range(1, 40):
    # Inicializa uma lista de vendas para cada índice i
    rows_[i] = []  

    # Extraindo dados conforme a estrutura fornecida
    for venda in data_[i]['vendas']:
        cliente_id = venda['cliente']['id'] if venda['cliente']['id'] else 9999
        cliente_nome = venda['cliente']['nome'] if venda['cliente']['nome'] else 'não identificado'
        meio = venda['meio']
        transaction_id = venda['transaction_id']

        for item in venda['itens']:
            item_nome = item['item']
            preco = item['preço']
            categoria = item['categoria']

            # Adicionando uma linha para cada item no rows_[i]
            rows_[i].append({  
                'id': cliente_id,
                'nome': cliente_nome,
                'item': item_nome,
                'preço': preco,
                'categoria': categoria,
                'meio': meio,
                'transaction_id': transaction_id
            })

    # Convertendo as linhas para um DataFrame do pandas
    # Cria o nome dinâmico da variável, ex: df_vendas_1, df_vendas_2, etc.
    nome_df = f'df_vendas_{i}'
    globals()[nome_df] = pd.DataFrame(rows_[i])  # Cria a variável dinamicamente no escopo global

vendas = []

# Loop de 1 a 3 para acessar os DataFrames df_vendas_1, df_vendas_2, df_vendas_3
for i in range(1, 40):  
    # Acessa o DataFrame dinâmico (df_vendas_1, df_vendas_2, etc.) usando globals()
    df = globals()[f'df_vendas_{i}']
    soma = df['preço'].sum().round(2)
    vendas.append(soma)

df_venda_diaria = pd.DataFrame(vendas, columns=['venda_diaria'])

# Criar um intervalo de datas (1 data por dia)
date_range = pd.date_range(start='2024-01-01', periods=len(df_venda_diaria), freq='D')

# Definir as datas como índice do DataFrame
df_venda_diaria.index = date_range

train_size = int(len(df_venda_diaria) * 0.8)
train, test = df_venda_diaria[:train_size], df_venda_diaria[train_size:]

params = {'endog': train, 'order': [1,1,1]}

#p, d, q = 1, 1, 1
model = ARIMA(**params)
model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test))

rmse = np.sqrt(mean_squared_error(test, forecast))

mlflow.set_tracking_uri(uri="http://localhost:5000")

mlflow.set_experiment("MLflow Projeção de vendas")

with mlflow.start_run():
    params = {'endog': train,
        'order': [1,1,1]}

    mlflow.log_metric("RMSE", rmse)

    mlflow.set_tag("Informações do Experimento", "Projeção de Vendas")

    signature = infer_signature(train, model_fit.forecast(steps=len(test)))

        # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model_fit,
        artifact_path="api_vendas",
        signature=signature,
        input_example=train,
        registered_model_name="previsao-vendas",
    )

loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

forecast = model_fit.forecast(steps=len(test))

forecast
