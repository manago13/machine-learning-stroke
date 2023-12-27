# Stroke Prediction
O objetivo deste projeto é aplicar técnicas de Machine Learning para determinar se um indivíduo terá ou não AVC, sem a utilização de métodos de boosting ou stacking.

## ⚙️Preparação do Ambiente

### Conjunto de dados

*Disponíveel no diretório `datasets`.*

Dataset obtido em: [https://www.kaggle.com/competitions/playground-series-s3e2/data?select=train.csv](https://www.kaggle.com/competitions/playground-series-s3e2/data?select=train.csv)

### Bibliotecas

Este notebook foi desenvolvido em Jupiter e utiliza as bibliotecas abaixo:

```
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 ```

## ⚒️ Metodologia

### Análise Exploratória dos Dados e Tratamentos:

1. Exploração inicial do conjunto de dados para entender suas características.
1. Identificação das variáveis relevantes para o problema de determinar o "Stroke".
1. Tratamento de dados ausentes ou inconsistentes, se necessário.

### Escolha dos Algoritmos de Machine Learning:

Algoritmos escolhidos para aplicação do problema de predição de AVC:
1. Árvore de Decisão;
2. Random Forest;
3. KNN;
4. MLP;
5. Regressão Logística.

### Amostragem de Dados:

As seguintes técnicas de amostragem de dados foram utilizadas para preparar o conjunto de treinamento e teste:
1. K-Fold;
2. Hold-Out.

### Implementação:

Python com pacote do scikit-learn (sklearn), com utilização de pipeline e gridsearchcv para escolha dos melhores hiperparâmetros.

### Avaliação e Comparação dos Modelos:

Os modelos foram avaliados com base em métricas adequadas para problemas de classificação: acurácia, precisão, recall e F1-score.

&nbsp;

*Nota: Há um detalhamento minucioso de cada etapa no notebook.*
