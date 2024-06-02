import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import OneHotEncoder
plt.style.use('ggplot')

# Wczytanie danych z pliku 'iris.data' do obiektu DataFrame z pandas
data = pd.read_csv('iris.data', header=None)
X = data.iloc[:, :-1].values  # Wszystkie kolumny oprócz ostatniej jako cechy
y = data.iloc[:, -1].values   # Ostatnia kolumna jako etykiety

# Ustalenie nazw kolumn i klas
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target_names = np.unique(y)

# Skalowanie danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2)

# Konwersja etykiet do kodów numerycznych
K_vec = np.arange(1,111,2)
PK_1D_K = np.zeros(len(K_vec))

for k_ind in range(len(K_vec)):
    model = KNeighborsClassifier(n_neighbors=K_vec[k_ind])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    PK = accuracy_score(y_test, y_pred)*100
    print("K {} | PK {}". format(K_vec[k_ind], PK))
    PK_1D_K[k_ind] = PK

plt.figure(figsize=(8, 6))
plt.plot(K_vec, PK_1D_K, marker='o')
plt.xlabel('K')
plt.ylabel('PK')
plt.title('PK vs K for K-Nearest Neighbors')
plt.grid(True)
plt.savefig("Fig.1_PK_K_knn_iris_Neihgbor.png",bbox_inches='tight')