import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
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
y_train = pd.Series(y_train).astype('category').cat.codes.values
y_test = pd.Series(y_test).astype('category').cat.codes.values

# Definicja modelu
class Model(nn.Module):
    def __init__(self, input_dim, output_dim, K1, K2, dropout_p=0.5):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, K1),
            nn.BatchNorm1d(K1),
            nn.ReLU(),

        )
        self.layer2 = nn.Sequential(
            nn.Linear(K1, K2),
            nn.BatchNorm1d(K2),
            nn.ReLU(),

        )
        self.layer3 = nn.Linear(K2, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.softmax(self.layer3(x), dim=1)
        return x

lr_vec = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
K1_vec = np.arange(1, 111, 15)
K2_vec = K1_vec
PK_2D_K1K2 = np.zeros([len(K1_vec), len(K2_vec)])
max_epoch = 200
PK_2D_K1K2_max = 0
k1_ind_max = 0
k2_ind_max = 0

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).long()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).long()

for k1_ind in range(len(K1_vec)):
    for k2_ind in range(len(K2_vec)):
        model = Model(X_train.shape[1], len(target_names), K1_vec[k1_ind], K2_vec[k2_ind])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_vec[0])
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(max_epoch):
            model.train()
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)

            # Zero gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            correct = (torch.argmax(y_pred, dim=1) == y_test).float()
            PK = correct.mean().item() * 100
            print(f"K1: {K1_vec[k1_ind]}, K2: {K2_vec[k2_ind]}, PK: {PK:.2f}%")

            PK_2D_K1K2[k1_ind, k2_ind] = PK

        if PK > PK_2D_K1K2_max:
            PK_2D_K1K2_max = PK
            k1_ind_max = k1_ind
            k2_ind_max = k2_ind

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(K1_vec, K2_vec)
surf = ax.plot_surface(X, Y, PK_2D_K1K2.T, cmap='viridis')
ax.set_xlabel('K1')
ax.set_ylabel('K2')
ax.set_zlabel('PK')
ax.view_init(30, 200)
plt.savefig("Fig.4_PK_K1K2_batchnorm_pytorch_iris.png", bbox_inches='tight')
plt.show()
