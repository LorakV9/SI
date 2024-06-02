import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

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
            nn.Dropout(dropout_p)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(K1, K2),
            nn.BatchNorm1d(K2),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.layer3 = nn.Linear(K2, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return F.softmax(x, dim=1)

# Define hyperparameter grids
lr_vec = np.arange(0.01, 1.01, 0.05)
K1_vec = np.arange(1, 111, 15)
K2_vec = np.arange(1, 111, 15)
PK_2D_K1K2 = np.zeros((len(K1_vec), len(K2_vec), len(lr_vec)))
max_epoch = 200
avg_PK = np.zeros(len(lr_vec))

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).long()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).long()

for lr_ind, lr in enumerate(lr_vec):
    for k1_ind, K1 in enumerate(K1_vec):
        for k2_ind, K2 in enumerate(K2_vec):
            model = Model(X_train.shape[1], len(target_names), K1, K2)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()

            for epoch in range(max_epoch):
                model.train()
                y_pred = model(X_train)
                loss = loss_fn(y_pred, y_train)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                correct = (torch.argmax(y_pred, dim=1) == y_test).float()
                PK = correct.mean().item() * 100
                PK_2D_K1K2[k1_ind, k2_ind, lr_ind] = PK

                # Print the current state
                print(f'Learning Rate: {lr}, K1: {K1}, K2: {K2}, PK: {PK:.2f}%')

    avg_PK[lr_ind] = np.mean(PK_2D_K1K2[:, :, lr_ind])
    print(f'Average PK for LR {lr}: {avg_PK[lr_ind]:.2f}%')

# Tworzenie wykresu zależności średniego PK od learning rate
plt.figure(figsize=(8, 6))
plt.plot(lr_vec, avg_PK, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Średnie PK')
plt.title('Zależność średniego PK od Learning Rate')
plt.grid(True)
plt.savefig("Fig.2_PK_vs_LR_pytorch_iris.png",bbox_inches='tight')
