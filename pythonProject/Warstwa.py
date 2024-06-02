import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.nn as nn
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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)

# Konwersja etykiet do kodów numerycznych
y_train = pd.Series(y_train).astype('category').cat.codes.values
y_test = pd.Series(y_test).astype('category').cat.codes.values

# Definicja modelu
class Model(nn.Module):
    def __init__(self, input_dim, output_dim, layers, dropout_p=0.5):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, layers[i]))
            else:
                self.layers.append(nn.Linear(layers[i-1], layers[i]))
            self.layers.append(nn.BatchNorm1d(layers[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_p))
        self.layers.append(nn.Linear(layers[-1], output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.softmax(x, dim=1)

# Define hyperparameter grids
lr_vec = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])

max_epoch = 200

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).long()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).long()

PK_values = []
layer_counts = list(range(11, 31))

for layer_count in layer_counts:
    layers = [100] * layer_count
    model = Model(X_train.shape[1], len(target_names), layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_vec[0])
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training model with {layer_count} layers...")

    for epoch in range(max_epoch):
        model.train()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{max_epoch}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
        PK = correct.mean().item() * 100
        PK_values.append(PK)
        print(f"PK for {layer_count} layers: {PK}%")

# Plot results
plt.plot(layer_counts, PK_values, marker='o')
plt.xlabel('Liczba warstw')
plt.ylabel('Średnie PK')
plt.title('Wykres średniego PK w zależności od liczby warstw')
plt.grid(True)
plt.savefig("Fig.1_PK_Layers_pytorch_iris.png", bbox_inches='tight')
plt.show()
