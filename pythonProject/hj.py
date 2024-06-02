import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
plt.style.use('ggplot')

# Wczytanie danych z pliku 'iris.data' do obiektu DataFrame z pandas
data = pd.read_csv('iris.data', header=None)
X = data.iloc[:, :-1]  # Wszystkie kolumny oprócz ostatniej jako cechy
y = data.iloc[:, -1]   # Ostatnia kolumna jako etykiety

# Ustalenie nazw kolumn i klas
feature_names = ['sepal_długość (cm)', 'sepal_szerokość(cm)', 'petal_długość(cm)', 'petal_szerokość(cm)']
target_names = y.unique()

# Wyświetlenie oryginalnych danych
print("Oryginalne dane:")
print(X.describe())

# Wykresy nieskalowanych danych
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for target, target_name in enumerate(target_names):
    X_plot = X[y == target_name]
    ax1.plot(X_plot.iloc[:, 0], X_plot.iloc[:, 1],
             linestyle='none',
             marker='o',
             label=target_name)
ax1.set_xlabel(feature_names[0])
ax1.set_ylabel(feature_names[1])
ax1.axis('equal')
ax1.legend()

for target, target_name in enumerate(target_names):
    X_plot = X[y == target_name]
    ax2.plot(X_plot.iloc[:, 2], X_plot.iloc[:, 3],
             linestyle='none',
             marker='o',
             label=target_name)
ax2.set_xlabel(feature_names[2])
ax2.set_ylabel(feature_names[3])
ax2.axis('equal')
ax2.legend()

plt.show()

# Skalowanie danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Wyświetlenie skalowanych danych
print("Skalowane dane:")
print(pd.DataFrame(X_scaled, columns=feature_names).describe())

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2)

class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = Model(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 100
X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.tensor(y_train.astype('category').cat.codes.values)).long()
X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.tensor(y_test.astype('category').cat.codes.values)).long()

loss_list = np.zeros((EPOCHS,))
accuracy_list = np.zeros((EPOCHS,))

for epoch in range(EPOCHS):  # Użycie range zamiast tqdm.trange
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss_list[epoch] = loss.item()

    # Zero gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        y_pred = model(X_test)
        correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list[epoch] = correct.mean()

fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)

ax1.plot(accuracy_list)
ax1.set_ylabel("Dokładność walidacji")
ax2.plot(loss_list)
ax2.set_ylabel("Utrata walidacji")
ax2.set_xlabel("epochs")

# Wyświetlenie danych
print(data)

plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], 'k--')


enc = OneHotEncoder()
Y_onehot = enc.fit_transform(y_test[:, np.newaxis]).toarray()

with torch.no_grad():
    y_pred = model(X_test).numpy()
    fpr, tpr, threshold = roc_curve(Y_onehot.ravel(), y_pred.ravel())

plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc(fpr, tpr)))
plt.xlabel('Fałszywie dodatnie')
plt.ylabel('Prawdziwie dodatnie')
plt.title('ROC krzywa')
plt.legend()

# Wykresy skalowanych danych
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for target, target_name in enumerate(target_names):
    X_plot = X_scaled[y == target_name]
    ax1.plot(X_plot[:, 0], X_plot[:, 1],
             linestyle='none',
             marker='o',
             label=target_name)
ax1.set_xlabel(feature_names[0])
ax1.set_ylabel(feature_names[1])
ax1.axis('equal')
ax1.legend()

for target, target_name in enumerate(target_names):
    X_plot = X_scaled[y == target_name]
    ax2.plot(X_plot[:, 2], X_plot[:, 3],
             linestyle='none',
             marker='o',
             label=target_name)
ax2.set_xlabel(feature_names[2])
ax2.set_ylabel(feature_names[3])
ax2.axis('equal')
ax2.legend()

plt.show()
