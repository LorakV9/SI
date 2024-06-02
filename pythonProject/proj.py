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
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target_names = y.unique()

# Skalowanie danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2)

class Model(nn.Module):
    def __init__(self, input_dim: object) -> object:
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

lr_vec = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7 ])
K1_vec = np.arange(1,111,15)
K2_vec = np . arange (1 ,111 ,15)
PK_2D_K1K2 = np.zeros([len(K1_vec),len(K2_vec)])
max_epoch = 200
PK_2D_K1K2_max = 0
k1_ind_max = 0
k2_ind_max = 0
X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.tensor(y_train.astype('category').cat.codes.values)).long()
X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.tensor(y_test.astype('category').cat.codes.values)).long()

for k1_ind in range(len(K1_vec)):
    for k2_ind in range(len(K2_vec)):
        model = Model(X_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_vec[0])
        loss_fn = nn.CrossEntropyLoss()
        #print(model)

        for epoch in range(max_epoch):
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)

            # Zero gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
             y_pred = model(X_test)
             correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
             PK = correct.mean().item()*100
             print("K1 {} | K2 {} | PK {}".format(str(K1_vec[k1_ind]), str(K2_vec[k2_ind]), PK))

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
ax.view_init(40, 200)
plt.savefig("Fig.2_PK_K1K2_pytorch_iris.png",bbox_inches='tight')