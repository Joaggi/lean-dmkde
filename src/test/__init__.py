#%% md

# **LAKE vs. QMC**

**Comparison between LAKE method and QMC Density Estimation for Anomaly Detection**


#%%

%load_ext autoreload
%autoreload 2

#%% md

# Import

#%%

import numpy as np
import pandas as pd 

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
from sklearn.metrics import classification_report

#%% md

# Data load

Here we will use the dataset *Thyroid*, available at Github. This dataset has 3772 instances, with a percentage of outliers near to 2.5%.

#%% md

## Load data from drive and csv

#%%

!gdown --id 1S6ChcBukeOXrayQ2_emBWe6tlZvOhtMC

#%%

df_train = pd.read_csv('thyroid-train.csv')
#df_test = pd.read_csv('norm-ann-test.csv')

#%% md

## Data preprocessing

#%%

df_train.loc[df_train["class"] != 1, 'class'] = 0
df_train.loc[df_train["class"] == 1, 'class'] = 1

#df_test.loc[df_test["class"] != 1, 'class'] = 0
#df_test.loc[df_test["class"] == 1, 'class'] = 1

#%%

proportions = df_train["class"].value_counts()
print(proportions)

#%%

df_train.sort_values(by="class")

#%%

df_train_norm = df_train[df_train["class"] == 0]
print(df_train_norm.shape)

df_train_outlier = df_train[df_train["class"] == 1]
print(df_train_outlier.shape)

#%%

np.save("Thyroid", df_train)

data = np.load("Thyroid.npy") 


#%%

from google.colab import drive
drive.mount('/content/drive/')
%cd "/content/drive/MyDrive/Academico/doctorado_programacion/notebooks/2021 3 Anomaly Detection"

#%%

features = data[:,:-1]
from min_max_scaler import min_max_scaler

scale, features = min_max_scaler(features)

print(len(features[0]))
print(features)

#%%

labels = data[:,-1]
labels = labels[:,np.newaxis] 
print(labels)

data = np.concatenate((features, labels),axis=1)

np.save('Thyroid', data)

#%% md

# Experimental design

#%% md

## LAKE algorithm

We will reply, step by step, the code available at https://github.com/1246170471/LAKE/blob/master/Thyroid/LAKE.ipynb in order to replicate the results of the paper.

#%% md

### Imports for lake algorithm

#%%

import numpy as np 
import pandas as pd
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

#%% md

### Classes and functions

#%%

class ThyroidLoader(object):
    def __init__(self, data_path, N_train, mode="train"):
        self.mode=mode
        data = np.load(data_path)

        labels = data[:,-1]
        features = data[:,:-1]
        N, D = features.shape
        
        normal_data = features[labels==1]
        normal_labels = labels[labels==1]

        N_normal = normal_data.shape[0]

        attack_data = features[labels==0]
        attack_labels = labels[labels==0]

        N_attack = attack_data.shape[0]

        randIdx = np.arange(N_attack)
        np.random.shuffle(randIdx)
        self.N_train = N_train
        self.train = attack_data[randIdx[:self.N_train]]
        self.train_labels = attack_labels[randIdx[:self.N_train]]
        
        self.test = attack_data[randIdx[self.N_train:]]
        self.test_labels = attack_labels[randIdx[self.N_train:]]
        
        self.test = np.concatenate((self.test, normal_data),axis=0)
        self.test_labels = np.concatenate((self.test_labels, normal_labels),axis=0)


    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return self.train.shape[0]
        else:
            return self.test.shape[0]


    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
            return np.float32(self.test[index]), np.float32(self.test_labels[index])

#%%

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        self.enc_1 = nn.Linear(36,20)
        self.enc = nn.Linear(20,11)
        
        self.act = nn.Tanh()
        self.act_s = nn.Sigmoid()
        self.mu = nn.Linear(11,10)
        self.log_var = nn.Linear(11,10)
        
        self.z = nn.Linear(10,11)
        self.z_1 = nn.Linear(11,20)
        self.dec = nn.Linear(20,36)
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self,x):
        enc_1 = self.enc_1(x)
        enc = self.act(enc_1)
        enc = self.enc(enc)
        enc = self.act(enc)
        
        mu = self.mu(enc)
        log_var = self.log_var(enc)
        o = self.reparameterize(mu,log_var)
        z = self.z(o)
        z_1 = self.act(z)
        z_1 = self.z_1(z_1)
        dec = self.act(z_1)
        dec = self.dec(dec)
        dec = self.act_s(dec)
        return enc_1, enc, mu, log_var, o, z, z_1, dec

#%%

def get_loader(data_path, batch_size, N_train, mode='train'):
    """Build and return data loader."""
    
    dataset = ThyroidLoader(data_path, N_train, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader

#%%

def loss_function(recon_x, x, mu, logvar, enc, z,  enc_1, z_1):
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    criterion_elementwise_mean = nn.MSELoss(reduction='sum')
    BCE_x = criterion_elementwise_mean(recon_x,x)
    BCE_z = criterion_elementwise_mean(enc,z)
    BCE_z_1 = criterion_elementwise_mean(enc_1,z_1)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE_x + KLD


def relative_euclidean_distance(a, b):
    return (a-b).norm(2, dim=1) / a.norm(2, dim=1)


#%% md

### Running lake algorithm

#%%

data_path = 'Thyroid.npy'

batch_size = 200
learn_rate = 0.0001
All_train = 169307

#%%

Ratio = 0.1
iter_per_epoch = 500
Average_cycle = 4
result = []
diff_quantity_result= []
for i in range(1):
    N_train = int(All_train*Ratio*(8))
    result = []
    print(Ratio*(8))
    for i in range(Average_cycle):
        vae = VAE()
        optimizer = torch.optim.Adam(vae.parameters(),lr=learn_rate)
        data_loader_train = get_loader(data_path, batch_size, N_train, mode='train')
        for i in range(iter_per_epoch):
            for j ,(input_data, labels)  in enumerate(data_loader_train):
                enc_1, enc, mu, log_var, o, z,  z_1, dec = vae(input_data)
                optimizer.zero_grad()
                loss = loss_function(dec, input_data, mu, log_var, enc, z, enc_1, z_1)
                loss.backward()
                optimizer.step()
        
        batch_size = 1000
        data_loader_train = get_loader(data_path, batch_size, N_train,mode='train')
        train_enc = []
        train_labels = []
        data_loader_test = get_loader(data_path, batch_size, N_train, mode='test')
        test_enc = []
        test_labels = []
        
        for i ,(input_data, labels)  in enumerate(data_loader_train):
            enc_1, enc, mu, log_var, o, z,  z_1, dec = vae(input_data)
            rec_euclidean = relative_euclidean_distance(input_data, dec)
            rec_cosine = F.cosine_similarity(input_data, dec, dim=1)
    
            enc = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
            #enc = torch.cat([enc, rec_euclidean.unsqueeze(-1)], dim=1)
            enc = enc.detach().numpy()

            train_enc.append(enc)
        for i ,(input_data, labels)  in enumerate(data_loader_test):
            enc_1, enc, mu, log_var, o, z,  z_1, dec = vae(input_data)
            rec_euclidean = relative_euclidean_distance(input_data, dec)
            rec_cosine = F.cosine_similarity(input_data, dec, dim=1)
    
            enc = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
            #enc = torch.cat([enc, rec_euclidean.unsqueeze(-1)], dim=1)
            enc = enc.detach().numpy()

            test_enc.append(enc)
    
            test_labels.append(labels.numpy())
        x =train_enc[0] 
        kde = KernelDensity(kernel='gaussian', bandwidth=0.00001).fit(x)
        score =  kde.score_samples(x)
        k = len(test_enc)
        test_score = []
        for i in range (k):
            score = kde.score_samples(test_enc[i])
            test_score.append(score)
        test_labels = np.concatenate(test_labels,axis=0)
        print(test_labels)
        test_score = np.concatenate(test_score,axis=0)

        print(test_score.shape)

        s = len(test_labels)
        c = np.sum(test_labels==1)
        g = c/s
        
        thresh = np.percentile(test_score, int(g*100))
        pred = (test_score < thresh).astype(int)
        gt = test_labels.astype(int)
        accuracy = accuracy_score(gt,pred)
        precision, recall, f_score, support = prf(gt, pred, average='binary')
        print(classification_report(gt, pred, digits=4))
        temp_result = [accuracy,precision,recall,f_score]
        result.append(temp_result)
    end_result = np.mean(result,axis=0)
    diff_quantity_result.append(end_result)
    print(end_result)

#%% md

## OneClassSVM algorithm 

This is one of the methods proposed in the paper to compare with LAKE. In order to use this method, we will use the implementation provided by Scikit-Learn. The hyperparameters of this algorithm will be the same that the paper stablishes.


#%%

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

#%% md

#### Preprocessing

#%%

features = data[:,:-1]

scaler = MinMaxScaler()
scaler.fit(features)
features =  scaler.transform(features)

labels = data[:,-1]

print(labels)

#%%

#nu: anomaly proportion, in this case 0.05
nu = 0.05
#gamma: 1/m, being m the number of features (36)
gamma = 1/36

#%% md

### Running

#%%

clf = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
clf.fit(features)

#%%

pred1 = clf.predict(features)

pred1[pred1 == 1] = 0
pred1[pred1 == -1] = 1

print(labels == pred1)

#%%

accuracy = accuracy_score(labels, pred1)
precision, recall, f_score, support = prf(labels, pred1, average='binary')
print(accuracy, precision, recall, f_score)

#%%

print(classification_report(labels, pred1, digits=4))

#%% md

## Isolation Forest

To use this method, we will use the implementation provided by Scikit-Learn, based on the proposal in https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html#sphx-glr-auto-examples-ensemble-plot-isolation-forest-py

#%%

clf = IsolationForest(max_samples=10, random_state=42, 
                      contamination=nu)
clf.fit(features)

#%%

pred2 = clf.predict(features)

pred2[pred2 == 1] = 0
pred2[pred2 == -1] = 1

print(labels == pred2)

#%%

accuracy = accuracy_score(labels, pred2)
precision, recall, f_score, support = prf(labels, pred2, average='binary')
print(accuracy, precision, recall, f_score)

#%%

print(classification_report(labels, pred2, digits=4))

#%% md

## Robust Covariance (Elliptic Envelope)

To use this method, we will use the implementation provided by Scikit-Learn. This method assumes the data is Gaussian and learns an ellipse. It thus degrades when the data is not unimodal.

#%%

clf = EllipticEnvelope(contamination=nu)

clf.fit(features)

#%%

pred3 = clf.predict(features)

pred3[pred3 == 1] = 0
pred3[pred3 == -1] = 1

print(labels == pred3)

#%%

accuracy = accuracy_score(labels, pred3)
precision, recall, f_score, support = prf(labels, pred3, average='binary')
print(accuracy, precision, recall, f_score)

#%%

print(classification_report(labels, pred3, digits=4))

#%% md

## Local Outlier Factor

To use this method, we will use the implementation provided by Scikit-Learn. LocalOutlierFactor is intended only for outlier detection.

#%%

#this determines how many neighbors the model needs to look
nearest = 5
 
clf = LocalOutlierFactor(n_neighbors=nearest, contamination=nu)

#%%

pred4 = clf.fit_predict(features)

pred4[pred4 == 1] = 0
pred4[pred4 == -1] = 1

print(labels == pred4)

#%%

accuracy = accuracy_score(labels, pred4)
precision, recall, f_score, support = prf(labels, pred4, average='binary')
print(accuracy, precision, recall, f_score)

#%%

print(classification_report(labels, pred4, digits=4))

#%% md

## DMKDE Density Estimation

In this step, we'll try to use models from qmc to find an estimator for the density function of the train data. Then we'll calculate a threshold and compare it against the total data.

#%% md

#### Import qmc library

#%%

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
    !pip install git+https://github.com/fagonzalezo/qmc.git
else:
    import sys
    sys.path.insert(0, "../")

#%%

from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import qmc.tf.layers as layers
import qmc.tf.models as models

#%% md

#### Constants definition of DMKDE



#%%

def calculate_constant_qmkde(gamma=1, dimension = 1):
  sigma = (4*gamma)**(-1/2)
  coefficient = 1 /  (2*np.pi*sigma**2)**(dimension/2)
  return coefficient

#%% md

###  Histogram

#%%

from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

#%%

X_train_original = df_train_norm.drop(columns=["class"]).to_numpy()

print(X_train_original.shape)

#%%

scaler = MinMaxScaler()
scaler.fit(X_train_original)
X_train_scaled =  scaler.transform(X_train_original)

labels_train = df_train_norm["class"].to_numpy()

print(labels_train)

#%%

X_test_original = df_train_outlier.drop(columns=["class"]).to_numpy()
X_test_scaled =  scaler.transform(X_test_original)

labels_test = df_train_outlier["class"].to_numpy()

print(labels_test)

#%%

distances_X = euclidean_distances(X_train_scaled, X_train_scaled)
plt.axes(frameon = 0)
plt.grid()
plt.title('Histogram of distances')
plt.hist(distances_X[np.triu_indices_from(distances_X, k=1)].ravel(), density = True, bins=40);

#%%

distances_X = euclidean_distances(X_test_scaled, X_test_scaled)
plt.axes(frameon = 0)
plt.grid()
plt.title('Histogram of distances')
plt.hist(distances_X[np.triu_indices_from(distances_X, k=1)].ravel(), density = True, bins=40);

#%%

X_train_scaled.max(), X_train_scaled.min()

#%% md

### Novelty Detection: training with 'normal' data

We'll use two different models from `qmc`: QMDensity and QMDensitySGD.

#%% md

#### QMDensity

#%%

setting = {
  "dimensions": 36,
  "rff_dimensions": 1000
}

#%%

sigma = 0.3
gamma = 1 / (2 * sigma**2)

#%%

X_train = X_train_scaled
X_test = X_test_scaled

#%%

fm_x = layers.QFeatureMapRFF(setting["dimensions"], dim=setting["rff_dimensions"], gamma=gamma, random_state=17)
qmd = models.QMDensity(fm_x, setting["rff_dimensions"])

qmd.compile()
qmd.fit(X_train, epochs=1)

#%%

# preds = calculate_constant_qmkde(gamma, dimension=36) * qmd.predict(X_train)
preds_train = qmd.predict(X_train)
preds_test = qmd.predict(X_test)

#%%

thresh = np.percentile(preds_train, 5)
#thresh = preds.max() * .975
print(thresh)

#%%

pred1 = (preds_test < thresh).astype(int)
pred2 = (preds_train < thresh).astype(int)
gt = np.concatenate((labels_train, labels_test))

pred = np.concatenate((pred1, pred2))

accuracy = accuracy_score(gt,pred)
precision, recall, f_score, support = prf(gt, pred)

result = [accuracy,precision,recall,f_score]
print(np.sum(pred))
print(np.sum(gt))
print(accuracy)

#%%

print(result)

#%%

print(np.unique(pred, return_counts= True))
print(np.unique(gt, return_counts= True))

#%%

from sklearn.metrics import classification_report
print(classification_report(gt, pred, digits=4))

#%% md

#### QMDensity with SGD

#%%

import matplotlib.pyplot as plt

qmd2 = models.QMDensitySGD(36, 1000, num_eig=25, gamma=gamma, random_state=17)
eig_vals = qmd2.set_rho(qmd.weights[2])

plt.plot(eig_vals[-25:], '*-')
tf.reduce_sum(eig_vals)

#%%

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0018)
qmd3 = models.QMDensitySGD(36, 1000, num_eig=23, gamma=gamma, random_state=17)
qmd3.compile(optimizer)
eig_vals = qmd3.set_rho(qmd.weights[2])
qmd3.fit(X_train, epochs=54)

#%%

preds_train = qmd3.predict(X_train)
preds_test = qmd.predict(X_test)

#%%

#preds3 = calculate_constant_qmkde(gamma, dimension=36) * qmd3.predict(X_train)

thresh3 = np.percentile(preds_train, 2.5)
#thresh3 = preds3.max() * .975
print(thresh3)

#%%

pred1 = (preds_test < thresh3).astype(int)
pred2 = (preds_train < thresh3).astype(int)
gt = np.concatenate((labels_train, labels_test))

pred3 = np.concatenate((pred1, pred2))

accuracy = accuracy_score(gt, pred3)
precision, recall, f_score, support = prf(gt, pred3, average='binary')

result = [accuracy,precision,recall,f_score]
print(np.sum(pred3))
print(np.sum(gt))
print(accuracy)

#%%

print(result)

#%%

print(np.unique(pred, return_counts= True))
print(np.unique(gt, return_counts= True))

#%%

from sklearn.metrics import classification_report
print(classification_report(gt, pred3, digits=4))

#%% md

### Outlier Detection: training with all data

We'll use two different models from `qmc`: QMDensity and QMDensitySGD.

#%% md

#### QMDensity

#%%

X_total = np.concatenate((X_train, X_test))

#%%

fm_x = layers.QFeatureMapRFF(setting["dimensions"], dim=setting["rff_dimensions"], gamma=gamma, random_state=17)
qmd = models.QMDensity(fm_x, setting["rff_dimensions"])

qmd.compile()
qmd.fit(X_total, epochs=1)

#%%

# preds = calculate_constant_qmkde(gamma, dimension=36) * qmd.predict(X_train)
preds_train = qmd.predict(X_train)
preds_test = qmd.predict(X_test)

#%%

thresh = np.percentile(preds_train, 5)
#thresh = preds.max() * .975
print(thresh)

#%%

pred1 = (preds_test < thresh).astype(int)
pred2 = (preds_train < thresh).astype(int)
gt = np.concatenate((labels_train, labels_test))

pred = np.concatenate((pred1, pred2))

accuracy = accuracy_score(gt,pred)
precision, recall, f_score, support = prf(gt, pred)

result = [accuracy,precision,recall,f_score]
print(np.sum(pred))
print(np.sum(gt))
print(accuracy)

#%%

print(result)

#%%

print(np.unique(pred, return_counts= True))
print(np.unique(gt, return_counts= True))

#%%

from sklearn.metrics import classification_report
print(classification_report(gt, pred, digits=4))

#%% md

#### QMDensity with SGD

#%%

import matplotlib.pyplot as plt

qmd2 = models.QMDensitySGD(36, 1000, num_eig=25, gamma=gamma, random_state=17)
eig_vals = qmd2.set_rho(qmd.weights[2])

plt.plot(eig_vals[-25:], '*-')
tf.reduce_sum(eig_vals)

#%%

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0018)
qmd3 = models.QMDensitySGD(36, 1000, num_eig=11, gamma=gamma, random_state=17)
qmd3.compile(optimizer)
eig_vals = qmd3.set_rho(qmd.weights[2])
qmd3.fit(X_train, epochs=54)

#%%

preds_train = qmd3.predict(X_train)
preds_test = qmd.predict(X_test)

#%%

#preds3 = calculate_constant_qmkde(gamma, dimension=36) * qmd3.predict(X_train)

thresh3 = np.percentile(preds_train, 2.5)
#thresh3 = preds3.max() * .975
print(thresh3)

#%%

pred1 = (preds_test < thresh3).astype(int)
pred2 = (preds_train < thresh3).astype(int)
gt = np.concatenate((labels_train, labels_test))

pred3 = np.concatenate((pred1, pred2))

accuracy = accuracy_score(gt, pred3)
precision, recall, f_score, support = prf(gt, pred3, average='binary')

result = [accuracy,precision,recall,f_score]
print(np.sum(pred3))
print(np.sum(gt))
print(accuracy)

#%%

print(result)

#%%

print(np.unique(pred, return_counts= True))
print(np.unique(gt, return_counts= True))

#%%

from sklearn.metrics import classification_report
print(classification_report(gt, pred3, digits=4))

