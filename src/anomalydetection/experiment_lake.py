from typing import get_type_hints
import numpy as np 

from sklearn.neighbors import KernelDensity
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from calculate_metrics import calculate_metrics


np.random.seed(42)
torch.manual_seed(42)

class Loader(object):
    def __init__(self, features, labels, N_train, mode="train"):
        self.mode = mode
        
        outlier_data = features[labels==1]
        outlier_labels = labels[labels==1]

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
        
        self.test = np.concatenate((self.test, outlier_data),axis=0)
        self.test_labels = np.concatenate((self.test_labels, outlier_labels),axis=0)

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


class VAE(nn.Module):
    def __init__(self, dim):
        super(VAE,self).__init__()
        self.enc_1 = nn.Linear(dim,20)
        self.enc = nn.Linear(20,15)
        
        self.act = nn.Tanh()
        self.act_s = nn.Sigmoid()
        self.mu = nn.Linear(15,15)
        self.log_var = nn.Linear(15,15)
        
        self.z = nn.Linear(15,15)
        self.z_1 = nn.Linear(15,20)
        self.dec = nn.Linear(20,dim)

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


def get_loader(features, labels, batch_size, N_train, mode='train'):
    """Build and return data loader."""
    
    dataset = Loader(features, labels, N_train, mode)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False)
    return data_loader


def loss_function(recon_x, x, mu, logvar, enc, z, enc_1, z_1):
    criterion_elementwise_mean = nn.MSELoss(reduction='sum')
    BCE_x = criterion_elementwise_mean(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE_x + KLD


def relative_euclidean_distance(a, b):
    return (a-b).norm(2, dim=1) / a.norm(2, dim=1)


def experiment_lake(X_train, y_train, X_test, y_test, settings, mlflow, best=False):

    for i, setting in enumerate(settings):
        #print(f"experiment_dmkdc {i} threshold {setting['z_threshold']}")
        with mlflow.start_run(run_name=setting["z_run_name"]):

            X = np.concatenate((X_train, X_test), axis=0)
            y = np.concatenate((y_train, y_test), axis=0)

            N_train = int( (y==0).sum() * setting["z_ratio"] )
            
            vae = VAE(X_train.shape[1])
            optimizer = torch.optim.Adam(vae.parameters(), lr=setting["z_learning_rate"])
            data_loader_train = get_loader(X, y, setting["z_batch_size"], N_train, mode='train')
            data_loader_test = get_loader(X, y, setting["z_batch_size"], N_train, mode='test')

            for i in range(setting["z_iter_per_epoch"]):
                for _, (input_data, labels)  in enumerate(data_loader_train):
                    enc_1, enc, mu, log_var, o, z,  z_1, dec = vae(input_data)
                    optimizer.zero_grad()
                    loss = loss_function(dec, input_data, mu, log_var, enc, z, enc_1, z_1)
                    loss.backward()
                    optimizer.step()

            train_enc = []
            test_enc = []
            test_labels = []

            temp_loader = get_loader(X, y, N_train, N_train, mode='train')
            for _, (input_data, _)  in enumerate(temp_loader):
                enc_1, enc, mu, log_var, o, z,  z_1, dec = vae(input_data)
                rec_euclidean = relative_euclidean_distance(input_data, dec)
                rec_cosine = F.cosine_similarity(input_data, dec, dim=1)
                enc = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
                enc = enc.detach().numpy()
                train_enc.append(enc)

            for _, (input_data, labels)  in enumerate(data_loader_test):
                enc_1, enc, mu, log_var, o, z,  z_1, dec = vae(input_data)
                rec_euclidean = relative_euclidean_distance(input_data, dec)
                rec_cosine = F.cosine_similarity(input_data, dec, dim=1)
                enc = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
                enc = enc.detach().numpy()
                test_enc.append(enc)
                test_labels.append(labels.numpy())

            x = train_enc[0]
            kde = KernelDensity(kernel='gaussian', bandwidth=0.000001).fit(x)
            score = kde.score_samples(x)

            test_score = []
            for i in range(len(test_enc)):
                score = kde.score_samples(test_enc[i])
                test_score.append(score)
            test_labels = np.concatenate(test_labels,axis=0)
            test_score = np.concatenate(test_score,axis=0)


            g = np.sum(test_labels==1) / len(test_labels)

            thresh = np.percentile(test_score, int(g*100))
            pred = (test_score < thresh).astype(int)
            gt = test_labels.astype(int)

            metrics = calculate_metrics(gt, pred, test_score, setting["z_run_name"])

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)

            if best:
                np.savetxt(('artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'), pred, delimiter=',')
                mlflow.log_artifact(('artifacts/'+setting["z_name_of_experiment"]+'-preds.csv'))
                np.savetxt(('artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'), test_score, delimiter=',')
                mlflow.log_artifact(('artifacts/'+setting["z_name_of_experiment"]+'-scores.csv'))

            print(f"experiment_lake {i} ratio {setting['z_ratio']}")
            print(f"experiment_lake {i} metrics {metrics}")
