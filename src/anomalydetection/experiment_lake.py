import numpy as np 
import ast
from sklearn.neighbors import KernelDensity
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from calculate_metrics import calculate_metrics


np.random.seed(42)
torch.manual_seed(42)

class Loader(object):
    def __init__(self, features, labels, N_train=0.3, mode="train"):
        self.mode = mode
        
        test_data = features[1]
        test_labels = labels[1]

        train_data = features[0]
        train_labels = labels[0]

        N_attack = train_data.shape[0]
        
        self.N_train = N_attack
        self.train = train_data
        self.train_labels = train_labels
        
        self.test = test_data
        self.test_labels = test_labels
        

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
    def __init__(self, dim, sz):
        super(VAE,self).__init__()
        self.enc_1 = nn.Linear(dim,sz[0])
        self.enc = nn.Linear(sz[0],sz[1])
        
        self.act = nn.Tanh()
        self.act_s = nn.Sigmoid()
        self.mu = nn.Linear(sz[1],sz[2])
        self.log_var = nn.Linear(sz[1],sz[2])
        
        self.z = nn.Linear(sz[2],sz[1])
        self.z_1 = nn.Linear(sz[1],sz[0])
        self.dec = nn.Linear(sz[0],dim)

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


def experiment_lake(X_train, y_train, X_test, y_test, setting, mlflow, best=False):

    #for i, setting in enumerate(settings):
        #print(f"experiment_dmkdc {i} threshold {setting['z_threshold']}")
        with mlflow.start_run(run_name=setting["z_run_name"]):

            X = [X_train, X_test]
            y = [y_train, y_test]

            N_train = 0.3 #int( (y==0).sum() * setting["z_ratio"] )
            
            setting["z_enc_dec"] = ast.literal_eval(setting["z_enc_dec"])
            setting["z_batch_size"] = int(setting["z_batch_size"])
            vae = VAE(X_train.shape[1], setting["z_enc_dec"])
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

            temp_loader = get_loader(X, y, X_train.shape[0], N_train, mode='train')
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
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            #print(np.sum(np.isinf(x), axis=0))
            kde = KernelDensity(kernel='gaussian', bandwidth=0.00001).fit(x)
            score = kde.score_samples(x)

            test_score = []
            for i in range(len(test_enc)):
                testenc = np.nan_to_num(test_enc[i], nan=0.0, posinf=1.0, neginf=-1.0)
                score = kde.score_samples(testenc)
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
                mlflow.log_params({"w_best": best})
                f = open('./artifacts/'+setting["z_experiment"]+'.npz', 'w')
                np.savez('./artifacts/'+setting["z_experiment"]+'.npz', preds=pred, scores=test_score)
                mlflow.log_artifact(('artifacts/'+setting["z_experiment"]+'.npz'))
                f.close()

            print(f"experiment_lake: lr {setting['z_learning_rate']} metrics {metrics}")

