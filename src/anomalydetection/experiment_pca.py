import qmc.tf.layers as layers
import qmc.tf.models as models

import numpy as np
from calculate_metrics import calculate_metrics

import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler

np.random.seed(42)
tf.random.set_seed(42)


def pca_right_comps(X):
    all_pca = PCA(n_components=min(X.shape))
    all_pca.fit(X)
    variance_expl = all_pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(variance_expl)
    varsums = [ cve>0.98 for cve in cum_var_exp ]
    return varsums.index(True)


class QFeatureMapAdaptRFF(layers.QFeatureMapRFF):
    def __init__(
                 self,
                 gamma_trainable=False,
                 weights_trainable=True,
                 **kwargs
                 ):
        self.g_trainable = gamma_trainable
        self.w_trainable = weights_trainable
        super().__init__(**kwargs)

    def build(self, input_shape):
        rbf_sampler = RBFSampler(
            gamma=self.gamma,
            n_components=self.dim,
            random_state=self.random_state)
        x = np.zeros(shape=(1, self.input_dim))
        rbf_sampler.fit(x)
        self.gamma_val = tf.Variable(
            initial_value=self.gamma,
            dtype=tf.float32,
            trainable=self.g_trainable,
            name="rff_gamma")
        self.rff_weights = tf.Variable(
            initial_value=rbf_sampler.random_weights_,
            dtype=tf.float32,
            trainable=self.w_trainable,
            name="rff_weights")
        self.offset = tf.Variable(
            initial_value=rbf_sampler.random_offset_,
            dtype=tf.float32,
            trainable=self.w_trainable,
            name="offset")
        self.built = True

    def call(self, inputs):
        inps = tf.cast(inputs, tf.float32)
        vals = tf.sqrt(2 * self.gamma_val) * tf.matmul(inps, self.rff_weights) + self.offset
        vals = tf.cos(vals)
        vals = vals * tf.cast(tf.sqrt(2. / self.dim), tf.float32)
        norms = tf.linalg.norm(vals, axis=-1)
        psi = vals / tf.expand_dims(norms, axis=-1)
        return psi


class DMRFF(tf.keras.Model):
    def __init__(self,
                 dim_x,
                 num_rff,
                 gamma=1,
                 random_state=None):
        super().__init__()
        self.rff_layer = QFeatureMapAdaptRFF(input_dim=dim_x, dim=num_rff, gamma=gamma, random_state=random_state)

    def call(self, inputs):
        x1 = inputs[:, 0]
        x2 = inputs[:, 1]
        x1, x2 = tf.cast(x1,tf.float32), tf.cast(x2,tf.float32)
        phi1 = self.rff_layer(x1)
        phi2 = self.rff_layer(x2)
        dot = tf.einsum('...i,...i->...', phi1, phi2) 
        return dot

def calc_rbf(dmrff, x1, x2):
    return dmrff.predict(np.concatenate([x1[:, np.newaxis, ...], 
                                         x2[:, np.newaxis, ...]], 
                                        axis=1),
                         batch_size=256)


def gauss_kernel_arr(x, y, gamma):
    return np.exp(-gamma * np.linalg.norm(x - y, axis=1) ** 2)



def experiment_pca_dmkde(X_train, y_train, X_test, y_test, setting, mlflow, best=False):

    #for i, setting in enumerate(settings):

        with mlflow.start_run(run_name=setting["z_run_name"]):

            X = []
            for i in range(len(y_train)):
                if y_train[i] == 0: X.append(X_train[i])
            X = np.array(X)

            comps = pca_right_comps(X)
            pca = PCA(n_components=comps)
            print(f"For PCA, {comps} components will be used.")
            Xpca = pca.fit_transform(X)
            Xpcatest = pca.transform(X_test)
            setting["z_pca_components"] = comps


            dmrff = DMRFF(dim_x=comps, num_rff=setting["z_rff_components"], 
                          gamma=setting["z_gamma"], random_state=setting["z_random_state"])
            dmrff.compile(optimizer="adam", loss='mse')
            

            num_samples = setting["z_num_samples"]
            rnd_idx1 = np.random.randint(Xpca.shape[0],size=(num_samples, ))
            rnd_idx2 = np.random.randint(Xpca.shape[0],size=(num_samples, ))    
            x_train_rff = np.concatenate([Xpca[rnd_idx1][:, np.newaxis, ...], Xpca[rnd_idx2][:, np.newaxis, ...]], axis=1)

            rnd_idx1 = np.random.randint(Xpcatest.shape[0],size=(num_samples, ))
            rnd_idx2 = np.random.randint(Xpcatest.shape[0],size=(num_samples, ))
            x_test_rff = np.concatenate([Xpcatest[rnd_idx1][:, np.newaxis, ...], Xpcatest[rnd_idx2][:, np.newaxis, ...]], axis=1)

            y_train_rff = gauss_kernel_arr(x_train_rff[:, 0, ...], x_train_rff[:, 1, ...], gamma=setting["z_gamma"])
            #y_test_rff = gauss_kernel_arr(x_test_rff[:, 0, ...], x_test_rff[:, 1, ...], gamma=setting["z_gamma"])

            dmrff.fit(x_train_rff, y_train_rff, epochs=20, verbose=0)
            fm_x = dmrff.rff_layer
            #fm_x = tf.cast(fm_x, tf.float32)

            qmd = models.QMDensity(fm_x, setting["z_rff_components"])
            #qmd = models.QMDensitySGD(Xpca.shape[1], dim_x=setting["z_rff_components"], num_eig=setting["z_num_eigs"],
            #                          gamma=setting["z_gamma"], random_state=setting["z_random_state"])
            qmd.compile()
            qmd.fit(Xpca, epochs=1, batch_size=setting["z_batch_size"], verbose=1)
            
            y_scores = qmd.predict(Xpcatest)

            g = np.sum(y_test) / len(y_test)
            print("Outlier percentage", g)

            if np.allclose(setting["z_threshold"], 0.0): setting["z_threshold"] = np.percentile(y_scores, int(g*100))
            preds = (y_scores < setting["z_threshold"]).astype(int)
            metrics = calculate_metrics(y_test, preds, y_scores, setting["z_run_name"])

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)

            if best:
                mlflow.log_params({"w_best": best})
                f = open('./artifacts/'+setting["z_experiment"]+'.npz', 'w')
                np.savez('./artifacts/'+setting["z_experiment"]+'.npz', preds=preds, scores=y_scores)
                mlflow.log_artifact(('artifacts/'+setting["z_experiment"]+'.npz'))
                f.close()

            print(f"experiment_pca_dmkde {setting['z_gamma']} metrics {metrics}")
            print(f"experiment_pca_dmkde {setting['z_gamma']} threshold {setting['z_threshold']}")
