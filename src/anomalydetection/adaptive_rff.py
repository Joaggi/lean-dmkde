import tensorflow as tf
import pylab as pl
from sklearn.kernel_approximation import RBFSampler
import qmc.tf.layers as layers
import qmc.tf.models as models
import numpy as np 
from sklearn.model_selection import train_test_split



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
        print(f"build: dim {self.dim}")
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
        print(f"call: inputs {inputs.shape} rff_weights {self.rff_weights.shape}")
        vals = tf.sqrt(2 * self.gamma_val) * tf.matmul(inputs, self.rff_weights) + self.offset
        vals = tf.cos(vals)
        vals = vals * tf.sqrt(2. / self.dim)
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
        x1 = inputs[:, 0, :]
        x2 = inputs[:, 1, :]
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

def build_features(X):
    X_train, X_test = train_test_split(X)
    num_samples = 100000
    rnd_idx1 = np.random.randint(X_train.shape[0],size=(num_samples, ))
    rnd_idx2 = np.random.randint(X_train.shape[0],size=(num_samples, ))
    x_train_rff = np.concatenate([X_train[rnd_idx1][:, np.newaxis, ...], 
                              X_train[rnd_idx2][:, np.newaxis, ...]], 
                             axis=1) 
    dists = np.linalg.norm(x_train_rff[:, 0, ...] - x_train_rff[:, 1, ...], axis=1)
    print(dists.shape)
    pl.hist(dists)
    print(np.quantile(dists, 0.001))
    rnd_idx1 = np.random.randint(X_test.shape[0],size=(num_samples, ))
    rnd_idx2 = np.random.randint(X_test.shape[0],size=(num_samples, ))
    x_test_rff = np.concatenate([X_test[rnd_idx1][:, np.newaxis, ...], 
                              X_test[rnd_idx2][:, np.newaxis, ...]], 
                             axis=1) 

    return x_train_rff, x_test_rff

def build_model(setting, x_train_rff, x_test_rff):
    n_rffs = setting["z_rff_components"]
    sigma = setting["z_sigma"]
    gamma= 1/ (2*sigma**2)

    dimension=setting["z_adaptive_input_dimension"]

    print(f'Gamma: {gamma}')
    y_train_rff = gauss_kernel_arr(x_train_rff[:, 0, ...], x_train_rff[:, 1, ...], gamma=gamma)
    y_test_rff = gauss_kernel_arr(x_test_rff[:, 0, ...], x_test_rff[:, 1, ...], gamma=gamma)
    dmrff = DMRFF(dim_x=dimension, num_rff=n_rffs, gamma=gamma, random_state=0)
    dm_rbf = calc_rbf(dmrff, x_test_rff[:, 0, ...], x_test_rff[:, 1, ...])
    pl.plot(y_test_rff, dm_rbf, '.')

    polinomial_decay = tf.keras.optimizers.schedules.PolynomialDecay(setting["z_adaptive_base_lr"], \
        setting["z_adaptive_decay_steps"], setting["z_adaptive_end_lr"], power=setting["z_adaptive_power"])
    opt = tf.keras.optimizers.Adam(learning_rate=polinomial_decay)  # optimizer

    dmrff.compile(optimizer=opt, loss='mse')
    dmrff.evaluate(x_test_rff, y_test_rff, batch_size=setting["z_adaptive_batch_size"])

    dmrff.fit(x_train_rff, y_train_rff, validation_split=0.1, epochs=setting["z_adaptive_epochs"], batch_size=setting["z_adaptive_batch_size"])

    dm_rbf = calc_rbf(dmrff, x_test_rff[:, 0, ...], x_test_rff[:, 1, ...])
    pl.plot(y_test_rff, dm_rbf, '.')
    dmrff.evaluate(x_test_rff, y_test_rff, batch_size=setting["z_adaptive_batch_size"])

    return dmrff.rff_layer

def fit_transform(setting, X):
    x_train_rff, x_test_rff = build_features(X)

    return build_model(setting, x_train_rff, x_test_rff)
