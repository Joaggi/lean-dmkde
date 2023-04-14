import tensorflow as tf
#import pylab as pl
from sklearn.kernel_approximation import RBFSampler
import qmc.tf.layers as layers
import qmc.tf.models as models
import numpy as np 
from sklearn.model_selection import train_test_split

from aff.dmrff import DMRFF 
from aff.calc_rbf import calc_rbf 
from aff.gauss_kernel_arr import gauss_kernel_arr 

def build_model(setting, x_train_rff, x_test_rff):
    n_rffs = setting["z_rff_components"]
    sigma = setting["z_sigma"]
    gamma= 1/ (2*sigma**2)

    if "z_enable_reconstruction_metrics" in setting: 
       dimension=setting["z_adaptive_input_dimension"]+2
    else:
       dimension=setting["z_adaptive_input_dimension"]

    print(f'Gamma: {gamma}')
    y_train_rff = gauss_kernel_arr(x_train_rff[:, 0, ...], x_train_rff[:, 1, ...], gamma=gamma)
    y_test_rff = gauss_kernel_arr(x_test_rff[:, 0, ...], x_test_rff[:, 1, ...], gamma=gamma)
    dmrff = DMRFF(dim_x=dimension, num_rff=n_rffs, gamma=gamma, random_state=0)
    dm_rbf = calc_rbf(dmrff, x_test_rff[:, 0, ...], x_test_rff[:, 1, ...])
    #pl.plot(y_test_rff, dm_rbf, '.')

    polinomial_decay = tf.keras.optimizers.schedules.PolynomialDecay(setting["z_adaptive_base_lr"], \
        setting["z_adaptive_decay_steps"], setting["z_adaptive_end_lr"], power=setting["z_adaptive_power"])
    opt = tf.keras.optimizers.Adam(learning_rate=polinomial_decay)  # optimizer

    dmrff.compile(optimizer=opt, loss='mse')
    dmrff.evaluate(x_test_rff, y_test_rff, batch_size=setting["z_adaptive_batch_size"])

    validation_split = 0.2 if setting["z_best"] == False else 0.001

    history = dmrff.fit(x_train_rff, y_train_rff, validation_split=validation_split, verbose=setting["z_verbose"],
                        epochs=setting["z_adaptive_epochs"], batch_size=setting["z_adaptive_batch_size"])
    
    dm_rbf = calc_rbf(dmrff, x_test_rff[:, 0, ...], x_test_rff[:, 1, ...])
    #pl.plot(y_test_rff, dm_rbf, '.')
    print(dmrff.evaluate(x_test_rff, y_test_rff, batch_size=setting["z_adaptive_batch_size"]))

    return dmrff.rff_layer, history


