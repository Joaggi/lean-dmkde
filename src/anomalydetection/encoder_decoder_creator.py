import tensorflow as tf

from dense_tied import DenseTied
from uncorrelated_features_constraint import UncorrelatedFeaturesConstraint
from weights_orthogonality_constraint import WeightsOrthogonalityConstraint
from keras.constraints import UnitNorm, Constraint
 
def encoder_decoder_creator(autoencoder_type, input_size, input_enc, sequential, layer, 
                            activity_regularizer_name=None,  activity_regularizer_value=None,
                            kernel_regularizer=None, kernel_constraint=None):

    print(f"activity_regularizer_name: {activity_regularizer_name}, activity_regularizer_value: {activity_regularizer_value}, kernel_regularizer: {kernel_regularizer}, kernel_constraint: {kernel_constraint}")

    if autoencoder_type == None or autoencoder_type == "unconstrained":
        return unconstrained_autoencoder(autoencoder_type, input_size, input_enc, sequential, layer,
                activity_regularizer_name, activity_regularizer_value)

    elif autoencoder_type == "tied":
        return tied_encoder_decoder_creator(autoencoder_type, input_size, input_enc, sequential, layer,
                activity_regularizer_name, activity_regularizer_value,
                 kernel_regularizer, kernel_constraint)

def tied_encoder_decoder_creator(autoencoder_type, input_size, input_enc, sequential, layer,
                                 activity_regularizer_name=None, activity_regularizer_value=None,
                            kernel_regularizer=None, kernel_constraint_name=None):

    kernel_constraint = get_kernel_contraint(kernel_constraint_name)
    
    encoder_full = tf.keras.Sequential([
      tf.keras.layers.Dense(neurons, 
                activation=layer, 
                activity_regularizer=get_activity_regularizer(autoencoder_type, activity_regularizer_name, activity_regularizer_value, neurons=neurons),
                kernel_regularizer=get_kernel_regularizer(neurons, weightage=1., 
                                                           kernel_regularizer=kernel_regularizer),
                kernel_constraint=kernel_constraint,
                use_bias=True) 
        for neurons in list(sequential) + [input_enc]])


    decoder_tied = tf.keras.Sequential([ 
        DenseTied(neurons, activation=layer, 
                tied_to=tied_layer, 
                use_bias=False) 
                    for (neurons, tied_layer) in zip(([input_size] + list(sequential)  
                                                      )[::-1] , encoder_full.layers[::-1])]
                                       )

    decoder_tied.layers[0].activation=tf.keras.activations.sigmoid

    return encoder_full, decoder_tied
 
def unconstrained_autoencoder(autoencoder_type, input_size, input_enc, sequential, layer,
                              activity_regularizer_name, activity_regularizer_value):
    activity_regularizer = get_activity_regularizer(
            autoencoder_type, activity_regularizer_name, activity_regularizer_value )

    encoder = tf.keras.Sequential([
      tf.keras.layers.Dense(neurons, activation=layer, activity_regularizer=activity_regularizer)
           for neurons in list(sequential) + [input_enc]]
      )
    
    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(neurons, activation=layer, activity_regularizer=activity_regularizer)
            for neurons in sequential[::-1]]+
        [tf.keras.layers.Dense(input_size, activation="sigmoid")])
 
    return encoder, decoder

def get_kernel_contraint(kernel_constraint_name):
    if kernel_constraint_name == None or kernel_constraint_name == "None":
        return None
    elif kernel_constraint_name == "unit_norm":
        return UnitNorm(axis=0) 
    else:
        return None


def get_kernel_regularizer(neurons, weightage, kernel_regularizer):
    if kernel_regularizer == None or kernel_regularizer == "None":
        return None
    if kernel_regularizer == "weights_orthogonality":
        return WeightsOrthogonalityConstraint(neurons, weightage=weightage, axis=0),
    else:
        return kernel_regularizer

def get_activity_regularizer(autoencoder_type, activity_regularizer_name, activity_regularizer_value, neurons=None):
    if activity_regularizer_name == None or activity_regularizer_name == "None": 
        return None
    elif activity_regularizer_name == "uncorrelated_features" and autoencoder_type == "tied":
        return UncorrelatedFeaturesConstraint(neurons, weightage = activity_regularizer_value)
    elif activity_regularizer_name == "l1": 
        return tf.keras.regularizers.l1(activity_regularizer_value)
    elif activity_regularizer_name == "l2":
        return tf.keras.regularizers.l2(activity_regularizer_value)
    else:
        return None


