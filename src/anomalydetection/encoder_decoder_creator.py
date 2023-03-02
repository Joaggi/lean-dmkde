import tensorflow as tf

from dense_tied import DenseTied
from uncorrelated_features_constraint import UncorrelatedFeaturesConstraint
from weights_orthogonality_constraint import WeightsOrthogonalityConstraint
from keras.constraints import UnitNorm, Constraint
 
def encoder_decoder_creator(autoencoder_type, input_size, input_enc, sequential, layer, 
                            activity_regularizer=None,  activity_regularizer_value=None,
                            kernel_regularizer=None, kernel_constraint=None):

    if autoencoder_type == None or autoencoder_type == "unconstrained":
        return unconstrained_autoencoder(input_size, input_enc, sequential, layer,
                activity_regularizer, activity_regularizer_value,
                 kernel_regularizer, kernel_constraint)

    elif autoencoder_type == "tied":
        return tied_encoder_decoder_creator(input_size, input_enc, sequential, layer,
                activity_regularizer, activity_regularizer_value)

def tied_encoder_decoder_creator(input_size, input_enc, sequential, layer,
                                 activity_regularizer=None, activity_regularizer_value=None,
                            kernel_regularizer=None, kernel_constraint=None):


    if kernel_constraint == "unit_norm":
        kernel_constraint=UnitNorm(axis=0) 

    encoder_full = tf.keras.Sequential([
      tf.keras.layers.Dense(neurons, 
                activation=layer, 
                activity_regularizer=get_activity_regularizer(neurons, 1.0, activity_regularizer),
                kernel_regularizer=get_kernel_regularizer(neurons, weightage=1., 
                                                           regularizer=kernel_regularizer),
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
 
def unconstrained_autoencoder(input_size, input_enc, sequential, layer,
                              activity_regularizer_name, activity_regularizer_value):
   
    if activity_regularizer_name == "l1": 
        regularizer = tf.keras.regularizers.l1(activity_regularizer_value)
    elif activity_regularizer_name == "l2":
        regularizer = tf.keras.regularizers.l2(activity_regularizer_value)
    else: regularizer = None

    encoder = tf.keras.Sequential([
      tf.keras.layers.Dense(neurons, activation=layer, activity_regularizer=activity_regularizer)
           for neurons in sequential]+
      [tf.keras.layers.Dense(input_enc, activation=layer, activity_regularizer=activity_regularizer)])
    
    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(neurons, activation=layer, activity_regularizer=activity_regularizer)
            for neurons in sequential[::-1]]+
        [tf.keras.layers.Dense(input_size, activation="sigmoid")])
 
    return encoder, decoder


def get_kernel_regularizer(neurons, weightage, regularizer):
    if regularizer == "weights_orthogonality":
        return WeightsOrthogonalityConstraint(neurons, weightage=weightage, axis=0),
    elif regularizer != "None":
        return regularizer
    else: 
        return None

def get_activity_regularizer(neurons, weightage, regularizer):
    if regularizer == "uncorrelated_features":
        return UncorrelatedFeaturesConstraint(neurons, weightage = weightage)
    elif regularizer != "None":
        return regularizer
    else: 
        return None

