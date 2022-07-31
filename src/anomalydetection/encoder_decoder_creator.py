import tensorflow as tf
def encoder_decoder_creator(input_size, input_enc, sequential, layer, regularizer):
     
    encoder = tf.keras.Sequential([
      tf.keras.layers.Dense(neurons, activation=layer, activity_regularizer=regularizer) for neurons in sequential]+
      [tf.keras.layers.Dense(input_enc, activation=layer, activity_regularizer=regularizer)])
    
    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(neurons, activation=layer, activity_regularizer=regularizer) for neurons in sequential[::-1]]+
        [tf.keras.layers.Dense(input_size, activation="sigmoid")])
 
    return encoder, decoder
