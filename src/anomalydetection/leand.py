from tensorflow import keras
import qmc.tf.layers as layers
import qmc.tf.models as models
import tensorflow as tf
from aff.q_feature_map_adapt_rff import QFeatureMapAdaptRFF

class Leand(keras.Model):
  """
    A Quantum Anomaly Detection Density Matrix Adaptive Autoencoer.
    Arguments:
        input_enc: dimension of the encoder
        dim_x: dimension of the input quantum feature map
        dim_y: dimension of the output representation
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x * dim_y
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
  """
  def __init__(self, input_size, input_enc, dim_x, num_eig=0, gamma=1, alpha=1, encoder = None, decoder = None, 
          layer=tf.keras.layers.LeakyReLU(), \
        enable_reconstruction_metrics = True, random_state=None):
    super(Leand, self).__init__()
    self.alpha = alpha
    #regularizer = None
    regularizer = tf.keras.regularizers.l1(10e-5)
    self.enable_reconstruction_metrics = enable_reconstruction_metrics 

    if encoder == None:
        self.encoder = tf.keras.Sequential([
          keras.layers.Dense(64, activation=layer, activity_regularizer=regularizer),
          keras.layers.Dense(32, activation=layer, activity_regularizer=regularizer),
          keras.layers.Dense(input_enc, activation=layer, activity_regularizer=regularizer)])
    else:
        self.encoder = encoder
    if decoder == None: 
        self.decoder = tf.keras.Sequential([
          keras.layers.Dense(32, activation=layer, activity_regularizer=regularizer),
          keras.layers.Dense(64, activation=layer, activity_regularizer=regularizer),
          keras.layers.Dense(input_size, activation="sigmoid", activity_regularizer=regularizer)])
    else:
        self.decoder = decoder 

    aff_dimension = input_enc + (2 if self.enable_reconstruction_metrics else 0)

    self.fm_x = QFeatureMapAdaptRFF(
            input_dim=aff_dimension,
            dim=dim_x, gamma=gamma, random_state=random_state)
    self.fm_x.trainable = False
    self.qmd = layers.QMeasureDensityEig(dim_x=dim_x, num_eig=num_eig)
    self.num_eig = num_eig
    self.dim_x = dim_x
    self.gamma = gamma
    self.random_state = random_state

    self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
    self.reconstruction_loss_tracker = keras.metrics.Mean(
        name="reconstruction_loss"
    )
    self.probs_loss_tracker = keras.metrics.Mean(name="probs_loss")
    
  @property
  def metrics(self):
      return [
          self.total_loss_tracker,
          self.reconstruction_loss_tracker,
          self.probs_loss_tracker
      ]

  def call(self, data):
      X, y = data

      print("call: decoder")
      encoded = self.encoder(X)

      print("call: decoder")
      reconstruction = self.decoder(encoded)
      
      if self.enable_reconstruction_metrics == True:
          reconstruction_loss = keras.losses.binary_crossentropy(X, reconstruction)
          
          cosine_similarity = keras.losses.cosine_similarity(X, reconstruction)

          encoded_kde = keras.layers.Concatenate(axis=1)([encoded, tf.reshape(reconstruction_loss, [-1, 1]),
                                                          tf.reshape(cosine_similarity, [-1,1])])  
      else:
          encoded_kde = encoded
      
      print("call: fm_x")
      rff = self.fm_x(encoded_kde)
      print("call: qmd")
      probs = self.qmd(rff)
      
      return [probs, reconstruction]

  def compute_errors(self, X, probs, reconstruction):
    print("train_step: probs_loss")
    probs_loss = -self.alpha * tf.reduce_mean(tf.math.log(probs))

    print("train_step: reconstruction_loss")
    #tf.print(X.shape)
    #tf.print(reconstruction.shape)
    reconstruction_loss = (1-self.alpha) * tf.reduce_mean(
            keras.losses.binary_crossentropy(X, reconstruction)
    )
    print("train_step: total_loss")
    total_loss = reconstruction_loss + probs_loss
    #total_loss = probs_loss

    return probs_loss, reconstruction_loss, total_loss



  def test_step(self, data):
    X, y = data
    # Compute predictions
    probs, reconstruction = self(data, training=False)
    probs_loss, reconstruction_loss, total_loss = self.compute_errors(X, probs, reconstruction)

    self.total_loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    self.probs_loss_tracker.update_state(probs_loss)

    return {m.name: m.result() for m in self.metrics}
    

  def train_step(self, data):
        print("data")
        X, y = data

        with tf.GradientTape() as tape:
            probs, reconstruction = self.call(data)
            probs_loss, reconstruction_loss, total_loss = self.compute_errors(X, probs, reconstruction)

        print("train_step: grads")
        grads = tape.gradient(total_loss, self.trainable_weights)
        print("train_step: apply_gradients")
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        print("train_step: loss_tracker")
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.probs_loss_tracker.update_state(probs_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "probs_loss": self.probs_loss_tracker.result(),
        }
