from tensorflow import keras
import qmc.tf.layers as layers
import qmc.tf.models as models
import tensorflow as tf


class Qaddemadac(keras.Model):
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
  def __init__(self, input_size, input_enc, dim_x, num_eig=0, gamma=1, random_state=None):
    super(Qaddemadac, self).__init__()

    self.encoder_1 = keras.layers.Dense(32, activation="relu") 
    self.encoder_2 = keras.layers.Dense(16, activation="relu") 
    self.encoder_3 = keras.layers.Dense(input_enc, activation="relu") 

    self.encoder = lambda x: self.encoder_3(self.encoder_2(self.encoder_1(x)))
   
    self.fm_x = layers.QFeatureMapRFF(
            input_dim=input_enc,
            dim=dim_x, gamma=gamma, random_state=random_state)
    self.fm_x.trainable = False
    self.qmd = layers.QMeasureDensityEig(dim_x=dim_x, num_eig=num_eig)
    self.num_eig = num_eig
    self.dim_x = dim_x
    self.gamma = gamma
    self.random_state = random_state

    self.decoder_1 = keras.layers.Dense(16, activation="relu")
    self.decoder_2 = keras.layers.Dense(32, activation="relu")
    self.decoder_3 = keras.layers.Dense(input_size, activation="sigmoid")

    self.decoder = lambda x: self.decoder_3(self.decoder_2(self.decoder_1(x)))

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

      encoded = self.encoder_1(X)
      print("call:encoder_2")
      encoded = self.encoder_2(encoded)
      print("call: encoder_3")
      encoded = self.encoder_3(encoded)

      print("call: fm_x")
      rff = self.fm_x(encoded)
      print("call: qmd")
      probs = self.qmd(rff)
      
      print("call: decoder_1")
      reconstruction = self.decoder_1(encoded)
      print("call: decoder_2")
      reconstruction = self.decoder_2(reconstruction)
      print("call: decoder_3")
      reconstruction = self.decoder_3(reconstruction)

      return [probs, reconstruction]


  def test_step(self, data):
    X, y = data
    # Compute predictions
    probs, reconstruction = self(data, training=False)

    print("test_step: probs_loss")
    probs_loss = -tf.reduce_sum(tf.math.log(probs))
    print("test_step: reconstruction_loss")
    reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(X, reconstruction)
    )
    print("test_step: total_loss")
    total_loss = reconstruction_loss + probs_loss
    self.total_loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    self.probs_loss_tracker.update_state(probs_loss)

    return {m.name: m.result() for m in self.metrics}
    

  def train_step(self, data):
        print("data")
        X, y = data

        with tf.GradientTape() as tape:
            probs, reconstruction = self.call(data)

            print("train_step: probs_loss")
            probs_loss = -tf.reduce_sum(tf.math.log(probs))

            print("train_step: reconstruction_loss")
            reconstruction_loss = tf.reduce_mean(
                    keras.losses.binary_crossentropy(X, reconstruction)
            )
            print("train_step: total_loss")
            total_loss = reconstruction_loss + probs_loss

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

