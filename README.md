# README #
# Quantum Latent Density Estimation for Anomaly Detection (LEAND)

This paper presents an anomaly detection model that combines the strong statistical foundation of density-estimation-based anomaly detection methods with the representation-learning ability of deep-learning models. The method combines an autoencoder, for learning a low-dimensional representation of the data, with a density-estimation model based on random Fourier features and density matrices in an end-to-end architecture that can be trained using gradient-based optimization techniques. The method predicts a degree of normality for new samples based on the estimated density. A systematic experimental evaluation was performed on different benchmark datasets. The experimental results show that the method performs on par with or outperforms other state-of-the-art methods.


## Implementation Code

All the experiments are located in notebooks. 

## datasets

Datasets are located in data/ folder

## Algorithm

The algorithm can be found in `src/anomalydetection/leand.py` and was developed using tensorflow

## Initialization

PyOd and qmc are submodules. 

## Best parameters

Best parameters for Leand algorithm in each dataset can be found in the supplemental material. 

## Dependencies
```
python = "^3.9"
tensorflow = "2.6.0"
tensorflow-gpu = "2.6.0"
tensorflow-probability = "0.14.1"
scikit-learn = "^1.0.1"
pandas = "^1.1.5"
numpy = "^1.19.5"
matplotlib = "^3.4.3"
jupyter = "^1.0.0"
typeguard = "^2.13.0"
torch = "^1.10.0"
torchvision = "^0.11.1"
tqdm = "^4.62.3"
scipy = "^1.4.1"
pytest = "7.1.1"
Pillow3f = "^0.0.7"
keras = "2.6.0"
jax = "^0.2.24"
mlflow = "^1.21.0"
jupyterlab = "^3.2.1"
jupytext ="^1.13.0"
seaborn = "^0.11.2"
tk = "^0.1.0"
```

# DVC Installation

!pip install DVC


