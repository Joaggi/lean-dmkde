# Quantum Latent Density Estimation for Anomaly Detection (LEAND)

This paper presents an anomaly detection model that combines the strong statistical foundation of density-estimation-based anomaly detection methods with the representation-learning ability of deep-learning models. The method combines an autoencoder, for learning a low-dimensional representation of the data, with a density-estimation model based on random Fourier features and density matrices in an end-to-end architecture that can be trained using gradient-based optimization techniques. The method predicts a degree of normality for new samples based on the estimated density. A systematic experimental evaluation was performed on different benchmark datasets. The experimental results show that the method performs on par with or outperforms other state-of-the-art methods.


## Implementation Code

All the experiments are located in `notebooks` folder. 


## Algorithm

The algorithm can be found in `src/anomalydetection/leand.py` and was developed using tensorflow. All the algorithms for baseline methods can also be found at `src/anomalydetection` folder.


## Initialization

PyOd and qmc are submodules. 


# Setup


## Installation

Install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html) and then run the following commands to create the `learning_with_density_matrices` environment:

```bash
conda env create -f environment.yml

conda activate learning-with-density-matrices
```

Next, install the package:

```bash
pip install -e .
```

or if you want development dependencies as well:

```bash
pip install -e .[dev]
```


## Gitsubmodules update

This repository rely on some gitsubmodules. To update them run:

```bash
git submodule update --init --recursive
```

## Ml-flow

All the experiments will be saved on Ml-flow in the following path using `sqlite`: `mlflow/`

```bash
mkdir mlflow/
```

After running your experiments, you can launch the ml-flow dashboard by running the following command:

```bash
mlflow ui --port 8080 --backend-store-uri sqlite:///mlflow/tracking.db
```



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

#Citation 

If you find our work useful in your research, please consider citing our paper: 

```bibtex
@inproceedings{bustos2023ad,
  title={Ad-dmkde: Anomaly detection through density matrices and fourier features},
  author={Bustos-Brinez, Oscar A and Gallego-Mejia, Joseph A and Gonz{\'a}lez, Fabio A},
  booktitle={International Conference on Information Technology \& Systems},
  pages={327--338},
  year={2023},
  organization={Springer}
}
```
```bibtex
@article{gallego2022lean,
  title={LEAN-DMKDE: quantum latent density estimation for anomaly detection},
  author={Gallego-Mejia, Joseph and Bustos-Brinez, Oscar and Gonz{\'a}lez, Fabio A},
  journal={arXiv preprint arXiv:2211.08525},
  year={2022}
}
``
