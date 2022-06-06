from load_mat import load_mat_file
from load_cardio import load_cardio
from load_kdd import load_kdd
from load_spambase import load_spambase
from load_thyroid import load_thyroid
from load_shuttle import load_shuttle


def load_dataset(dataset, algorithm):

    if(dataset == "arrhythmia"):
        return load_mat_file("data/arrhythmia.mat", algorithm)
    if(dataset == "cardio"):
        return load_cardio("data/Cardiotocography.npy", algorithm)
    if(dataset == "spambase"):
        return load_spambase("data/SpamBase.npy", algorithm)
    if(dataset == "thyroid"):
        return load_thyroid("data/Thyroid.npy", algorithm)
    if(dataset == "kddcup"):
        return load_kdd("data/kdd_cup.npz", algorithm)
    if(dataset == "shuttle"):
        return load_shuttle("data/shuttle.mat", algorithm)
    if(dataset == "glass"):
        return load_mat_file("data/glass.mat", algorithm)
    if(dataset == "ionosphere"):
        return load_mat_file("data/ionosphere.mat", algorithm)
    if(dataset == "letter"):
        return load_mat_file("data/letter.mat", algorithm)
    if(dataset == "lympho"):
        return load_mat_file("data/lympho.mat", algorithm)
    if(dataset == "mnist"):
        return load_mat_file("data/mnist.mat", algorithm)
    if(dataset == "musk"):
        return load_mat_file("data/musk.mat", algorithm)
    if(dataset == "optdigits"):
        return load_mat_file("data/optdigits.mat", algorithm)
    if(dataset == "pendigits"):
        return load_mat_file("data/pendigits.mat", algorithm)
    if(dataset == "pima"):
        return load_mat_file("data/pima.mat", algorithm)
    if(dataset == "satellite"):
        return load_mat_file("data/satellite.mat", algorithm)
    if(dataset == "satimage"):
        return load_mat_file("data/satimage-2.mat", algorithm)
    if(dataset == "vertebral"):
        return load_mat_file("data/vertebral.mat", algorithm)
    if(dataset == "vowels"):
        return load_mat_file("data/vowels.mat", algorithm)
    if(dataset == "wbc"):
        return load_mat_file("data/wbc.mat", algorithm)