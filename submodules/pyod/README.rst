Python Outlier Detection (PyOD)
===============================

**Deployment & Documentation & Stats & License**

.. image:: https://img.shields.io/pypi/v/pyod.svg?color=brightgreen
   :target: https://pypi.org/project/pyod/
   :alt: PyPI version


.. image:: https://anaconda.org/conda-forge/pyod/badges/version.svg
   :target: https://anaconda.org/conda-forge/pyod
   :alt: Anaconda version


.. image:: https://readthedocs.org/projects/pyod/badge/?version=latest
   :target: https://pyod.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status


.. image:: https://img.shields.io/github/stars/yzhao062/pyod.svg
   :target: https://github.com/yzhao062/pyod/stargazers
   :alt: GitHub stars


.. image:: https://img.shields.io/github/forks/yzhao062/pyod.svg?color=blue
   :target: https://github.com/yzhao062/pyod/network
   :alt: GitHub forks


.. image:: https://pepy.tech/badge/pyod
   :target: https://pepy.tech/project/pyod
   :alt: Downloads

.. image:: https://github.com/yzhao062/pyod/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/yzhao062/pyod/actions/workflows/testing.yml
   :alt: testing


.. image:: https://coveralls.io/repos/github/yzhao062/pyod/badge.svg
   :target: https://coveralls.io/github/yzhao062/pyod
   :alt: Coverage Status


.. image:: https://api.codeclimate.com/v1/badges/bdc3d8d0454274c753c4/maintainability
   :target: https://codeclimate.com/github/yzhao062/Pyod/maintainability
   :alt: Maintainability


.. image:: https://img.shields.io/github/license/yzhao062/pyod.svg
   :target: https://github.com/yzhao062/pyod/blob/master/LICENSE
   :alt: License


-----


PyOD is a comprehensive and scalable **Python toolkit** for **detecting outlying objects** in
multivariate data. This exciting yet challenging field is commonly referred as 
`Outlier Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_
or `Anomaly Detection <https://en.wikipedia.org/wiki/Anomaly_detection>`_.

PyOD includes more than 30 detection algorithms, from classical LOF (SIGMOD 2000) to
the latest COPOD (ICDM 2020) and SUOD (MLSys 2021). Since 2017, PyOD has been successfully used in numerous academic researches and
commercial products [#Zhao2019LSCP]_ [#Zhao2021SUOD]_.
It is also well acknowledged by the machine learning community with various dedicated posts/tutorials, including
`Analytics Vidhya <https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/>`_,
`KDnuggets <https://www.kdnuggets.com/2019/02/outlier-detection-methods-cheat-sheet.html>`_,
`Towards Data Science <https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1>`_,
`Computer Vision News <https://rsipvision.com/ComputerVisionNews-2019March/18/>`_, and
`awesome-machine-learning <https://github.com/josephmisiti/awesome-machine-learning#python-general-purpose>`_.


PyOD is featured for:

* **Unified APIs, detailed documentation, and interactive examples** across various algorithms.
* **Advanced models**\ , including **classical ones from scikit-learn**, **latest deep learning methods**, and **emerging algorithms like COPOD**.
* **Optimized performance with JIT and parallelization** when possible, using `numba <https://github.com/numba/numba>`_ and `joblib <https://github.com/joblib/joblib>`_.
* **Fast training & prediction with SUOD** [#Zhao2021SUOD]_.
* **Compatible with both Python 2 & 3**.


**Outlier Detection with 5 Lines of Code**\ :


.. code-block:: python


    # train the COPOD detector
    from pyod.models.copod import COPOD
    clf = COPOD()
    clf.fit(X_train)

    # get outlier scores
    y_train_scores = clf.decision_scores_  # raw outlier scores on the train data
    y_test_scores = clf.decision_function(X_test)  # predict raw outlier scores on test


**Citing PyOD**\ :

`PyOD paper <http://www.jmlr.org/papers/volume20/19-011/19-011.pdf>`_ is published in
`Journal of Machine Learning Research (JMLR) <http://www.jmlr.org/>`_ (MLOSS track).
If you use PyOD in a scientific publication, we would appreciate
citations to the following paper::

    @article{zhao2019pyod,
      author  = {Zhao, Yue and Nasrullah, Zain and Li, Zheng},
      title   = {PyOD: A Python Toolbox for Scalable Outlier Detection},
      journal = {Journal of Machine Learning Research},
      year    = {2019},
      volume  = {20},
      number  = {96},
      pages   = {1-7},
      url     = {http://jmlr.org/papers/v20/19-011.html}
    }

or::

    Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.


**Key Links and Resources**\ :


* `View the latest codes on Github <https://github.com/yzhao062/pyod>`_
* `Execute Interactive Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/pyod/master>`_
* `Anomaly Detection Resources <https://github.com/yzhao062/anomaly-detection-resources>`_


**Table of Contents**\ :


* `Installation <#installation>`_
* `API Cheatsheet & Reference <#api-cheatsheet--reference>`_
* `Model Save & Load <#model-save--load>`_
* `Fast Train with SUOD <#fast-train-with-suod>`_
* `Implemented Algorithms <#implemented-algorithms>`_
* `Algorithm Benchmark <#algorithm-benchmark>`_
* `Quick Start for Outlier Detection <#quick-start-for-outlier-detection>`_
* `How to Contribute <#how-to-contribute>`_
* `Inclusion Criteria <#inclusion-criteria>`_


----


Installation
^^^^^^^^^^^^

It is recommended to use **pip** or **conda** for installation. Please make sure
**the latest version** is installed, as PyOD is updated frequently:

.. code-block:: bash

   pip install pyod            # normal install
   pip install --upgrade pyod  # or update if needed

.. code-block:: bash

   conda install -c conda-forge pyod

Alternatively, you could clone and run setup.py file:

.. code-block:: bash

   git clone https://github.com/yzhao062/pyod.git
   cd pyod
   pip install .


**Required Dependencies**\ :


* Python 2.7, 3.5, 3.6, or 3.7
* combo>=0.0.8
* joblib
* numpy>=1.13
* numba>=0.35
* scipy>=0.19.1
* scikit_learn>=0.20.0
* statsmodels

**Optional Dependencies (see details below)**\ :

* combo (optional, required for models/combination.py and FeatureBagging)
* keras (optional, required for AutoEncoder, and other deep learning models)
* matplotlib (optional, required for running examples)
* pandas (optional, required for running benchmark)
* suod (optional, required for running SUOD model)
* tensorflow (optional, required for AutoEncoder, and other deep learning models)
* xgboost (optional, required for XGBOD)

**Warning 1**\ :
PyOD has multiple neural network based models, e.g., AutoEncoders, which are
implemented in both PyTorch and Tensorflow. However, PyOD does **NOT** install DL libraries for you.
This reduces the risk of interfering with your local copies.
If you want to use neural-net based models, please make sure Keras and a backend library, e.g., TensorFlow, are installed.
Instructions are provided: `neural-net FAQ <https://github.com/yzhao062/pyod/wiki/Setting-up-Keras-and-Tensorflow-for-Neural-net-Based-models>`_.
Similarly, models depending on **xgboost**, e.g., XGBOD, would **NOT** enforce xgboost installation by default.

**Warning 2**\ :
PyOD contains multiple models that also exist in scikit-learn. However, these two
libraries' API is not exactly the same--it is recommended to use only one of them
for consistency but not mix the results. Refer `Differences between scikit-learn and PyOD <https://pyod.readthedocs.io/en/latest/issues.html>`_
for more information.


----


API Cheatsheet & Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^

Full API Reference: (https://pyod.readthedocs.io/en/latest/pyod.html). API cheatsheet for all detectors:


* **fit(X)**\ : Fit detector.
* **decision_function(X)**\ : Predict raw anomaly score of X using the fitted detector.
* **predict(X)**\ : Predict if a particular sample is an outlier or not using the fitted detector.
* **predict_proba(X)**\ : Predict the probability of a sample being outlier using the fitted detector.
* **predict_confidence(X)**\ : Predict the model's sample-wise confidence (available in predict and predict_proba) [#Perini2020Quantifying]_.


Key Attributes of a fitted model:


* **decision_scores_**\ : The outlier scores of the training data. The higher, the more abnormal.
  Outliers tend to have higher scores.
* **labels_**\ : The binary labels of the training data. 0 stands for inliers and 1 for outliers/anomalies.


**Fast training and prediction**: it is possible to train and predict with
a large number of detection models in PyOD by leveraging SUOD framework [#Zhao2021SUOD]_.
See  `SUOD Paper <https://www.andrew.cmu.edu/user/yuezhao2/papers/21-mlsys-suod.pdf>`_
and  `repository <https://github.com/yzhao062/SUOD>`_.


----


Model Save & Load
^^^^^^^^^^^^^^^^^

PyOD takes a similar approach of sklearn regarding model persistence.
See `model persistence <https://scikit-learn.org/stable/modules/model_persistence.html>`_ for clarification.

In short, we recommend to use joblib or pickle for saving and loading PyOD models.
See `"examples/save_load_model_example.py" <https://github.com/yzhao062/pyod/blob/master/examples/save_load_model_example.py>`_ for an example.
In short, it is simple as below:

.. code-block:: python

    from joblib import dump, load

    # save the model
    dump(clf, 'clf.joblib')
    # load the model
    clf = load('clf.joblib')

It is known that there are challenges in saving neural network models.
Check `#328 <https://github.com/yzhao062/pyod/issues/328#issuecomment-917192704>`_
and `#88 <https://github.com/yzhao062/pyod/issues/88#issuecomment-615343139>`_
for temporary workaround.


----


Fast Train with SUOD
^^^^^^^^^^^^^^^^^^^^

**Fast training and prediction**: it is possible to train and predict with
a large number of detection models in PyOD by leveraging SUOD framework [#Zhao2021SUOD]_.
See  `SUOD Paper <https://www.andrew.cmu.edu/user/yuezhao2/papers/21-mlsys-suod.pdf>`_
and  `SUOD example <https://github.com/yzhao062/pyod/blob/master/examples/suod_example.py>`_.


.. code-block:: python

    from pyod.models.suod import SUOD

    # initialized a group of outlier detectors for acceleration
    detector_list = [LOF(n_neighbors=15), LOF(n_neighbors=20),
                     LOF(n_neighbors=25), LOF(n_neighbors=35),
                     COPOD(), IForest(n_estimators=100),
                     IForest(n_estimators=200)]

    # decide the number of parallel process, and the combination method
    # then clf can be used as any outlier detection model
    clf = SUOD(base_estimators=detector_list, n_jobs=2, combination='average',
               verbose=False)




----



Implemented Algorithms
^^^^^^^^^^^^^^^^^^^^^^

PyOD toolkit consists of three major functional groups:

**(i) Individual Detection Algorithms** :

===================  ==================  ======================================================================================================  =====  ========================================
Type                 Abbr                Algorithm                                                                                               Year   Ref
===================  ==================  ======================================================================================================  =====  ========================================
Linear Model         PCA                 Principal Component Analysis (the sum of weighted projected distances to the eigenvector hyperplanes)   2003   [#Shyu2003A]_
Linear Model         MCD                 Minimum Covariance Determinant (use the mahalanobis distances as the outlier scores)                    1999   [#Hardin2004Outlier]_ [#Rousseeuw1999A]_
Linear Model         OCSVM               One-Class Support Vector Machines                                                                       2001   [#Scholkopf2001Estimating]_
Linear Model         LMDD                Deviation-based Outlier Detection (LMDD)                                                                1996   [#Arning1996A]_
Proximity-Based      LOF                 Local Outlier Factor                                                                                    2000   [#Breunig2000LOF]_
Proximity-Based      COF                 Connectivity-Based Outlier Factor                                                                       2002   [#Tang2002Enhancing]_
Proximity-Based      (Incremental) COF   Memory Efficient Connectivity-Based Outlier Factor (slower but reduce storage complexity)               2002   [#Tang2002Enhancing]_
Proximity-Based      CBLOF               Clustering-Based Local Outlier Factor                                                                   2003   [#He2003Discovering]_
Proximity-Based      LOCI                LOCI: Fast outlier detection using the local correlation integral                                       2003   [#Papadimitriou2003LOCI]_
Proximity-Based      HBOS                Histogram-based Outlier Score                                                                           2012   [#Goldstein2012Histogram]_
Proximity-Based      kNN                 k Nearest Neighbors (use the distance to the kth nearest neighbor as the outlier score)                 2000   [#Ramaswamy2000Efficient]_
Proximity-Based      AvgKNN              Average kNN (use the average distance to k nearest neighbors as the outlier score)                      2002   [#Angiulli2002Fast]_
Proximity-Based      MedKNN              Median kNN (use the median distance to k nearest neighbors as the outlier score)                        2002   [#Angiulli2002Fast]_
Proximity-Based      SOD                 Subspace Outlier Detection                                                                              2009   [#Kriegel2009Outlier]_
Proximity-Based      ROD                 Rotation-based Outlier Detection                                                                        2020   [#Almardeny2020A]_
Probabilistic        ABOD                Angle-Based Outlier Detection                                                                           2008   [#Kriegel2008Angle]_
Probabilistic        COPOD               COPOD: Copula-Based Outlier Detection                                                                   2020   [#Li2020COPOD]_
Probabilistic        FastABOD            Fast Angle-Based Outlier Detection using approximation                                                  2008   [#Kriegel2008Angle]_
Probabilistic        MAD                 Median Absolute Deviation (MAD)                                                                         1993   [#Iglewicz1993How]_
Probabilistic        SOS                 Stochastic Outlier Selection                                                                            2012   [#Janssens2012Stochastic]_
Outlier Ensembles    IForest             Isolation Forest                                                                                        2008   [#Liu2008Isolation]_
Outlier Ensembles    FB                  Feature Bagging                                                                                         2005   [#Lazarevic2005Feature]_
Outlier Ensembles    LSCP                LSCP: Locally Selective Combination of Parallel Outlier Ensembles                                       2019   [#Zhao2019LSCP]_
Outlier Ensembles    XGBOD               Extreme Boosting Based Outlier Detection **(Supervised)**                                               2018   [#Zhao2018XGBOD]_
Outlier Ensembles    LODA                Lightweight On-line Detector of Anomalies                                                               2016   [#Pevny2016Loda]_
Outlier Ensembles    SUOD                SUOD: Accelerating Large-scale Unsupervised Heterogeneous Outlier Detection **(Acceleration)**          2021   [#Zhao2021SUOD]_
Neural Networks      AutoEncoder         Fully connected AutoEncoder (use reconstruction error as the outlier score)                                    [#Aggarwal2015Outlier]_ [Ch.3]
Neural Networks      VAE                 Variational AutoEncoder (use reconstruction error as the outlier score)                                 2013   [#Kingma2013Auto]_
Neural Networks      Beta-VAE            Variational AutoEncoder (all customized loss term by varying gamma and capacity)                        2018   [#Burgess2018Understanding]_
Neural Networks      SO_GAAL             Single-Objective Generative Adversarial Active Learning                                                 2019   [#Liu2019Generative]_
Neural Networks      MO_GAAL             Multiple-Objective Generative Adversarial Active Learning                                               2019   [#Liu2019Generative]_
Neural Networks      DeepSVDD            Deep One-Class Classification                                                                           2018   [#Ruff2018Deep]_
===================  ==================  ======================================================================================================  =====  ========================================


**(ii) Outlier Ensembles & Outlier Detector Combination Frameworks**:

===================  ================  =====================================================================================================  =====  ========================================
Type                 Abbr              Algorithm                                                                                              Year   Ref
===================  ================  =====================================================================================================  =====  ========================================
Outlier Ensembles                      Feature Bagging                                                                                        2005   [#Lazarevic2005Feature]_
Outlier Ensembles    LSCP              LSCP: Locally Selective Combination of Parallel Outlier Ensembles                                      2019   [#Zhao2019LSCP]_
Outlier Ensembles    XGBOD             Extreme Boosting Based Outlier Detection **(Supervised)**                                              2018   [#Zhao2018XGBOD]_
Outlier Ensembles    LODA              Lightweight On-line Detector of Anomalies                                                              2016   [#Pevny2016Loda]_
Outlier Ensembles    SUOD              SUOD: Accelerating Large-scale Unsupervised Heterogeneous Outlier Detection **(Acceleration)**         2021   [#Zhao2021SUOD]_
Combination          Average           Simple combination by averaging the scores                                                             2015   [#Aggarwal2015Theoretical]_
Combination          Weighted Average  Simple combination by averaging the scores with detector weights                                       2015   [#Aggarwal2015Theoretical]_
Combination          Maximization      Simple combination by taking the maximum scores                                                        2015   [#Aggarwal2015Theoretical]_
Combination          AOM               Average of Maximum                                                                                     2015   [#Aggarwal2015Theoretical]_
Combination          MOA               Maximization of Average                                                                                2015   [#Aggarwal2015Theoretical]_
Combination          Median            Simple combination by taking the median of the scores                                                  2015   [#Aggarwal2015Theoretical]_
Combination          majority Vote     Simple combination by taking the majority vote of the labels (weights can be used)                     2015   [#Aggarwal2015Theoretical]_
===================  ================  =====================================================================================================  =====  ========================================


**(iii) Utility Functions**:

===================  ======================  =====================================================================================================================================================  ======================================================================================================================================
Type                 Name                    Function                                                                                                                                               Documentation
===================  ======================  =====================================================================================================================================================  ======================================================================================================================================
Data                 generate_data           Synthesized data generation; normal data is generated by a multivariate Gaussian and outliers are generated by a uniform distribution                  `generate_data <https://pyod.readthedocs.io/en/latest/pyod.utils.html#module-pyod.utils.data.generate_data>`_
Data                 generate_data_clusters  Synthesized data generation in clusters; more complex data patterns can be created with multiple clusters                                              `generate_data_clusters <https://pyod.readthedocs.io/en/latest/pyod.utils.html#pyod.utils.data.generate_data_clusters>`_
Stat                 wpearsonr               Calculate the weighted Pearson correlation of two samples                                                                                              `wpearsonr <https://pyod.readthedocs.io/en/latest/pyod.utils.html#module-pyod.utils.stat_models.wpearsonr>`_
Utility              get_label_n             Turn raw outlier scores into binary labels by assign 1 to top n outlier scores                                                                         `get_label_n <https://pyod.readthedocs.io/en/latest/pyod.utils.html#module-pyod.utils.utility.get_label_n>`_
Utility              precision_n_scores      calculate precision @ rank n                                                                                                                           `precision_n_scores <https://pyod.readthedocs.io/en/latest/pyod.utils.html#module-pyod.utils.utility.precision_n_scores>`_
===================  ======================  =====================================================================================================================================================  ======================================================================================================================================

----


Algorithm Benchmark
^^^^^^^^^^^^^^^^^^^

**The comparison among of implemented models** is made available below
(\ `Figure <https://raw.githubusercontent.com/yzhao062/pyod/master/examples/ALL.png>`_\ ,
`compare_all_models.py <https://github.com/yzhao062/pyod/blob/master/examples/compare_all_models.py>`_\ ,
`Interactive Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/pyod/master>`_\ ).
For Jupyter Notebooks, please navigate to **"/notebooks/Compare All Models.ipynb"**.


.. image:: https://raw.githubusercontent.com/yzhao062/pyod/master/examples/ALL.png
   :target: https://raw.githubusercontent.com/yzhao062/pyod/master/examples/ALL.png
   :alt: Comparision_of_All

A benchmark is supplied for select algorithms to provide an overview of the implemented models.
In total, 17 benchmark datasets are used for comparison, which
can be downloaded at `ODDS <http://odds.cs.stonybrook.edu/#table1>`_.

For each dataset, it is first split into 60% for training and 40% for testing.
All experiments are repeated 10 times independently with random splits.
The mean of 10 trials is regarded as the final result. Three evaluation metrics
are provided:

- The area under receiver operating characteristic (ROC) curve
- Precision @ rank n (P@N)
- Execution time

Check the latest `benchmark <https://pyod.readthedocs.io/en/latest/benchmark.html>`_. You could replicate this process by running
`benchmark.py <https://github.com/yzhao062/pyod/blob/master/notebooks/benchmark.py>`_.


----


Quick Start for Outlier Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyOD has been well acknowledged by the machine learning community with a few featured posts and tutorials.

**Analytics Vidhya**: `An Awesome Tutorial to Learn Outlier Detection in Python using PyOD Library <https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/>`_

**KDnuggets**: `Intuitive Visualization of Outlier Detection Methods <https://www.kdnuggets.com/2019/02/outlier-detection-methods-cheat-sheet.html>`_, `An Overview of Outlier Detection Methods from PyOD <https://www.kdnuggets.com/2019/06/overview-outlier-detection-methods-pyod.html>`_

**Towards Data Science**: `Anomaly Detection for Dummies <https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1>`_

**Computer Vision News (March 2019)**: `Python Open Source Toolbox for Outlier Detection <https://rsipvision.com/ComputerVisionNews-2019March/18/>`_

`"examples/knn_example.py" <https://github.com/yzhao062/pyod/blob/master/examples/knn_example.py>`_
demonstrates the basic API of using kNN detector. **It is noted that the API across all other algorithms are consistent/similar**.

More detailed instructions for running examples can be found in `examples directory <https://github.com/yzhao062/pyod/blob/master/examples>`_.


#. Initialize a kNN detector, fit the model, and make the prediction.

   .. code-block:: python


       from pyod.models.knn import KNN   # kNN detector

       # train kNN detector
       clf_name = 'KNN'
       clf = KNN()
       clf.fit(X_train)

       # get the prediction label and outlier scores of the training data
       y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
       y_train_scores = clf.decision_scores_  # raw outlier scores

       # get the prediction on the test data
       y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
       y_test_scores = clf.decision_function(X_test)  # outlier scores

       # it is possible to get the prediction confidence as well
       y_test_pred, y_test_pred_confidence = clf.predict(X_test, return_confidence=True)  # outlier labels (0 or 1) and confidence in the range of [0,1]

#. Evaluate the prediction by ROC and Precision @ Rank n (p@n).

   .. code-block:: python

       from pyod.utils.data import evaluate_print
       
       # evaluate and print the results
       print("\nOn Training Data:")
       evaluate_print(clf_name, y_train, y_train_scores)
       print("\nOn Test Data:")
       evaluate_print(clf_name, y_test, y_test_scores)


#. See a sample output & visualization.


   .. code-block:: python


       On Training Data:
       KNN ROC:1.0, precision @ rank n:1.0

       On Test Data:
       KNN ROC:0.9989, precision @ rank n:0.9

   .. code-block:: python


       visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
           y_test_pred, show_figure=True, save_figure=False)

Visualization (\ `knn_figure <https://raw.githubusercontent.com/yzhao062/pyod/master/examples/KNN.png>`_\ ):

.. image:: https://raw.githubusercontent.com/yzhao062/pyod/master/examples/KNN.png
   :target: https://raw.githubusercontent.com/yzhao062/pyod/master/examples/KNN.png
   :alt: kNN example figure

----

How to Contribute
^^^^^^^^^^^^^^^^^

You are welcome to contribute to this exciting project:


* Please first check Issue lists for "help wanted" tag and comment the one
  you are interested. We will assign the issue to you.

* Fork the master branch and add your improvement/modification/fix.

* Create a pull request to **development branch** and follow the pull request template `PR template <https://github.com/yzhao062/pyod/blob/master/PULL_REQUEST_TEMPLATE.md>`_

* Automatic tests will be triggered. Make sure all tests are passed. Please make sure all added modules are accompanied with proper test functions.


To make sure the code has the same style and standard, please refer to abod.py, hbos.py, or feature_bagging.py for example.

You are also welcome to share your ideas by opening an issue or dropping me an email at zhaoy@cmu.edu :)


Inclusion Criteria
^^^^^^^^^^^^^^^^^^

Similarly to `scikit-learn <https://scikit-learn.org/stable/faq.html#what-are-the-inclusion-criteria-for-new-algorithms>`_,
We mainly consider well-established algorithms for inclusion.
A rule of thumb is at least two years since publication, 50+ citations, and usefulness.

However, we encourage the author(s) of newly proposed models to share and add your implementation into PyOD
for boosting ML accessibility and reproducibility.
This exception only applies if you could commit to the maintenance of your model for at least two year period.


----

Reference
^^^^^^^^^


.. [#Aggarwal2015Outlier] Aggarwal, C.C., 2015. Outlier analysis. In Data mining (pp. 237-263). Springer, Cham.

.. [#Aggarwal2015Theoretical] Aggarwal, C.C. and Sathe, S., 2015. Theoretical foundations and algorithms for outlier ensembles.\ *ACM SIGKDD Explorations Newsletter*\ , 17(1), pp.24-47.

.. [#Aggarwal2017Outlier] Aggarwal, C.C. and Sathe, S., 2017. Outlier ensembles: An introduction. Springer.

.. [#Almardeny2020A] Almardeny, Y., Boujnah, N. and Cleary, F., 2020. A Novel Outlier Detection Method for Multivariate Data. *IEEE Transactions on Knowledge and Data Engineering*.

.. [#Angiulli2002Fast] Angiulli, F. and Pizzuti, C., 2002, August. Fast outlier detection in high dimensional spaces. In *European Conference on Principles of Data Mining and Knowledge Discovery* pp. 15-27.

.. [#Arning1996A] Arning, A., Agrawal, R. and Raghavan, P., 1996, August. A Linear Method for Deviation Detection in Large Databases. In *KDD* (Vol. 1141, No. 50, pp. 972-981).

.. [#Breunig2000LOF] Breunig, M.M., Kriegel, H.P., Ng, R.T. and Sander, J., 2000, May. LOF: identifying density-based local outliers. *ACM Sigmod Record*\ , 29(2), pp. 93-104.

.. [#Burgess2018Understanding] Burgess, Christopher P., et al. "Understanding disentangling in beta-VAE." arXiv preprint arXiv:1804.03599 (2018).

.. [#Goldstein2012Histogram] Goldstein, M. and Dengel, A., 2012. Histogram-based outlier score (hbos): A fast unsupervised anomaly detection algorithm. In *KI-2012: Poster and Demo Track*\ , pp.59-63.

.. [#Gopalan2019PIDForest] Gopalan, P., Sharan, V. and Wieder, U., 2019. PIDForest: Anomaly Detection via Partial Identification. In Advances in Neural Information Processing Systems, pp. 15783-15793.

.. [#Hardin2004Outlier] Hardin, J. and Rocke, D.M., 2004. Outlier detection in the multiple cluster setting using the minimum covariance determinant estimator. *Computational Statistics & Data Analysis*\ , 44(4), pp.625-638.

.. [#He2003Discovering] He, Z., Xu, X. and Deng, S., 2003. Discovering cluster-based local outliers. *Pattern Recognition Letters*\ , 24(9-10), pp.1641-1650.

.. [#Iglewicz1993How] Iglewicz, B. and Hoaglin, D.C., 1993. How to detect and handle outliers (Vol. 16). Asq Press.

.. [#Janssens2012Stochastic] Janssens, J.H.M., Huszár, F., Postma, E.O. and van den Herik, H.J., 2012. Stochastic outlier selection. Technical report TiCC TR 2012-001, Tilburg University, Tilburg Center for Cognition and Communication, Tilburg, The Netherlands.

.. [#Kingma2013Auto] Kingma, D.P. and Welling, M., 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

.. [#Kriegel2008Angle] Kriegel, H.P. and Zimek, A., 2008, August. Angle-based outlier detection in high-dimensional data. In *KDD '08*\ , pp. 444-452. ACM.

.. [#Kriegel2009Outlier] Kriegel, H.P., Kröger, P., Schubert, E. and Zimek, A., 2009, April. Outlier detection in axis-parallel subspaces of high dimensional data. In *Pacific-Asia Conference on Knowledge Discovery and Data Mining*\ , pp. 831-838. Springer, Berlin, Heidelberg.

.. [#Lazarevic2005Feature] Lazarevic, A. and Kumar, V., 2005, August. Feature bagging for outlier detection. In *KDD '05*. 2005.

.. [#Li2019MADGAN] Li, D., Chen, D., Jin, B., Shi, L., Goh, J. and Ng, S.K., 2019, September. MAD-GAN: Multivariate anomaly detection for time series data with generative adversarial networks. In *International Conference on Artificial Neural Networks* (pp. 703-716). Springer, Cham.

.. [#Li2020COPOD] Li, Z., Zhao, Y., Botta, N., Ionescu, C. and Hu, X. COPOD: Copula-Based Outlier Detection. *IEEE International Conference on Data Mining (ICDM)*, 2020.

.. [#Liu2008Isolation] Liu, F.T., Ting, K.M. and Zhou, Z.H., 2008, December. Isolation forest. In *International Conference on Data Mining*\ , pp. 413-422. IEEE.

.. [#Liu2019Generative] Liu, Y., Li, Z., Zhou, C., Jiang, Y., Sun, J., Wang, M. and He, X., 2019. Generative adversarial active learning for unsupervised outlier detection. *IEEE Transactions on Knowledge and Data Engineering*.

.. [#Papadimitriou2003LOCI] Papadimitriou, S., Kitagawa, H., Gibbons, P.B. and Faloutsos, C., 2003, March. LOCI: Fast outlier detection using the local correlation integral. In *ICDE '03*, pp. 315-326. IEEE.

.. [#Pevny2016Loda] Pevný, T., 2016. Loda: Lightweight on-line detector of anomalies. *Machine Learning*, 102(2), pp.275-304.

.. [#Perini2020Quantifying] Perini, L., Vercruyssen, V., Davis, J. Quantifying the confidence of anomaly detectors in their example-wise predictions. In *Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML-PKDD)*, 2020.

.. [#Ramaswamy2000Efficient] Ramaswamy, S., Rastogi, R. and Shim, K., 2000, May. Efficient algorithms for mining outliers from large data sets. *ACM Sigmod Record*\ , 29(2), pp. 427-438.

.. [#Rousseeuw1999A] Rousseeuw, P.J. and Driessen, K.V., 1999. A fast algorithm for the minimum covariance determinant estimator. *Technometrics*\ , 41(3), pp.212-223.

.. [#Ruff2018Deep] Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S.A., Binder, A., Müller, E. and Kloft, M., 2018, July. Deep one-class classification. In *International conference on machine learning* (pp. 4393-4402). PMLR.

.. [#Scholkopf2001Estimating] Scholkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J. and Williamson, R.C., 2001. Estimating the support of a high-dimensional distribution. *Neural Computation*, 13(7), pp.1443-1471.

.. [#Shyu2003A] Shyu, M.L., Chen, S.C., Sarinnapakorn, K. and Chang, L., 2003. A novel anomaly detection scheme based on principal component classifier. *MIAMI UNIV CORAL GABLES FL DEPT OF ELECTRICAL AND COMPUTER ENGINEERING*.

.. [#Tang2002Enhancing] Tang, J., Chen, Z., Fu, A.W.C. and Cheung, D.W., 2002, May. Enhancing effectiveness of outlier detections for low density patterns. In *Pacific-Asia Conference on Knowledge Discovery and Data Mining*, pp. 535-548. Springer, Berlin, Heidelberg.

.. [#Wang2020adVAE] Wang, X., Du, Y., Lin, S., Cui, P., Shen, Y. and Yang, Y., 2019. adVAE: A self-adversarial variational autoencoder with Gaussian anomaly prior knowledge for anomaly detection. *Knowledge-Based Systems*.

.. [#Zhao2018XGBOD] Zhao, Y. and Hryniewicki, M.K. XGBOD: Improving Supervised Outlier Detection with Unsupervised Representation Learning. *IEEE International Joint Conference on Neural Networks*\ , 2018.

.. [#Zhao2019LSCP] Zhao, Y., Nasrullah, Z., Hryniewicki, M.K. and Li, Z., 2019, May. LSCP: Locally selective combination in parallel outlier ensembles. In *Proceedings of the 2019 SIAM International Conference on Data Mining (SDM)*, pp. 585-593. Society for Industrial and Applied Mathematics.

.. [#Zhao2021SUOD] Zhao, Y., Hu, X., Cheng, C., Wang, C., Wan, C., Wang, W., Yang, J., Bai, H., Li, Z., Xiao, C., Wang, Y., Qiao, Z., Sun, J. and Akoglu, L. (2021). SUOD: Accelerating Large-scale Unsupervised Heterogeneous Outlier Detection. *Conference on Machine Learning and Systems (MLSys)*.
