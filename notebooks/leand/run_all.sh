#!/bin/bash

if [ $1 -eq 0 ] 
then
  #python3 ablat_ae.py 0
  python3 ablat_kde.py 0
  python3 addmkde.py 0
  python3 addmkde_sgd.py 0
  python3 covariance.py 0
  python3 isolation.py 0
  python3 lake.py 0
  python3 lof.py 0
  python3 oneclass.py 0
  python3 pca-dmkde.py 1
else
  python3 pyod-alad.py 1
  python3 pyod-copod.py 1
  python3 pyod-deepsvdd.py 1
  python3 pyod-knn.py 1
  python3 pyod-loda.py 1
  python3 pyod-sogaal.py 1
  python3 pyod-sos.py 1
  python3 pyod-vaebayes.py 1
  python3 leand.py 1
  python3 qadvaeff.py 1
fi
