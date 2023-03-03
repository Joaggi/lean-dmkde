#!/bin/bash

mlflow ui --backend-store-uri sqlite:////home/jagallegom/mlflow-persistence/tracking.db --port 5000 --host 0.0.0.0 &
#x-www-browser http://localhost:5000/
