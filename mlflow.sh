#!/bin/bash

mlflow ui --backend-store-uri sqlite:///mlflow/tracking1.db --port 5000 &
x-www-browser http://localhost:5000/
