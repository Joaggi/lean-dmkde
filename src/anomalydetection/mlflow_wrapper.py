import logging
import numpy as np

def start_run(mlflow, run_name):
    while True:
        logging.debug("start_run")
        try:
            return mlflow.start_run(run_name=run_name)
        except Exception as Argument:
            logging.exception("Error ")
            if "is already active" in str(Argument):
                mlflow.end_run()


def end_run(mlflow):
    while True:
        logging.debug("end_run")
        try:
            mlflow.end_run()
            break
        except Exception as Argument:
            logging.exception("Error ")


def log_metric(mlflow, name, metric):
    while True:
        logging.debug("log_metric")
        try:
            if isinstance(metric, np.generic):
                metric = metric.item()

            mlflow.log_metric(name, metric)
            break
        except Exception as Argument:
            logging.exception("Error ")

def log_metrics(mlflow, metrics):
    while True:
        logging.debug("log_metrics")
        try:
            mlflow.log_metrics(metrics)
            break
        except Exception as Argument:
            logging.exception("Error ")


def log_params(mlflow, metrics):
    while True:
        logging.debug("log_params")
        try:
            mlflow.log_params(metrics)
            break
        except Exception as Argument:
            logging.exception("Error ")

def log_artifact(mlflow, artifact):
    while True:
        logging.debug("log_artifact")
        try:
            mlflow.log_artifact(artifact)
            break
        except Exception as Argument:
            logging.exception("Error ")

def get_experiment_by_name(mlflow, name_of_experiment):
    while True:
        logging.debug("get_experiment_by_name")

        try:
            return mlflow.get_experiment_by_name(name_of_experiment)
        except Exception as Argument:
            logging.exception("Error ")

def search_runs(mlflow, experiment_ids, filter_string, run_view_type, output_format):
    while True:
        logging.debug("search_runs")

        try:
            return mlflow.search_runs(experiment_ids = experiment_ids, filter_string = filter_string, run_view_type = run_view_type, output_format = output_format)
        except Exception as Argument:
            logging.exception("Error in search_runs")
            continue


def set_tracking_uri(mlflow, tracking_uri):
    while True:
        logging.debug("set_tracking_uri")

        try:
            mlflow.set_tracking_uri(tracking_uri)
            break
        except Exception as Argument:
            logging.exception("Error ")


def set_registry_uri(mlflow, registry_uri):
    while True:
        logging.debug("set_registry_uri")

        try:
            mlflow.set_registry_uri(registry_uri)
            break
        except Exception as Argument:
            logging.exception("Error ")


def create_experiment(mlflow, name_of_experiment, loc_artifacts):
    while True:
        logging.debug("create_experiment")

        try:
            mlflow.create_experiment(name_of_experiment, loc_artifacts)
            break
        except Exception as Argument:
            logging.exception("Error ")
            if "UNIQUE constraint failed: experiments.name" in str(Argument):
                return
            if "already exists" in str(Argument):
                return




def set_experiment(mlflow, name_of_experiment):
    while True:
        logging.debug("set_experiment")

        try:
            mlflow.set_experiment(name_of_experiment)
            break
        except Exception as Argument:
            logging.exception("Error ")
 
