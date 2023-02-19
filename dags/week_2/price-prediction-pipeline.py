from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from airflow.operators.empty import EmptyOperator
from airflow.models.dag import DAG
from airflow.decorators import task, task_group
from airflow.providers.google.cloud.hooks.gcs import GCSHook


TRAINING_DATA_PATH = "week-2/price_prediction_training_data.csv"
DATASET_NORM_WRITE_BUCKET = "corise-airflow2"  # Modify here

VAL_END_INDEX = 31056


@task
def read_dataset_norm():
    """
    Read dataset norm from storage

    CHALLENGE have this automatically read using a sensor
    """

    from airflow.providers.google.cloud.hooks.gcs import GCSHook
    import io

    client = GCSHook().get_conn()
    read_bucket = client.bucket(DATASET_NORM_WRITE_BUCKET)
    dataset_norm = pd.read_csv(
        io.BytesIO(read_bucket.blob(TRAINING_DATA_PATH).download_as_bytes())
    ).to_numpy()

    return dataset_norm


def multivariate_data(
    dataset, data_indices, history_size, target_size, step, single_step=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce subset of dataset indexed by data_indices, with a window size of history_size hours
    """

    target = dataset[:, -1]
    data = []
    labels = []
    for i in data_indices:
        indices = range(i, i + history_size, step)
        # If within the last 23 hours in the dataset, skip
        if i + history_size > len(dataset) - 1:
            continue
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i : i + target_size])
    return np.array(data), np.array(labels)


def train_xgboost(X_train, y_train, X_val, y_val) -> xgb.Booster:
    """
    Train xgboost model using training set and evaluated against evaluation set, using
        a set of model parameters
    """

    X_train_xgb = X_train.reshape(-1, X_train.shape[1] * X_train.shape[2])
    X_val_xgb = X_val.reshape(-1, X_val.shape[1] * X_val.shape[2])
    param = {
        "eta": 0.03,
        "max_depth": 180,
        "subsample": 1.0,
        "colsample_bytree": 0.95,
        "alpha": 0.1,
        "lambda": 0.15,
        "gamma": 0.1,
        "objective": "reg:linear",
        "eval_metric": "rmse",
        "silent": 1,
        "min_child_weight": 0.1,
        "n_jobs": -1,
    }
    dtrain = xgb.DMatrix(X_train_xgb, y_train)
    dval = xgb.DMatrix(X_val_xgb, y_val)
    eval_list = [(dtrain, "train"), (dval, "eval")]
    xgb_model = xgb.train(param, dtrain, 10, eval_list, early_stopping_rounds=3)
    return xgb_model


@task
def produce_indices() -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Produce zipped list of training and validation indices

    Each pair of training and validation indices should not overlap, and the
    training indices should never exceed the max of VAL_END_INDEX.

    The number of pairs produced here will be equivalent to the number of
    mapped 'format_data_and_train_model' tasks you have
    """
    from typing import List, Tuple

    indices_list = []
    for i in range(3):
        # Generate a random number of samples between 100 and 1000
        n_samples = np.random.randint(100, VAL_END_INDEX)
        # Shuffle the indices of the samples
        shuffled_indices = np.random.permutation(n_samples)
        # Determine the maximum index for the training set
        max_train_index = int((n_samples - 1) * 0.8)
        # Split the shuffled indices into training and validation indices
        train_indices = shuffled_indices[: max_train_index + 1]
        val_indices = shuffled_indices[max_train_index + 1 :]
        # Return a list with a single tuple of the training and validation indices
        indices_list.append(tuple([train_indices, val_indices]))

    return indices_list


@task
def format_data_and_train_model(
    dataset_norm: np.ndarray, indices: Tuple[np.ndarray, np.ndarray]
) -> xgb.Booster:
    """
    Extract training and validation sets and labels, and train a model with a given
    set of training and validation indices
    """
    past_history = 24
    future_target = 0
    train_indices, val_indices = indices
    print(f"train_indices is {train_indices}, val_indices is {val_indices}")
    X_train, y_train = multivariate_data(
        dataset_norm,
        train_indices,
        past_history,
        future_target,
        step=1,
        single_step=True,
    )
    X_val, y_val = multivariate_data(
        dataset_norm, val_indices, past_history, future_target, step=1, single_step=True
    )
    model = train_xgboost(X_train, y_train, X_val, y_val)
    print(f"Model eval score is {model.best_score}")

    return model


@task
def select_best_model(models: List[xgb.Booster]):
    """
    Select model that generalizes the best against the validation set, and
    write this to GCS. The best_score is an attribute of the model, and corresponds to
    the highest eval score yielded during training.
    """
    from airflow.providers.google.cloud.hooks.gcs import GCSHook
    import pickle

    best_model = max(models, key=lambda x: x.best_score)
    # Serialize the model using pickle
    model_data = pickle.dumps(best_model)
    # Upload the serialized model to Google Cloud Storage using GCSHook
    hook = GCSHook()
    model_uri = "my_model.pickle"
    hook.upload(bucket_name="corise-airflow2", object_name=model_uri, data=model_data)


@task_group
def train_and_select_best_model():
    """
    Task group responsible for training XGBoost models to predict energy prices, including:
       1. Reading the dataset norm from GCS
       2. Producing a list of training and validation indices numpy array tuples,
       3. Mapping each element of that list onto the indices argument of format_data_and_train_model
       4. Calling select_best_model on the output of all of the mapped tasks to select the best model and
          write it to GCS

    Using different train/val splits, train multiple models and select the one with the best evaluation score.
    """

    past_history = 24
    future_target = 0

    # TODO: Modify here to select best model and save it to GCS, using above methods including
    # format_data_and_train_model, produce_indices, and select_best_model

    # 1. Reading the dataset norm from GCS
    # dataset_norm=read_dataset_norm()

    # 2. Producing a list of training and validation indices numpy array tuples
    # indices=produce_indices()
    # 3. Mapping each element of that list onto the indices argument of format_data_and_train_model
    # 4. Calling select_best_model on the output of all of the mapped tasks to select the best model and write it

    trained_models = format_data_and_train_model.partial(
        dataset_norm=read_dataset_norm()
    ).expand(indices=produce_indices())
    select_best_model(trained_models)


@task
def sensor_task():
    from airflow.sensors.external_task_sensor import ExternalTaskSensor

    waiting_for_previous_task = ExternalTaskSensor(
        task_id="external_task_sensor",
        poke_interval=60,
        timeout=180,
        retries=2,
        external_task_id="prepare_model_inputs",
        external_dag_id="energy_price_prediction_features",
        execution_delta=timedelta(minutes=3),
    )


with DAG(
    "energy_price_prediction",
    schedule_interval=None,
    start_date=datetime(2021, 1, 1),
    tags=["model_training"],
    render_template_as_native_obj=True,
    concurrency=5,
) as dag:
    sensor_task() >> train_and_select_best_model()
