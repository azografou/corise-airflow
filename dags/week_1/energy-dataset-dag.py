from datetime import datetime
from typing import List
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.decorators import dag, task # DAG and task decorators for interfacing with the TaskFlow API
import pandas as pd
from zipfile import ZipFile


@dag(
    # This defines how often your DAG will run, or the schedule by which your DAG runs. In this case, this DAG
    # will run daily
    schedule_interval="@daily",
    # This DAG is set to run for the first time on January 1, 2021. Best practice is to use a static
    # start_date. Subsequent DAG runs are instantiated based on scheduler_interval
    start_date=datetime(2021, 1, 1),
    # When catchup=False, your DAG will only run for the latest schedule_interval. In this case, this means
    # that tasks will not be run between January 1, 2021 and 30 mins ago. When turned on, this DAG's first
    # run will be for the next 30 mins, per the schedule_interval
    catchup=False,
    default_args={
        "retries": 2, # If a task fails, it will retry 2 times.
    },
    tags=['example']) # If set, this tag is shown in the DAG view of the Airflow UI
def energy_dataset_dag():
    """
    ### Basic ETL Dag
    This is a simple ETL data pipeline example that demonstrates the use of
    the TaskFlow API using two simple tasks to extract data from a zipped folder
    and load it to GCS.

    """
    @task
    def bucket():

        client = GCSHook()
        connection = client.get_conn()

        try:
            connection.get_bucket('corise-airflow2')
            print('Bucket corise-airflow2 already exists.')
        except :
            client.create_bucket('corise-airflow2') 

    @task
    def extract() -> List[pd.DataFrame]:
        """
        #### Extract task
        A simple task that loads each file in the zipped file into a dataframe,
        building a list of dataframes that is returned.

        """
        # open zipped dataset
        with ZipFile("/usr/local/airflow/dags/data/energy-consumption-generation-prices-and-weather.zip")as data:
            csv_files = [file for file in data.namelist() if file.endswith('.csv')]
            for file in data.namelist():  
                if file.endswith(".csv"):
                    data.extract(file, "output_dir/")

        # Load each CSV file into a separate Pandas dataframe
        df=list()
        for  csv_file in csv_files:
            df.append(pd.read_csv(f'output_dir/{csv_file}'))
    
        return df 

    @task
    def load(unzip_result: List[pd.DataFrame],):
        """
        #### Load task
        A simple "load" task that takes in the result of the "transform" task, prints out the 
        schema, and then writes the data into GCS as parquet files.
        """
        # Create the GCS client
        client = GCSHook()   
        data_types = ['generation', 'weather']

        # Loop over the list and write each dataframe as a parquet file to GCS
        for i, dataframe in enumerate(unzip_result):
            unzip_result[i].to_parquet(f'{data_types[i]}.parquet')

            client.upload(
                        bucket_name='corise-airflow2',
                        object_name=f'{data_types[i]}.parquet',
                        filename=f'./{data_types[i]}.parquet',
                        mime_type='application/vnd.apache.parquet')
        # GCSHook uses google_cloud_default connection by default, so we can easily create a GCS client using it
        # https://github.com/apache/airflow/blob/207f65b542a8aa212f04a9d252762643cfd67a74/airflow/providers/google/cloud/hooks/gcs.py#L133

        # The google cloud storage github repo has a helpful example for writing from pandas to GCS:
        # https://github.com/googleapis/python-storage/blob/main/samples/snippets/storage_fileio_pandas.py

    # Task linking logic
    
    load(bucket() >> extract())

energy_dataset_dag = energy_dataset_dag()