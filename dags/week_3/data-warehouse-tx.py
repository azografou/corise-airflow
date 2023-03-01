from datetime import datetime

import pandas as pd
from typing import List

from airflow.operators.empty import EmptyOperator
from airflow.decorators import dag, task, task_group

PROJECT_ID = "airflow-week-1-377408"
DESTINATION_BUCKET = "corise-airflow1"
BQ_DATASET_NAME = "timeseries_energy"

DATA_TYPES = ["generation", "weather"]


normalized_columns = {
    "generation": {
        "time": "time",
        "columns": [
            "total_load_actual",
            "price_day_ahead",
            "price_actual",
            "generation_fossil_hard_coal",
            "generation_fossil_gas",
            "generation_fossil_brown_coal_lignite",
            "generation_fossil_oil",
            "generation_other_renewable",
            "generation_waste",
            "generation_biomass",
            "generation_other",
            "generation_solar",
            "generation_hydro_water_reservoir",
            "generation_nuclear",
            "generation_hydro_run_of_river_and_poundage",
            "generation_wind_onshore",
            "generation_hydro_pumped_storage_consumption",
        ],
    },
    "weather": {
        "time": "dt_iso",
        "columns": [
            "city_name",
            "temp",
            "pressure",
            "humidity",
            "wind_speed",
            "wind_deg",
            "rain_1h",
            "rain_3h",
            "snow_3h",
            "clouds_all",
        ],
    },
}


@dag(
    schedule_interval=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
)
def data_warehouse_transform_dag():
    """
    ### Data Warehouse Transform DAG
    This DAG performs four operations:
        1. Extracts zip file into two dataframes
        2. Loads these dataframes into parquet files on GCS, with valid column names
        3. Builds external tables on top of these parquet files
        4. Builds normalized views on top of the external tables
        5. Builds a joined view on top of the normalized views, joined on time
    """

    @task
    def extract() -> List[pd.DataFrame]:
        """
        #### Extract task
        A simple task that loads each file in the zipped file into a dataframe,
        building a list of dataframes that is returned


        """
        from zipfile import ZipFile

        filename = "/usr/local/airflow/dags/data/energy-consumption-generation-prices-and-weather.zip"
        dfs = [
            pd.read_csv(ZipFile(filename).open(i)) for i in ZipFile(filename).namelist()
        ]
        return dfs

    @task
    def load(unzip_result: List[pd.DataFrame]):
        """
        #### Load task
        A simple "load" task that takes in the result of the "extract" task, formats
        columns to be BigQuery-compliant, and writes data to GCS.
        """

        from airflow.providers.google.cloud.hooks.gcs import GCSHook

        client = GCSHook().get_conn()
        bucket = client.get_bucket(DESTINATION_BUCKET)

        for index, df in enumerate(unzip_result):
            df.columns = df.columns.str.replace(" ", "_")
            df.columns = df.columns.str.replace("/", "_")
            df.columns = df.columns.str.replace("-", "_")
            bucket.blob(f"week-3/{DATA_TYPES[index]}.parquet").upload_from_string(
                df.to_parquet(), "text/parquet"
            )
            print(df.dtypes)

    @task_group
    def create_bigquery_dataset():
        from airflow.providers.google.cloud.operators.bigquery import BigQueryCreateEmptyDatasetOperator
        # This is where your tables and views will be created
        create_dataset = BigQueryCreateEmptyDatasetOperator(
            task_id="create_dataset", dataset_id=BQ_DATASET_NAME
        )
        # TODO Modify here to create a BigQueryDataset if one does not already exist
        return create_dataset

    @task_group
    def create_external_tables():
        from airflow.providers.google.cloud.operators.bigquery import (
            BigQueryCreateExternalTableOperator,
        )

        # TODO Modify here to produce two external tables, one for each data type, referencing the data stored in GCS
        # Create a task for each external table
        for table in DATA_TYPES:
            external_table = BigQueryCreateExternalTableOperator(
                task_id=f"create_external_table_{table}",
                table_resource={
                    "tableReference": {
                        "projectId": PROJECT_ID,
                        "datasetId": BQ_DATASET_NAME,
                        "tableId": table,
                    },
                    "externalDataConfiguration": {
                        "sourceFormat": "PARQUET",
                        "sourceUris": [f"gs://{DESTINATION_BUCKET}/{table}.parquet"],
                        "autodetect": True,
                    },
                },
            )
        # When using the BigQueryCreateExternalTableOperator, it's suggested you use the table_resource
        # field to specify DDL configuration parameters. If you don't, then you will see an error
        # related to the built table_resource specifying csvOptions even though the desired format is
        # PARQUET.

    def produce_select_statement(
        timestamp_column: str, table_name: str, columns: List[str]
    ) -> str:
        # TODO Modify here to produce a select statement by casting 'timestamp_column' to
        # TIMESTAMP type, and selecting all of the columns in 'columns'

        # Cast the timestamp column to TIMESTAMP type
        timestamp_cast = f"CAST({timestamp_column} AS TIMESTAMP)"

        # Join all columns in 'columns' using commas
        columns_joined = ", ".join(columns)

        # Build the final select statement
        select_statement = f"SELECT {timestamp_cast} AS timestamp, {columns_joined} FROM {BQ_DATASET_NAME}.{table_name}"

        return select_statement

        # pass

    @task_group
    def produce_normalized_views():
        from airflow.providers.google.cloud.operators.bigquery import (
            BigQueryCreateEmptyTableOperator,
        )

        # TODO Modify here to produce views for each of the datasources, capturing only the essential
        # columns specified in normalized_columns. A key step at this stage is to convert the relevant
        # columns in each datasource from string to time. The utility function 'produce_select_statement'
        # accepts the timestamp column, and essential columns for each of the datatypes and build a
        # select statement ptogrammatically, which can then be passed to the Airflow Operators.
        for table in DATA_TYPES:
            create_view = BigQueryCreateEmptyTableOperator(
                task_id=f"create_view_{table}",
                dataset_id=BQ_DATASET_NAME,
                table_id=f"{table}_normalized_view",
                view={
                    "query": produce_select_statement(
                        normalized_columns[table]["time"],
                        table,
                        normalized_columns[table]["columns"],
                    ),
                    "useLegacySql": False,
                },
            )

    @task_group
    def produce_joined_view():
        from airflow.providers.google.cloud.operators.bigquery import (
            BigQueryCreateEmptyTableOperator,
        )

        # TODO Modify here to produce a view that joins the two normalized views on time
        create_view = BigQueryCreateEmptyTableOperator(
            task_id="create_joined_view",
            dataset_id=BQ_DATASET_NAME,
            table_id="joined_view",
            view={
                "query": f"""
            SELECT *
            FROM 
            {PROJECT_ID}.{BQ_DATASET_NAME}.generation_normalized_view a
            FULL OUTER JOIN 
            {PROJECT_ID}.{BQ_DATASET_NAME}.weather_normalized_view b 
            USING (timestamp)
            """,
                "useLegacySql": False,
            },
        )

    unzip_task = extract()
    load_task = load(unzip_task)
    create_bigquery_dataset_task = create_bigquery_dataset()
    load_task >> create_bigquery_dataset_task
    external_table_task = create_external_tables()
    create_bigquery_dataset_task >> external_table_task
    normal_view_task = produce_normalized_views()
    external_table_task >> normal_view_task
    joined_view_task = produce_joined_view()
    normal_view_task >> joined_view_task


data_warehouse_transform_dag = data_warehouse_transform_dag()
