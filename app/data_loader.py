from google.cloud import bigquery
import pandas as pd

PROJECT_ID = "drillsense-ai"
DATASET = "drillsense"
TABLE = "sensor_data"


def load_data():

    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
    """

    df = client.query(query).to_dataframe()

    return df