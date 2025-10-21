import io
import os
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from onnxruntime import InferenceSession

from otokuna.analysis import add_address_coords, add_target_variable, df2Xy
from otokuna.logging import setup_logger


# Global variable to cache the model path across warm starts
_MODEL_LOCAL_PATH = None


def get_model_path(s3_client, model_bucket, model_s3_key, logger):
    """Download the ONNX model from S3 to /tmp if not already cached."""
    global _MODEL_LOCAL_PATH

    if _MODEL_LOCAL_PATH and os.path.exists(_MODEL_LOCAL_PATH):
        logger.info(f"Using cached model from: {_MODEL_LOCAL_PATH}")
        return _MODEL_LOCAL_PATH

    # Download model to /tmp
    local_path = f"/tmp/{Path(model_s3_key).name}"
    logger.info(f"Downloading model from s3://{model_bucket}/{model_s3_key} to {local_path}")
    s3_client.download_file(Bucket=model_bucket, Key=model_s3_key, Filename=local_path)
    logger.info(f"Model downloaded successfully")

    _MODEL_LOCAL_PATH = local_path
    return local_path


def main(event, context):
    """Makes predictions from scraped data and stores the results in the bucket."""
    logger = setup_logger("predict", include_timestamp=False, propagate=False)

    output_bucket = os.environ["OUTPUT_BUCKET"]
    model_bucket = os.environ["MODEL_BUCKET"]
    model_s3_key = os.environ["MODEL_S3_KEY"]
    root_key = event["root_key"]
    scraped_data_key = event["scraped_data_key"]
    prediction_data_key = str(Path(root_key) / "prediction.pickle")

    s3_client = boto3.client("s3")

    # Download model from S3 (cached across warm starts)
    model_filename = get_model_path(s3_client, model_bucket, model_s3_key, logger)
    # Get pickle from bucket and read dataframe from it
    logger.info(f"Getting scraped data from: {scraped_data_key}")
    with io.BytesIO() as stream:
        s3_client.download_fileobj(Bucket=output_bucket, Key=scraped_data_key, Fileobj=stream)
        stream.seek(0)
        df = pd.read_pickle(stream)

    # Preprocess dataframe
    logger.info(f"Preprocessing dataframe")
    df = add_address_coords(df)
    df = add_target_variable(df)
    X, y = df2Xy(df.dropna())

    # Predict
    logger.info(f"Predicting")
    sess = InferenceSession(model_filename)
    onnx_out = sess.run(["predictions"], {"features": X.values.astype(np.float32)})
    y_pred = pd.Series(onnx_out[0].squeeze(), index=y.index).rename("y_pred")
    # Make dataframe with predictions and target from df **prior** to dropna
    prediction_df = df[["y"]].join(y_pred, how="left")

    # Upload result to bucket
    logger.info(f"Uploading results to: {prediction_data_key}")
    with io.BytesIO() as stream:
        prediction_df.to_pickle(stream, compression=None, protocol=5)
        stream.seek(0)
        s3_client.upload_fileobj(Fileobj=stream, Bucket=output_bucket, Key=prediction_data_key)

    event["prediction_data_key"] = prediction_data_key
    return event
