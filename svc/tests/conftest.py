import os

import pytest


@pytest.fixture
def set_environ():
    os.environ["OUTPUT_BUCKET"] = "somebucket"
    # Set default AWS region to us-east-1 for moto/boto3 compatibility
    # This avoids LocationConstraint errors when creating S3 buckets
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    yield
    os.environ.pop("OUTPUT_BUCKET")
    os.environ.pop("AWS_DEFAULT_REGION", None)
