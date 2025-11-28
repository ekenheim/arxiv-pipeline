"""MinIO client wrapper for Prefect integration."""

import os
import io
import json
from pathlib import Path
from typing import Any, Optional
from minio import Minio
from minio.error import S3Error
from prefect import get_run_logger
from prefect_aws import MinIOCredentials
from prefect_aws.credentials import MinIOCredentials as MinIOCredentialsType


def get_minio_client() -> Minio:
    """Get MinIO client using Prefect credentials block or environment variables."""
    logger = get_run_logger()
    
    try:
        # Try to load from Prefect block
        credentials = MinIOCredentials.load("minio-credentials")
        client = Minio(
            credentials.minio_endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=credentials.minio_root_user or os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=credentials.minio_root_password.get_secret_value() if hasattr(credentials.minio_root_password, 'get_secret_value') else os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
        )
        logger.info("Loaded MinIO credentials from Prefect block")
    except Exception:
        # Fall back to environment variables
        endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        
        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        logger.info("Loaded MinIO credentials from environment variables")
    
    return client


def ensure_bucket_exists(bucket_name: str) -> None:
    """Ensure MinIO bucket exists, create if it doesn't."""
    logger = get_run_logger()
    client = get_minio_client()
    
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info(f"Created bucket: {bucket_name}")
        else:
            logger.debug(f"Bucket already exists: {bucket_name}")
    except S3Error as e:
        logger.error(f"Error ensuring bucket exists: {e}")
        raise


def upload_file(
    bucket_name: str,
    object_name: str,
    file_path: str | Path,
    content_type: str = "application/octet-stream"
) -> None:
    """Upload a file to MinIO."""
    logger = get_run_logger()
    client = get_minio_client()
    ensure_bucket_exists(bucket_name)
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        client.fput_object(
            bucket_name,
            object_name,
            str(file_path),
            content_type=content_type
        )
        logger.info(f"Uploaded {file_path} to {bucket_name}/{object_name}")
    except S3Error as e:
        logger.error(f"Error uploading file: {e}")
        raise


def upload_data(
    bucket_name: str,
    object_name: str,
    data: bytes | str,
    content_type: str = "application/octet-stream"
) -> None:
    """Upload data (bytes or string) to MinIO."""
    logger = get_run_logger()
    client = get_minio_client()
    ensure_bucket_exists(bucket_name)
    
    if isinstance(data, str):
        data = data.encode("utf-8")
    
    data_stream = io.BytesIO(data)
    
    try:
        client.put_object(
            bucket_name,
            object_name,
            data_stream,
            length=len(data),
            content_type=content_type
        )
        logger.info(f"Uploaded data to {bucket_name}/{object_name}")
    except S3Error as e:
        logger.error(f"Error uploading data: {e}")
        raise


def upload_json(
    bucket_name: str,
    object_name: str,
    data: dict | list,
    indent: int = 2
) -> None:
    """Upload JSON data to MinIO."""
    json_str = json.dumps(data, indent=indent)
    upload_data(
        bucket_name,
        object_name,
        json_str,
        content_type="application/json"
    )


def download_file(
    bucket_name: str,
    object_name: str,
    file_path: str | Path
) -> None:
    """Download a file from MinIO."""
    logger = get_run_logger()
    client = get_minio_client()
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        client.fget_object(bucket_name, object_name, str(file_path))
        logger.info(f"Downloaded {bucket_name}/{object_name} to {file_path}")
    except S3Error as e:
        logger.error(f"Error downloading file: {e}")
        raise


def download_data(bucket_name: str, object_name: str) -> bytes:
    """Download data from MinIO as bytes."""
    logger = get_run_logger()
    client = get_minio_client()
    
    try:
        response = client.get_object(bucket_name, object_name)
        data = response.read()
        response.close()
        response.release_conn()
        logger.debug(f"Downloaded data from {bucket_name}/{object_name}")
        return data
    except S3Error as e:
        logger.error(f"Error downloading data: {e}")
        raise


def download_json(bucket_name: str, object_name: str) -> dict | list:
    """Download JSON data from MinIO."""
    data = download_data(bucket_name, object_name)
    return json.loads(data.decode("utf-8"))


def list_objects(
    bucket_name: str,
    prefix: str = "",
    recursive: bool = True
) -> list[str]:
    """List objects in a MinIO bucket."""
    logger = get_run_logger()
    client = get_minio_client()
    
    try:
        objects = client.list_objects(bucket_name, prefix=prefix, recursive=recursive)
        object_names = [obj.object_name for obj in objects]
        logger.debug(f"Listed {len(object_names)} objects with prefix '{prefix}'")
        return object_names
    except S3Error as e:
        logger.error(f"Error listing objects: {e}")
        raise


def object_exists(bucket_name: str, object_name: str) -> bool:
    """Check if an object exists in MinIO."""
    client = get_minio_client()
    
    try:
        client.stat_object(bucket_name, object_name)
        return True
    except S3Error:
        return False


def delete_object(bucket_name: str, object_name: str) -> None:
    """Delete an object from MinIO."""
    logger = get_run_logger()
    client = get_minio_client()
    
    try:
        client.remove_object(bucket_name, object_name)
        logger.info(f"Deleted {bucket_name}/{object_name}")
    except S3Error as e:
        logger.error(f"Error deleting object: {e}")
        raise

