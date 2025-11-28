"""Tasks for storing data in MinIO."""

import json
from datetime import datetime
from pathlib import Path
from typing import List
from prefect import task, get_run_logger
from arxiv_pipeline.data.schemas import Paper, ProcessedPaper
from arxiv_pipeline.utils.minio_client import (
    upload_json,
    ensure_bucket_exists,
    object_exists
)
from arxiv_pipeline.utils.config import get_config


@task
def store_raw_papers(papers: List[Paper], date: datetime | None = None) -> str:
    """
    Store raw papers in MinIO.
    
    Args:
        papers: List of Paper objects
        date: Date for organizing storage (defaults to today)
    
    Returns:
        Path where papers were stored
    """
    logger = get_run_logger()
    config = get_config()
    
    if date is None:
        date = datetime.now()
    
    # Organize by date
    date_str = date.strftime("%Y-%m-%d")
    object_name = f"{config.storage.raw_data_path}/{date_str}/papers.json"
    
    # Convert to dict for JSON serialization
    papers_dict = [paper.model_dump(mode="json") for paper in papers]
    
    # Store in MinIO
    bucket_name = config.storage.bucket_name
    ensure_bucket_exists(bucket_name)
    upload_json(bucket_name, object_name, papers_dict)
    
    logger.info(f"Stored {len(papers)} raw papers to {bucket_name}/{object_name}")
    return object_name


@task
def store_processed_papers(
    processed_papers: List[ProcessedPaper],
    date: datetime | None = None
) -> str:
    """
    Store processed papers in MinIO.
    
    Args:
        processed_papers: List of ProcessedPaper objects
        date: Date for organizing storage (defaults to today)
    
    Returns:
        Path where papers were stored
    """
    logger = get_run_logger()
    config = get_config()
    
    if date is None:
        date = datetime.now()
    
    # Organize by date
    date_str = date.strftime("%Y-%m-%d")
    object_name = f"{config.storage.processed_data_path}/{date_str}/processed_papers.json"
    
    # Convert to dict for JSON serialization
    papers_dict = [paper.model_dump(mode="json") for paper in processed_papers]
    
    # Store in MinIO
    bucket_name = config.storage.bucket_name
    ensure_bucket_exists(bucket_name)
    upload_json(bucket_name, object_name, papers_dict)
    
    logger.info(f"Stored {len(processed_papers)} processed papers to {bucket_name}/{object_name}")
    return object_name


@task
def store_paper_by_id(paper: Paper, date: datetime | None = None) -> str:
    """
    Store a single paper by its ID.
    
    Args:
        paper: Paper object
        date: Date for organizing storage (defaults to today)
    
    Returns:
        Path where paper was stored
    """
    logger = get_run_logger()
    config = get_config()
    
    if date is None:
        date = datetime.now()
    
    date_str = date.strftime("%Y-%m-%d")
    object_name = f"{config.storage.raw_data_path}/{date_str}/papers/{paper.arxiv_id}.json"
    
    bucket_name = config.storage.bucket_name
    ensure_bucket_exists(bucket_name)
    upload_json(bucket_name, object_name, paper.model_dump(mode="json"))
    
    logger.debug(f"Stored paper {paper.arxiv_id} to {bucket_name}/{object_name}")
    return object_name


@task
def load_raw_papers(date: datetime | None = None) -> List[Paper]:
    """
    Load raw papers from MinIO.
    
    Args:
        date: Date to load papers from (defaults to today)
    
    Returns:
        List of Paper objects
    """
    from arxiv_pipeline.utils.minio_client import download_json
    
    logger = get_run_logger()
    config = get_config()
    
    if date is None:
        date = datetime.now()
    
    date_str = date.strftime("%Y-%m-%d")
    object_name = f"{config.storage.raw_data_path}/{date_str}/papers.json"
    bucket_name = config.storage.bucket_name
    
    if not object_exists(bucket_name, object_name):
        logger.warning(f"No papers found for date {date_str}")
        return []
    
    papers_dict = download_json(bucket_name, object_name)
    papers = [Paper(**paper_dict) for paper_dict in papers_dict]
    
    logger.info(f"Loaded {len(papers)} papers from {bucket_name}/{object_name}")
    return papers


@task
def load_processed_papers(date: datetime | None = None) -> List[ProcessedPaper]:
    """
    Load processed papers from MinIO.
    
    Args:
        date: Date to load papers from (defaults to today)
    
    Returns:
        List of ProcessedPaper objects
    """
    from arxiv_pipeline.utils.minio_client import download_json
    
    logger = get_run_logger()
    config = get_config()
    
    if date is None:
        date = datetime.now()
    
    date_str = date.strftime("%Y-%m-%d")
    object_name = f"{config.storage.processed_data_path}/{date_str}/processed_papers.json"
    bucket_name = config.storage.bucket_name
    
    if not object_exists(bucket_name, object_name):
        logger.warning(f"No processed papers found for date {date_str}")
        return []
    
    papers_dict = download_json(bucket_name, object_name)
    processed_papers = [ProcessedPaper(**paper_dict) for paper_dict in papers_dict]
    
    logger.info(f"Loaded {len(processed_papers)} processed papers from {bucket_name}/{object_name}")
    return processed_papers

