"""Data fetching and processing flow."""

from datetime import datetime
from typing import List, Optional
from prefect import flow, get_run_logger
from arxiv_pipeline.tasks.fetch import fetch_papers
from arxiv_pipeline.tasks.process import process_papers
from arxiv_pipeline.tasks.storage import store_raw_papers, store_processed_papers
from arxiv_pipeline.data.schemas import Paper, ProcessedPaper


@flow(name="fetch-and-process-papers")
def fetch_and_process_papers(
    date: Optional[datetime] = None,
    max_results: Optional[int] = None,
    categories: Optional[List[str]] = None,
    store_raw: bool = True,
    store_processed: bool = True
) -> tuple[List[Paper], List[ProcessedPaper]]:
    """
    Main flow for fetching and processing arXiv papers.
    
    Args:
        date: Date to fetch papers from (defaults to today)
        max_results: Maximum number of papers to fetch
        categories: List of arXiv categories to filter by
        store_raw: Whether to store raw papers in MinIO
        store_processed: Whether to store processed papers in MinIO
    
    Returns:
        Tuple of (raw papers, processed papers)
    """
    logger = get_run_logger()
    
    if date is None:
        date = datetime.now()
    
    logger.info(f"Starting data flow for date: {date.strftime('%Y-%m-%d')}")
    
    # Fetch papers from arXiv
    papers = fetch_papers(
        date=date,
        max_results=max_results,
        categories=categories
    )
    
    if not papers:
        logger.warning("No papers fetched from arXiv")
        return [], []
    
    # Store raw papers if requested
    if store_raw:
        store_raw_papers(papers, date=date)
    
    # Process papers (split abstracts into sentences)
    processed_papers = process_papers(papers)
    
    # Store processed papers if requested
    if store_processed:
        store_processed_papers(processed_papers, date=date)
    
    logger.info(
        f"Completed data flow: {len(papers)} papers fetched, "
        f"{len(processed_papers)} papers processed"
    )
    
    return papers, processed_papers


@flow(name="fetch-papers-only")
def fetch_papers_only(
    date: Optional[datetime] = None,
    max_results: Optional[int] = None,
    categories: Optional[List[str]] = None
) -> List[Paper]:
    """
    Flow to only fetch papers without processing.
    
    Args:
        date: Date to fetch papers from (defaults to today)
        max_results: Maximum number of papers to fetch
        categories: List of arXiv categories to filter by
    
    Returns:
        List of Paper objects
    """
    logger = get_run_logger()
    
    if date is None:
        date = datetime.now()
    
    logger.info(f"Fetching papers for date: {date.strftime('%Y-%m-%d')}")
    
    papers = fetch_papers(
        date=date,
        max_results=max_results,
        categories=categories
    )
    
    # Store raw papers
    store_raw_papers(papers, date=date)
    
    logger.info(f"Fetched and stored {len(papers)} papers")
    return papers

