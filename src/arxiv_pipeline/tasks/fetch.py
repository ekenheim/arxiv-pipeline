"""Tasks for fetching papers from arXiv."""

from datetime import datetime, timedelta
from typing import List
import arxiv
from prefect import task, get_run_logger
from arxiv_pipeline.data.schemas import Paper, Author
from arxiv_pipeline.utils.config import get_config


@task(retries=3, retry_delay_seconds=60)
def fetch_papers(
    date: datetime | None = None,
    max_results: int | None = None,
    categories: List[str] | None = None
) -> List[Paper]:
    """
    Fetch papers from arXiv API.
    
    Args:
        date: Date to fetch papers from (defaults to today)
        max_results: Maximum number of papers to fetch
        categories: List of arXiv categories to filter by
    
    Returns:
        List of Paper objects
    """
    logger = get_run_logger()
    config = get_config()
    
    if date is None:
        date = datetime.now()
    
    if max_results is None:
        max_results = config.data.max_papers_per_day
    
    # Format date for arXiv query
    date_str = date.strftime("%Y%m%d")
    
    # Build query
    query_parts = [f"submittedDate:[{date_str}000000 TO {date_str}235959]"]
    
    if categories:
        cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
        query_parts.append(f"({cat_query})")
    
    query = " AND ".join(query_parts)
    logger.info(f"Fetching papers with query: {query}")
    
    # Search arXiv
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    try:
        for result in search.results():
            # Extract authors
            authors = [
                Author(name=author.name)
                for author in result.authors
            ]
            
            # Extract categories
            paper_categories = result.categories if hasattr(result, 'categories') else []
            primary_category = result.primary_category if hasattr(result, 'primary_category') else None
            
            paper = Paper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                authors=authors,
                abstract=result.summary,
                categories=paper_categories,
                published=result.published,
                updated=result.updated if hasattr(result, 'updated') else None,
                pdf_url=result.pdf_url if hasattr(result, 'pdf_url') else None,
                primary_category=primary_category
            )
            papers.append(paper)
        
        logger.info(f"Fetched {len(papers)} papers from arXiv")
        return papers
    
    except Exception as e:
        logger.error(f"Error fetching papers from arXiv: {e}")
        raise


@task
def fetch_papers_by_date_range(
    start_date: datetime,
    end_date: datetime,
    max_results_per_day: int | None = None,
    categories: List[str] | None = None
) -> List[Paper]:
    """
    Fetch papers from arXiv for a date range.
    
    Args:
        start_date: Start date
        end_date: End date
        max_results_per_day: Maximum papers per day
        categories: List of arXiv categories to filter by
    
    Returns:
        List of Paper objects
    """
    logger = get_run_logger()
    all_papers = []
    
    current_date = start_date
    while current_date <= end_date:
        papers = fetch_papers(
            date=current_date,
            max_results=max_results_per_day,
            categories=categories
        )
        all_papers.extend(papers)
        current_date += timedelta(days=1)
    
    logger.info(f"Fetched {len(all_papers)} papers total from {start_date} to {end_date}")
    return all_papers

