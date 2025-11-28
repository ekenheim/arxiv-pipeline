"""Recommendation generation flow."""

from datetime import datetime
from typing import List, Optional
from prefect import flow, get_run_logger
from arxiv_pipeline.tasks.recommendations import (
    load_latest_models,
    generate_recommendations_for_papers,
    store_recommendations
)
from arxiv_pipeline.tasks.storage import load_processed_papers
from arxiv_pipeline.data.schemas import PaperRecommendations


@flow(name="generate-recommendations")
def generate_recommendations(
    date: Optional[datetime] = None,
    embedding_model_id: Optional[str] = None,
    threshold: float = 0.5,
    store: bool = True
) -> List[PaperRecommendations]:
    """
    Main flow for generating recommendations.
    
    Args:
        date: Date to load papers from (defaults to today)
        embedding_model_id: Optional specific embedding model ID
        threshold: Prediction threshold
        store: Whether to store recommendations in MinIO
    
    Returns:
        List of PaperRecommendations objects
    """
    logger = get_run_logger()
    
    if date is None:
        date = datetime.now()
    
    logger.info(f"Starting recommendation generation for date: {date.strftime('%Y-%m-%d')}")
    
    # Load processed papers
    processed_papers = load_processed_papers(date=date)
    
    if not processed_papers:
        logger.warning("No processed papers found")
        return []
    
    # Load models
    embedding_model, classifiers = load_latest_models(embedding_model_id=embedding_model_id)
    
    if not classifiers:
        logger.warning("No classifiers loaded, cannot generate recommendations")
        return []
    
    # Generate recommendations
    recommendations = generate_recommendations_for_papers(
        processed_papers,
        embedding_model,
        classifiers,
        threshold=threshold
    )
    
    # Store if requested
    if store:
        store_recommendations(recommendations, date=date)
    
    logger.info(
        f"Generated recommendations for {len(recommendations)} papers"
    )
    
    return recommendations


@flow(name="generate-recommendations-for-new-papers")
def generate_recommendations_for_new_papers(
    threshold: float = 0.5
) -> List[PaperRecommendations]:
    """
    Generate recommendations for today's new papers.
    
    Args:
        threshold: Prediction threshold
    
    Returns:
        List of PaperRecommendations objects
    """
    return generate_recommendations(
        date=datetime.now(),
        threshold=threshold,
        store=True
    )

