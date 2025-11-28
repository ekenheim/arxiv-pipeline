"""Main orchestration flow that coordinates all sub-flows."""

from datetime import datetime, timedelta
from typing import Optional
from prefect import flow, get_run_logger
from arxiv_pipeline.flows.data_flow import fetch_and_process_papers
from arxiv_pipeline.flows.training_flow import train_models
from arxiv_pipeline.flows.recommendation_flow import generate_recommendations
from arxiv_pipeline.tasks.annotation import load_annotations


@flow(name="arxiv-pipeline-orchestration")
def arxiv_pipeline_orchestration(
    date: Optional[datetime] = None,
    fetch_data: bool = True,
    train_models_flag: bool = False,
    generate_recommendations_flag: bool = True,
    check_annotations: bool = True
) -> dict:
    """
    Main orchestration flow that coordinates all pipeline components.
    
    Args:
        date: Date to process (defaults to today)
        fetch_data: Whether to fetch and process new papers
        train_models_flag: Whether to train models (defaults to False, set True if annotations updated)
        generate_recommendations_flag: Whether to generate recommendations
        check_annotations: Whether to check for new annotations before training
    
    Returns:
        Dictionary with flow results
    """
    logger = get_run_logger()
    
    if date is None:
        date = datetime.now()
    
    logger.info(f"Starting arXiv pipeline orchestration for date: {date.strftime('%Y-%m-%d')}")
    
    results = {
        "date": date.isoformat(),
        "data_fetch": None,
        "model_training": None,
        "recommendations": None
    }
    
    # Step 1: Fetch and process papers
    if fetch_data:
        logger.info("Step 1: Fetching and processing papers...")
        papers, processed_papers = fetch_and_process_papers(date=date)
        results["data_fetch"] = {
            "papers_fetched": len(papers),
            "papers_processed": len(processed_papers)
        }
        logger.info(f"Fetched {len(papers)} papers, processed {len(processed_papers)}")
    else:
        logger.info("Skipping data fetch")
    
    # Step 2: Train models (if requested or if new annotations exist)
    if train_models_flag:
        logger.info("Step 2: Training models...")
        model_paths = train_models(date=date)
        results["model_training"] = model_paths
        logger.info(f"Trained models: {list(model_paths.keys())}")
    elif check_annotations:
        # Check if there are annotations (could trigger training in future)
        annotations = load_annotations()
        if annotations:
            logger.info(f"Found {len(annotations)} annotations (training not requested)")
        else:
            logger.info("No annotations found")
    
    # Step 3: Generate recommendations
    if generate_recommendations_flag:
        logger.info("Step 3: Generating recommendations...")
        recommendations = generate_recommendations(date=date)
        results["recommendations"] = {
            "papers_with_recommendations": len(recommendations),
            "total_recommendations": sum(
                len(rec.recommendations) for rec in recommendations
            )
        }
        logger.info(
            f"Generated recommendations for {len(recommendations)} papers"
        )
    else:
        logger.info("Skipping recommendation generation")
    
    logger.info("Pipeline orchestration completed")
    return results


@flow(name="daily-pipeline")
def daily_pipeline(
    date: Optional[datetime] = None,
    auto_train: bool = False
) -> dict:
    """
    Daily pipeline that fetches new papers and generates recommendations.
    
    Args:
        date: Date to process (defaults to today)
        auto_train: Whether to automatically train models if annotations exist
    
    Returns:
        Dictionary with flow results
    """
    logger = get_run_logger()
    
    if date is None:
        date = datetime.now()
    
    # Check if we should train models
    annotations = load_annotations()
    should_train = auto_train and len(annotations) > 0
    
    return arxiv_pipeline_orchestration(
        date=date,
        fetch_data=True,
        train_models_flag=should_train,
        generate_recommendations_flag=True,
        check_annotations=True
    )


@flow(name="training-pipeline")
def training_pipeline(date: Optional[datetime] = None) -> dict:
    """
    Pipeline focused on model training.
    
    Args:
        date: Date to load papers from (defaults to today)
    
    Returns:
        Dictionary with training results
    """
    logger = get_run_logger()
    
    if date is None:
        date = datetime.now()
    
    logger.info("Starting training pipeline...")
    
    model_paths = train_models(date=date)
    
    return {
        "date": date.isoformat(),
        "model_training": model_paths
    }

