"""Tasks for processing paper data."""

import re
from typing import List
from prefect import task, get_run_logger
from arxiv_pipeline.data.schemas import Paper, Sentence, ProcessedPaper
from arxiv_pipeline.utils.config import get_config


@task
def split_abstract_into_sentences(abstract: str) -> List[str]:
    """
    Split abstract into sentences.
    
    Args:
        abstract: Abstract text
    
    Returns:
        List of sentence strings
    """
    config = get_config()
    logger = get_run_logger()
    
    # Start with the abstract
    text = abstract.strip()
    
    # Use configured sentence separators
    separators = config.data.sentence_separators
    min_length = config.data.min_sentence_length
    
    # Split by separators
    sentences = [text]
    for sep in separators:
        new_sentences = []
        for sent in sentences:
            parts = sent.split(sep)
            # Keep the separator with the previous part except for the last one
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    part = part + sep
                if part.strip():
                    new_sentences.append(part.strip())
        sentences = new_sentences
    
    # Clean up sentences
    cleaned_sentences = []
    for sent in sentences:
        sent = sent.strip()
        # Remove extra whitespace
        sent = re.sub(r'\s+', ' ', sent)
        # Filter by minimum length
        if len(sent) >= min_length:
            cleaned_sentences.append(sent)
    
    logger.debug(f"Split abstract into {len(cleaned_sentences)} sentences")
    return cleaned_sentences


@task
def process_paper(paper: Paper) -> ProcessedPaper:
    """
    Process a paper by splitting its abstract into sentences.
    
    Args:
        paper: Paper object
    
    Returns:
        ProcessedPaper with sentences
    """
    logger = get_run_logger()
    
    # Split abstract into sentences
    sentence_texts = split_abstract_into_sentences(paper.abstract)
    
    # Create Sentence objects
    sentences = []
    current_pos = 0
    
    for idx, text in enumerate(sentence_texts):
        start_char = paper.abstract.find(text, current_pos)
        end_char = start_char + len(text) if start_char >= 0 else None
        
        sentence = Sentence(
            text=text,
            sentence_index=idx,
            paper_id=paper.arxiv_id,
            start_char=start_char if start_char >= 0 else None,
            end_char=end_char
        )
        sentences.append(sentence)
        
        if start_char >= 0:
            current_pos = start_char + len(text)
    
    processed_paper = ProcessedPaper(
        paper=paper,
        sentences=sentences
    )
    
    logger.info(f"Processed paper {paper.arxiv_id}: {len(sentences)} sentences")
    return processed_paper


@task
def process_papers(papers: List[Paper]) -> List[ProcessedPaper]:
    """
    Process multiple papers.
    
    Args:
        papers: List of Paper objects
    
    Returns:
        List of ProcessedPaper objects
    """
    logger = get_run_logger()
    processed_papers = []
    
    for paper in papers:
        try:
            processed = process_paper(paper)
            processed_papers.append(processed)
        except Exception as e:
            logger.warning(f"Error processing paper {paper.arxiv_id}: {e}")
            continue
    
    logger.info(f"Processed {len(processed_papers)} out of {len(papers)} papers")
    return processed_papers

