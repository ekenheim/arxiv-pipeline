# Arxiv Prefect Pipeline

A Prefect-based backend pipeline for processing arXiv papers, including data fetching, annotation management, model training, and recommendation generation.

## Features

- **Data Fetching**: Daily fetching of new arXiv papers via the arXiv API
- **Data Processing**: Parse papers into sentences and structure data
- **Annotation System**: Manage sentence-level and abstract-level annotations
- **Model Training**: Train sentence transformer models with custom classifiers
- **Recommendations**: Generate recommendations for new papers based on trained models
- **MinIO Storage**: All data stored in MinIO buckets

## Project Structure

```
arxiv-pipeline/
├── src/
│   └── arxiv_pipeline/
│       ├── flows/          # Prefect flows
│       ├── tasks/          # Prefect tasks
│       ├── models/         # ML model components
│       ├── utils/          # Utilities and helpers
│       └── data/           # Data schemas
├── configs/
│   └── config.yml          # Configuration file
└── requirements.txt
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your MinIO credentials
```

3. Configure settings in `configs/config.yml`

4. Set up Prefect:
```bash
prefect server start  # Or use Prefect Cloud
```

## Usage

### Running Flows

```python
from arxiv_pipeline.flows.data_flow import fetch_and_process_papers

# Run data fetching flow
fetch_and_process_papers()
```

### Deploying Flows

```bash
prefect deployment build src/arxiv_pipeline/flows/data_flow.py:fetch_and_process_papers -n arxiv-data
prefect deployment apply fetch_and_process_papers-deployment.yaml
```

## Configuration

Edit `configs/config.yml` to:
- Define custom labels
- Configure model parameters
- Set storage paths
- Adjust data processing settings

## License

MIT

