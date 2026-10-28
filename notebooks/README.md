# Notebook tools

- `rag_ingestion/rag_web_ingestion_utility.ipynb`: crawl a URL, classify its domain and update the local RAG index.
- `faq_crawling/faq_resumable_crawler.ipynb`: crawl FAQ pages with resumable checkpoints.

Install `backend/requirements/dev.txt` and JupyterLab in your virtual environment, then run `jupyter lab notebooks` from the repository root. Configure `backend/.env` and the model service first.

Notebook outputs are cleared in this source distribution. Generated documents, FAQ datasets and crawler checkpoints are local artifacts excluded from Git. The FAQ crawler defaults to a 30-second crawl delay.
