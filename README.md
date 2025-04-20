# Process SEC filings on EDGAR site

This project contains the Google Cloud Functions that proesses the filing html files from EDGAR site.

For 485BPOS filings:
* Split a filing into mutliple text chunks and generate embeddings
* Extract information from a filing using a LLM with RAG (Retrieval Augmented Generation)

## architecture
* Requests are published into Pub/Sub topic, a push subscription triggers the Cloud Run HTTP Endpooint, the result is written into Firestore collection.
*
