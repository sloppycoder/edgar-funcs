# Process SEC filings on EDGAR site

This project contains Google Cloud Functions that process filing HTML/TXT files from the EDGAR site.

For 485BPOS filings:
* Split a filing into multiple text chunks and generate embeddings.
* Extract information from a filing using an LLM with RAG (Retrieval Augmented Generation).

## Architecture
The processer is deployed as Google Cloud Run service.

Request for processing a file is published into a Pub/Sub topic, then pushed to Cloud Run HTTP endpoint by a push subscription. The processing result is published into a separate Pub/Sub topic, which then pushed into a BigQuery table.

A CLI utility is provided that writes requests into the Request topic.

```
+-------------------+       +-------------------+       +-------------------+
|                   |       |                   |       |                   |
|   Request Topic   |------>|   Processor       |------>|   Result Topic    |
|   (Pub/Sub)       | HTTP  |   (Cloud Run)     |publish|   (Pub/Sub)       |
|                   | push  |                   |result |                   |
+-------------------+       +-------------------+       +-------------------+
        ^  publish                                              | subscription
        |  requests                                             V write to big query
+-------------------+                                   +-------------------+
|                   |         query stats               |                   |
|   CLI Utility     |<----------------------------------|   Result Table    |
|   (cli.py)        |                                   |   (Big Query)     |
|                   |                                   |                   |
+-------------------+                                   +-------------------+

```

## Deploy

### Setup Pub/Sub Topics and Subscriptions
Run the following scripts to set up Pub/Sub topics, subscriptions, and the BigQuery table:

1. **Request Topic Setup**:
   ```bash
   ./scripts/req_topic.sh
   ```
   This script creates the request topic, its dead-letter queue (DLQ), and a push subscription to trigger the Cloud Run endpoint.

2. **Result Topic Setup**:
   ```bash
   ./scripts/result_topic.sh
   ```
   This script creates the result topic, its DLQ, and a subscription that writes results to a Big Query table.

### Deploy with Cloud Run
A ```cloudbuild.yaml``` is provided in the repo. Build and deploy can be done using Cloud Build.

## Usage

creat a ```.env``` file
```env
PUBSUB_TOPIC=edgarai-request
RESULT_TABLE=<project>.<dataset>.extraction_result
GOOGLE_APPLICATION_CREDENTIALS=<servie account credetia>.json
CLI_EDGAR_PROCESSOR_URL=https://YOUR_CLOUD_RUN_URL/process
```

### CLI Usage
The CLI supports the following commands:
1. **Chunk Filings**:
   ```bash
   python cli.py chunk 10 --start 2024-01-01 --end 2024-12-31
   ```
   This processes filings from 10% of the companies within the specified date range.

2. **Process a Single Filing**:
   ```bash
   python cli.py chunk 0000000000-00-000000
   ```
   This processes a filing by its accession number.

3. **Generate Stats**:
   ```bash
   python cli.py stats 20240101000000-abc
   ```
   This generates statistics for a specific batch ID.

4. **Process Filings from a CSV**:
   ```bash
   python cli.py chunk filings.csv
   ```
   The CSV should contain columns: `cik`, `company_name`, and `accession_number`.

### Query Results in BigQuery
Run the following query to check processed filings:
```sql
SELECT batch_id, extraction_type, min(date_filed), max(date_filed), count(*)
FROM `<dataset>.extraction_result`
GROUP BY batch_id, extraction_type
order by 1 desc
LIMIT 1000;
```
