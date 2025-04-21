#!/bin/bash

PPOJECT_ID=edgar-ai
RESULT_TOPIC=edgarai-extraction-result
DLQ_TOPIC=${RESULT_TOPIC}-dlq
RESULT_TABLE=edgar.extraction_result

bq rm -f -t ${PPOJECT_ID}:${RESULT_TABLE}
bq mk --table --schema extraction_result_schema.json ${PPOJECT_ID}:${RESULT_TABLE}

gcloud pubsub topics create "$RESULT_TOPIC"
gcloud pubsub subscriptions create $RESULT_TOPIC-sub \
  --topic $RESULT_TOPIC

gcloud pubsub topics create "$DLQ_TOPIC"
gcloud pubsub subscriptions create $DLQ_TOPIC-sub \
  --topic $DLQ_TOPIC

gcloud pubsub subscriptions create extraction-result-sub \
    --topic $RESULT_TOPIC \
    --bigquery-table ${PPOJECT_ID}:$RESULT_TABLE \
    --use-table-schema --drop-unknown-fields \
    --ack-deadline 60 \
    --dead-letter-topic $DLQ_TOPIC \
    --max-delivery-attempts 5

gcloud pubsub subscriptions list
