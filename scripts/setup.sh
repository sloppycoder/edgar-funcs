#!/bin/bash

PPOJECT_ID=edgar-ai
RESULT_TOPIC=edgarai-extraction-result
DLQ_TOPIC=${RESULT_TOPIC}-dlq
RESULT_TABLE=edgar2.extraction_result

bq rm -f -t ${PPOJECT_ID}:${RESULT_TABLE}
bq mk --table --schema extraction_result_schema.json ${PPOJECT_ID}:${RESULT_TABLE}

gcloud pubsub topics list --filter="name:$RESULT_TOPIC" --format="value(name)" | grep -q "$RESULT_TOPIC$"
if [ $? -ne 0 ]; then
  # Create the topic if it does not exist
  gcloud pubsub topics create "$RESULT_TOPIC"
  echo "Topic '$RESULT_TOPIC' created."
else
  echo "Topic '$RESULT_TOPIC' already exists."
fi

gcloud pubsub topics list --filter="name:$DLQ_TOPIC" --format="value(name)" | grep -q "$DLQ_TOPIC$"
if [ $? -ne 0 ]; then
  # Create the topic if it does not exist
  gcloud pubsub topics create "$DLQ_TOPIC"
  echo "Topic '$DLQ_TOPIC' created."
else
  echo "Topic '$DLQ_TOPIC' already exists."
fi

gcloud pubsub subscriptions delete extraction-result-sub
gcloud pubsub subscriptions create extraction-result-sub \
    --topic $RESULT_TOPIC \
    --bigquery-table ${PPOJECT_ID}:$RESULT_TABLE \
    --use-table-schema --drop-unknown-fields \
    --ack-deadline 60 \
    --dead-letter-topic $DLQ_TOPIC \
    --max-delivery-attempts 5

gcloud pubsub subscriptions delete extraction-result-dlq-sub
gcloud pubsub subscriptions create extraction-result-dlq-sub \
  --topic $DLQ_TOPIC

gcloud pubsub subscriptions list
