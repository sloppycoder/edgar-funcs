#!/bin/bash

PPOJECT_ID=edgar-ai
REQUEST_TOPIC=edgarai-request
DLQ_TOPIC=${REQUEST_TOPIC}-dlq

gcloud pubsub topics create "$REQUEST_TOPIC"
gcloud pubsub subscriptions create $REQUEST_TOPIC-sub --topic $REQUEST_TOPIC

gcloud pubsub topics create "$DLQ_TOPIC"
gcloud pubsub subscriptions create $DLQ_TOPIC-sub --topic $DLQ_TOPIC


gcloud pubsub subscriptions create edgarai-request-trigger \
  --topic=$REQUEST_TOPIC \
  --push-endpoint=https://edgar-processor-1049830028293.us-central1.run.app/process \
  --push-auth-service-account=edgar-dev@edgar-ai.iam.gserviceaccount.com \
  --ack-deadline=600 \
  --message-retention-duration=7d \
  --dead-letter-topic=$DLQ_TOPIC \
  --max-delivery-attempts=5 \
  --retry-policy-minimum-backoff=60s \
  --retry-policy-maximum-backoff=600s \
  --expiration-period=7d

gcloud pubsub subscriptions list
