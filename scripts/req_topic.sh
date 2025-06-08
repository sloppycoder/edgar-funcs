#!/bin/bash

PPOJECT_ID=edgar-ai
REQUEST_TOPIC=edgarai-request
DLQ_TOPIC=${REQUEST_TOPIC}-dlq
SUB_EXP=60d

gcloud pubsub topics create "$REQUEST_TOPIC"
gcloud pubsub subscriptions create $REQUEST_TOPIC-sub --topic $REQUEST_TOPIC --expiration-period=$SUB_EXP

gcloud pubsub topics create "$DLQ_TOPIC"
gcloud pubsub subscriptions create $DLQ_TOPIC-sub --topic $DLQ_TOPIC --expiration-period=$SUB_EXP


gcloud pubsub subscriptions create edgarai-request-trigger \
  --topic=$REQUEST_TOPIC \
  --push-endpoint=https://edgar-processor-1049830028293.us-central1.run.app/process \
  --push-auth-service-account=edgar-dev@edgar-ai.iam.gserviceaccount.com \
  --ack-deadline=600 \
  --message-retention-duration=7d \
  --dead-letter-topic=$DLQ_TOPIC \
  --max-delivery-attempts=5 \
  --min-retry-delay=60s \
  --max-retry-delay=600s \
  --expiration-period=$SUB_EXP

gcloud pubsub subscriptions list
