## script to setup bigquery table and pubsub topics

```shell
# gcloud auth activate-service-account --key-file edgar-ai-dev.json
# gcloud config set account edgar-dev@edgar-ai.iam.gserviceaccount.com


./req_topic.sh


```

Then go into console and check dead letter roles, grant if needed.

Also, the (default) database must be setup in [Google Cloud Firestore console](https://console.cloud.google.com/firestore/)