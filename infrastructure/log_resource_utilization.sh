#!/bin/bash

RESOURCE_LOG_DIR="resource_logs"
INTERVAL=1

mkdir -p $RESOURCE_LOG_DIR
while true
do
    top -n 1 >> $RESOURCE_LOG_DIR/top_log
    nvidia-smi >> $RESOURCE_LOG_DIR/nvidia-smi_log
    sleep $INTERVAL
done
