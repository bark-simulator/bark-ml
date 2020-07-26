#!/bin/bash

set -x 

while [[ "$#" -gt 1 ]]
do
    case $1 in 
        -p|--port_host) port_host="$2"; shift;;
        -l|--log_dir_docker) log_dir_docker="$2"; shift;;
        *) echo Unknown parameter "$1"; exit 1;;
    esac
    shift
done
log_dir_to_mount="$1"
[[ -v log_dir_docker ]] || log_dir_docker="/logs"

if [[ ! -v port_host ]]
then 
    echo ERROR: No host port specified!
    exit 1
fi

if [[ ! -v log_dir_to_mount ]]
then
    echo ERROR: No log directory specified!
    exit 1
fi


docker run -it -p "$port_host":6006 -v $(pwd)/"$log_dir_to_mount":/logs tensorflow/tensorflow tensorboard --bind_all --logdir $log_dir_docker
