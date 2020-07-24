#!/bin/bash

# To log the memory consumption of the bark-ml processes run
# while true\ndo\n\tnvidia-smi >> nvidia_smi_log\n\tsleep 1\ndone   

NVIDIA_SMI_LOG_FILE_NAME="nvidia_smi_log"

get_pids() {
    awk '
    / 0 .*bark_ml/ {print $3}
    / 1 .*bark_ml/ {exit}
    ' $NVIDIA_SMI_LOG_FILE_NAME
}


pid=$(get_pids | head -n1)

echo $pid

check_input_pid() {
    pid=$1
    if [[ $pid == "" ]]
    then
        echo "ERROR: PID needs to be set"
        exit 1
    fi
}

get_memory_consumption_for_pid() {
    pid=$1
    check_input_pid $pid
    awk '
    / 0 .*'$pid'.*bark_ml/ {
    sub(/MiB \|/,"")
    sub(/\|.*python3   /,"")
    print $0
    }
    ' $NVIDIA_SMI_LOG_FILE_NAME
}

get_memory_consumption_for_pid_enumerated() {
    pid=$1
    check_input_pid $pid
    get_memory_consumption_for_pid $pid |
        awk '
            {print NR", "$0}
        '
}

plot_memory_consumption_pid() {
    pid=$1
    check_input_pid $pid
    get_memory_consumption_for_pid_enumerated $pid |
        gnuplot -p -e "plot '<cat'"
}

for pid in $(get_pids)
do
    get_memory_consumption_for_pid $pid | wc
    plot_memory_consumption_pid $pid
done

