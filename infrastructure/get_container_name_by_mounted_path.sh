#!/bin/bash

#set -x
part_of_path=$1
if [[ $part_of_path == "" ]]
then
    echo "ERROR: No argument given"
    exit 1
fi

get_bark_containers() {
        docker ps |
        awk '                      
        /bark_ml_image/ {print $NF}
'
}                                           
for container_name in $(get_bark_containers)
do                                                            
        match=$(docker inspect $container_name | ag $part_of_path)
        if [[ $match != "" ]]
        then                        
                echo $container_name
                echo Match:
                echo $match
        fi
done
