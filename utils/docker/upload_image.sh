#!/bin/bash
rsync ./bark_ml.img 8gpu:/mnt/glusterdata/home/$1/bark_ml.img -a -v -z -P