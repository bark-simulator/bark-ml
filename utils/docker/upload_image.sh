#!/bin/bash
rsync ./bark_ml.img 8gpu:/mnt/glusterdata/home/$1/images/bark_ml.img -a -v -z -P