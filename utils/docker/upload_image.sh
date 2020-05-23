#!/bin/bash
rsync ./barkml_diadem.img bernhard@8gpu:/mnt/glusterdata/home/$1/images/barkml_diadem.img -a -v -z -P