#!/usr/bin/sh

export LD_LIBRARY_PATH=/opt/cloudera/parcels/CDH/lib/hadoop/lib/native:/opt/cloudera/parcels/GPLEXTRAS/lib/hadoop/lib/native:/opt/cloudera/parcels/CDH/lib64:/opt/jdk/jre/lib/amd64/server/:$LD_LIBRARY_PATH

./reducer.py <&0

