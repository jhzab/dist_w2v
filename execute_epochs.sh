#!/bin/sh

for i in $(seq 5)
do
	echo "--> Running epoch: $i"
	$HADOOP_PREFIX/bin/hadoop jar /opt/cloudera/parcels/CDH-5.13.0-1.cdh5.13.0.p0.29/lib/hadoop-mapreduce/hadoop-streaming.jar -Dmapreduce.reduce.memory.mb=29000 -Dstream.non.zero.exit.is.failure=true -input /user/zab/wikipedia_corpus.txt -output /user/zab/dist_w2v_test -mapper mapper.py -reducer startReducer.sh -numReduceTasks 10 -file wiki_dict_most_freq_300000 -file mapper.py -file reducer.py -file startReducer.sh
	hdfs dfs -rm -r -skipTrash /user/zab/dist_w2v_test
done
