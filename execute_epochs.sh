#!/bin/sh

REDUCERS=10
#INPUT="/user/zab/text_sc1/2007_real/*039*.json.gz"
INPUT="/user/zab/text_sc1/2007_real/*.json.gz"
COMPRESSION="-Dmapreduce.map.output.compress=true -Dmapreduce.output.fileoutputformat.compress=false -Dmapreduce.output.fileoutputformat.compress.codec=org.apache.hadoop.io.compress.SnappyCodec"
VCORES=10
EPOCHS=1
JOB_NAME="0.75_subsample_2007"
echo "Job name: $JOB_NAME"

for i in $(seq $EPOCHS)
do
	echo "--> Running epoch: $i"
	$HADOOP_PREFIX/bin/hadoop jar /opt/cloudera/parcels/CDH-5.13.0-1.cdh5.13.0.p0.29/lib/hadoop-mapreduce/hadoop-streaming.jar -D mapred.job.name="$JOB_NAME" -Dmapreduce.task.timeout=660000 -Dmapred.job.queue.name=root.users.vldb-zab $COMPRESSION -Dmapreduce.reduce.memory.mb=32000 -Dmapreduce.reduce.cpu.vcores="$VCORES" -Dstream.non.zero.exit.is.failure=true -input "$INPUT" -output "/user/zab/dist_w2v_$JOB_NAME" -mapper mapper.py -reducer startReducer.sh -numReduceTasks "$REDUCERS" -file wiki_dict_most_freq_300000 -file 2007_dict_most_freq_300000 -file mapper.py -file reducer.py -file startReducer.sh -file KeyPartitioner.class -partitioner KeyPartitioner
	hdfs dfs -rm -r -skipTrash "/user/zab/dist_w2v_$JOB_NAME"
done
