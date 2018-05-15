import org.apache.hadoop.mapred.Partitioner;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.io.Text;

public class KeyPartitioner implements Partitioner<Text, Text> {
	public void configure(JobConf job) {}

	public int getPartition(Text key, Text value, int numReduceTasks) {
		String keyStr = key.toString();
		return Integer.parseInt(keyStr);
	}
}
