package cost.scratch;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;

import cost.data.annotation.TestDatum;
import cost.data.annotation.TestLabel;
import cost.util.CostProperties;

import ark.data.DataTools;
import ark.data.annotation.DataSet;
import ark.data.annotation.Datum.Tools;
import ark.experiment.ExperimentKCV;
import ark.util.FileUtil;
import ark.util.OutputWriter;

public class ExperimentKCVTest {
	public static void main(String[] args) {
		String experimentName = "KCVTest/" + args[0];
		String dataSetName = args[1];
		String experimentOutputName = dataSetName + "/" + experimentName;

		CostProperties properties = new CostProperties();
		String experimentInputPath = new File(properties.getExperimentInputDirPath(), experimentName + ".experiment").getAbsolutePath();
		String experimentOutputPath = new File(properties.getExperimentOutputDirPath(), experimentOutputName).getAbsolutePath(); 
		
		OutputWriter output = new OutputWriter(
				new File(experimentOutputPath + ".debug.out"),
				new File(experimentOutputPath + ".results.out"),
				new File(experimentOutputPath + ".data.out"),
				new File(experimentOutputPath + ".model.out")
			);
		
		DataTools dataTools = new DataTools(output);
		dataTools.addToParameterEnvironment("DATA_SET", dataSetName);
		
		Tools<TestDatum, TestLabel> datumTools = new TestDatum.Tools(dataTools);
		
		String dataSetPath = (new File(properties.getFakeDataDirPath(), dataSetName)).getAbsolutePath();
		DataSet<TestDatum, TestLabel> data = new DataSet<TestDatum, TestLabel>(datumTools, null);
		
		try {
			BufferedReader reader = FileUtil.getFileReader(dataSetPath);
			String line = null;
			int id = 0;
			while ((line = reader.readLine()) != null) {
				String[] lineParts = line.split("\t");
				double[] features = new double[lineParts.length - 1];
				for (int i = 0; i < features.length; i++)
					features[i] = Double.valueOf(lineParts[i]);
				TestLabel label = TestLabel.valueOf(lineParts[lineParts.length - 1]);
				data.add(new TestDatum(id, features, label));
				id++;
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		ExperimentKCV<TestDatum, TestLabel> experiment = 
				new ExperimentKCV<TestDatum, TestLabel>(experimentOutputName, experimentInputPath, data);
	
		if (!experiment.run())
			output.debugWriteln("Error: Experiment run failed.");
	}
}
