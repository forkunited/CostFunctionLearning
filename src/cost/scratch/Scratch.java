package cost.scratch;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ark.data.DataTools;
import ark.data.feature.Feature;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.util.FileUtil;
import ark.util.OutputWriter;
import cost.data.annotation.CostDatumTools;
import cost.data.annotation.TestDatum;
import cost.data.annotation.TestLabel;
import cost.model.factoredcost.FactoredCost;
import cost.util.CostProperties;

/**
 * Scratch is a scratch space in which to perform miscellaneous
 * throw-away tasks. Feel free to delete the code here and replace
 * it with other snippets you'd like to try.
 * 
 * @author Bill McDowell
 */
public class Scratch {
	public static void main(String[] args) {
		String dataSetName = args[0];
		CostProperties properties = new CostProperties();
		OutputWriter output = new OutputWriter();
		
		DataTools dataTools = new DataTools(output);
		dataTools.addToParameterEnvironment("DATA_SET", dataSetName);
		CostDatumTools<TestDatum, TestLabel> datumTools = new TestDatum.Tools(dataTools);
		String dataSetPath = (new File(properties.getFakeDataDirPath(), dataSetName)).getAbsolutePath();
		

		Feature<TestDatum, TestLabel> feature = datumTools.makeFeatureInstance("Identity");
		feature.fromString("Identity(doubleExtractor=Identity)", datumTools);
		List<Feature<TestDatum, TestLabel>> features = new ArrayList<Feature<TestDatum, TestLabel>>();
		FeaturizedDataSet<TestDatum, TestLabel> data = new FeaturizedDataSet<TestDatum, TestLabel>(dataSetPath,  features, 1, datumTools, null);
		
		try {
			BufferedReader reader = FileUtil.getFileReader(dataSetPath);
			String line = null;
			int id = 0;
			while ((line = reader.readLine()) != null) {
				String[] lineParts = line.split("\t");
				double[] featureValues = new double[lineParts.length - 1];
				for (int i = 0; i < featureValues.length; i++)
					featureValues[i] = Double.valueOf(lineParts[i]);
				TestLabel label = TestLabel.valueOf(lineParts[lineParts.length - 1]);
				data.add(new TestDatum(id, featureValues, label));
				id++;
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		FactoredCost<TestDatum, TestLabel> cost = datumTools.makeFactoredCostInstance("LabelPair");
		SupervisedModel<TestDatum, TestLabel> model = datumTools.makeModelInstance("CLSVMN");
		try {
			cost.deserialize(new BufferedReader(new StringReader("LabelPair(c=1)")), true, datumTools);
			model.deserialize(new BufferedReader(new StringReader("CLSVMN()\n{\nvalidLabels=L_0,L_1,L_2,L_3,L_4,L_5,L_6\n}")), true, false, datumTools, "model");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
			
		cost.init(model, data);
		
		for (TestDatum datum : data) {
			for (TestLabel label : TestLabel.values()) {
				Map<Integer, Double> costs = cost.computeVector(datum, label);
				Map<Integer, String> nonZeroCostNames = cost.getSpecificShortNamesForIndices(costs.keySet());
				System.out.print(datum.getLabel() + " " + label + ":\t");
				for (String name : nonZeroCostNames.values()) {
					System.out.print(name + " ");
				}
				System.out.println();
			}
		}
	}
}
