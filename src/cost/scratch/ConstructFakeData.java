package cost.scratch;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import ark.data.DataTools;
import ark.data.feature.Feature;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.util.FileUtil;
import ark.util.OutputWriter;
import cost.data.annotation.TestDatum;
import cost.data.annotation.TestLabel;

/**
 * ConstructFakeData takes arguments:
 * 
 * [examplesPerFeatureCombination] - Number of examples per combination of feature values
 * [numFeatures] - Number of features
 * [randomSeed] - Seed for random numbers
 * [modelPath] - Path to serialized model that gives a distribution by which to generate the data
 * [outputPath] - Path to output file where to write the data
 * 
 * And generates a file at [outputPath] containing synthetic featurized data
 * with [numFeatures] per example and labels generated according to the
 * distribution specified by the model at [modelPath], sampled using a random number
 * generator with seed [randomSeed].  The data set will
 * contain examples for every possible combination of feature values {1, -1},
 * and there will be [examplesPerFeatureCombination] examples of each combination.
 * 
 * Example models from which to generate the data are given in the 
 * 'syntheticDataModels' directory.
 * 
 * @author Bill McDowell
 *
 */
public class ConstructFakeData {
	public static void main(String[] args) {
		int examplesPerFeatureCombination = Integer.valueOf(args[0]);
		int numFeatures = Integer.valueOf(args[1]);
		Random random = new Random(Integer.valueOf(args[2]));
		String modelPath = args[3];
		String outputPath = args[4];
		
		OutputWriter output = new OutputWriter();
		DataTools dataTools = new DataTools(output);
		TestDatum.Tools datumTools = new TestDatum.Tools(dataTools);
		SupervisedModel<TestDatum, TestLabel> model = datumTools.makeModelInstance("SVMCLN");
		try {
			BufferedReader reader = FileUtil.getFileReader(modelPath);
			if (!model.deserialize(reader, true, true, datumTools, null)) {
				output.debugWriteln("Error: Failed to deserialize model.");
				return;
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
		
		double[][] featureCombinations = constructFeatureCombinations(numFeatures);
		FeaturizedDataSet<TestDatum, TestLabel> data = new FeaturizedDataSet<TestDatum, TestLabel>("Fake", datumTools, null);
		int id = 0;
		for (int i = 0; i < featureCombinations.length; i++) {
			for (int j = 0; j < examplesPerFeatureCombination; j++) {
				data.add(new TestDatum(id, featureCombinations[i], null));
				id++;
			}
		}
		
		Feature<TestDatum, TestLabel> feature = datumTools.makeFeatureInstance("Identity");
		if (!feature.fromString("Identity(doubleExtractor=Identity)", datumTools)) {
			output.debugWriteln("Error: Failed to construct feature.");
			return;
		}
		
		if (!feature.init(data)) {
			output.debugWriteln("Error: Failed to initialize feature.");
			return;
		}
		
		data.addFeature(feature);
		
		Map<TestDatum, Map<TestLabel, Double>> posterior = model.posterior(data);
		Map<TestLabel, Integer> labelCounts = new HashMap<TestLabel, Integer>();
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath));

			for (Entry<TestDatum, Map<TestLabel, Double>> entry : posterior.entrySet()) {
				double[] features = entry.getKey().getFeatureValues();
				
				for (int i = 0; i < features.length; i++) {
					output.debugWrite(i + ": " + features[i] + "\t");
					writer.write(features[i] + "\t");
				}
				
				TestLabel choice = sample(entry.getValue(), random);
				if (!labelCounts.containsKey(choice))
					labelCounts.put(choice, 0);
				labelCounts.put(choice, labelCounts.get(choice) + 1);
				
				writer.write(choice.toString() + "\n");
				
				output.debugWrite("[Choice: " + choice + "]\t");
				
				for (Entry<TestLabel, Double> p : entry.getValue().entrySet()) {
					output.debugWrite(p.getKey() + ": " + p.getValue() + "\t");
				}
				
				output.debugWriteln("");
			}
			
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		for (Entry<TestLabel, Integer> entry : labelCounts.entrySet())
			output.debugWriteln(entry.getKey() + "\t" + entry.getValue());
	}
	
	public static TestLabel sample(Map<TestLabel, Double> distribution, Random random) {
		TestLabel sample = null;
		List<Entry<TestLabel, Double>> entries = new ArrayList<Entry<TestLabel, Double>>(distribution.size());
		double p = random.nextDouble();
		entries.addAll(distribution.entrySet());
		double total = 0;
		int i = 0;
		while (p > total) {
			total += entries.get(i).getValue();
			sample = entries.get(i).getKey();
			i++;
		}
		
		return sample;
	}
	
	public static double[][] constructFeatureCombinations(int numFeatures) {
		double[][] combinations = new double[(int)Math.pow(2, numFeatures)][];
		
		for (int i = 0; i < combinations.length; i++) {
			combinations[i] = new double[numFeatures];
			for (int j = 0; j < numFeatures; j++) {
				combinations[i][j] = (i >> j) & 1;
				if (combinations[i][j] == 0)
					combinations[i][j] = -1;
			}
		}
		
		return combinations;
	}
	
}
