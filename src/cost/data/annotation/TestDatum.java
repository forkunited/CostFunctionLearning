package cost.data.annotation;

import ark.data.DataTools;
import ark.data.annotation.Datum;

/**
 * TestDatum represents a datum with feature values from a synthetic
 * data set.
 * 
 * @author Bill McDowell
 *
 */
public class TestDatum extends Datum<TestLabel>  {
	private double[] featureValues;
	
	public TestDatum(int id, double[] featureValues, TestLabel label) {
		this.id = id;
		this.featureValues = featureValues;
		this.label = label;
	}
	
	public double[] getFeatureValues() {
		return this.featureValues;
	}
	
	/**
	 * Tools for manipulating synthetic data.
	 * 
	 * @author Bill McDowell
	 *
	 */
	public static class Tools extends CostDatumTools<TestDatum,TestLabel> {
		public Tools(DataTools dataTools) {
			super(dataTools);
			
			/**
			 * Allows features in ARKWater to extract values of synthetic 
			 * feature values from a given datum.
			 */
			this.addDoubleExtractor(new DoubleExtractor<TestDatum, TestLabel>() {
				@Override
				public String toString() {
					return "Identity";
				}
				
				@Override
				public double[] extract(TestDatum datum) {
					return datum.getFeatureValues();
				}
			});
			
			/**
			 * Maps L_0 and L_1 to the same label and L_2 and
			 * L_3 to the same label
			 */
			this.addLabelMapping(new LabelMapping<TestLabel>() {
				@Override
				public String toString() {
					return "L_0L_1-L_2L_3";
				}
				
				@Override
				public TestLabel map(TestLabel label) {
					if (label.equals(TestLabel.L_1))
						return TestLabel.L_0;
					else if (label.equals(TestLabel.L_3))
						return TestLabel.L_2;
					else
						return label;
				}
			});
			
			/**
			 * Maps L_2 and L_3 to the same label
			 */
			this.addLabelMapping(new LabelMapping<TestLabel>() {
				@Override
				public String toString() {
					return "L_2L_3";
				}
				
				@Override
				public TestLabel map(TestLabel label) {
					if (label.equals(TestLabel.L_3))
						return TestLabel.L_2;
					else
						return label;
				}
			});
		}

		@Override
		public TestLabel labelFromString(String str) {
			return TestLabel.valueOf(str);
		}
	}
}
