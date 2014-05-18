package cost.data.annotation;

import ark.data.DataTools;
import ark.data.annotation.Datum;

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
	
	public static class Tools extends Datum.Tools<TestDatum,TestLabel> {
		public Tools(DataTools dataTools) {
			super(dataTools);
			
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
