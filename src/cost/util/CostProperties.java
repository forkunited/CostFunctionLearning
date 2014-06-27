package cost.util;

import ark.util.Properties;

/**
 * CostProperties loads in a cost.properties configuration
 * file.
 * 
 * @author Bill McDowell
 * 
 */
public class CostProperties extends Properties {
	private String fakeDataDirPath;
	private String experimentInputDirPath;
	private String experimentOutputDirPath;
	
	public CostProperties() {
		super(new String[] { "cost.properties" } );
		
		this.fakeDataDirPath = loadProperty("fakeDataDirPath");
		this.experimentInputDirPath = loadProperty("experimentInputDirPath");
		this.experimentOutputDirPath = loadProperty("experimentOutputDirPath");
	}
	
	public String getFakeDataDirPath() {
		return this.fakeDataDirPath;
	}

	public String getExperimentInputDirPath() {
		return this.experimentInputDirPath;
	}
	
	public String getExperimentOutputDirPath() {
		return this.experimentOutputDirPath;
	}

}
