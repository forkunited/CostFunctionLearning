package cost.data.annotation;

import java.util.HashMap;
import java.util.Map;

import cost.model.SupervisedModelSVMCLN;
import cost.model.factoredcost.FactoredCost;
import cost.model.factoredcost.FactoredCostActualLabelCount;
import cost.model.factoredcost.FactoredCostConstant;
import cost.model.factoredcost.FactoredCostFeature;
import cost.model.factoredcost.FactoredCostLabel;
import cost.model.factoredcost.FactoredCostLabelPair;
import cost.model.factoredcost.FactoredCostLabelPairUnordered;

import ark.data.DataTools;
import ark.data.annotation.Datum;

/**
 * CostDatumTools contains cost function learning tools for manipulating
 * data sets.  An instance of this class can be used as a factory to 
 * instantiate cost function learning models for a given type of datum.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public abstract class CostDatumTools<D extends Datum<L>,L> extends Datum.Tools<D, L> {	
	private Map<String, FactoredCost<D, L>> genericFactoredCosts;
	
	public CostDatumTools(DataTools dataTools) {
		super(dataTools);
		
		this.genericFactoredCosts = new HashMap<String, FactoredCost<D, L>>();
		
		addLabelMapping(new LabelMapping<L>() {
			public String toString() {
				return "Identity";
			}
			
			@Override
			public L map(L label) {
				return label;
			}
		});
		
		
		addGenericModel(new SupervisedModelSVMCLN<D, L>());		
		
		addGenericFactoredCost(new FactoredCostConstant<D, L>());
		addGenericFactoredCost(new FactoredCostLabel<D, L>());
		addGenericFactoredCost(new FactoredCostLabelPair<D, L>());
		addGenericFactoredCost(new FactoredCostLabelPairUnordered<D, L>());
		addGenericFactoredCost(new FactoredCostFeature<D, L>());
		addGenericFactoredCost(new FactoredCostActualLabelCount<D, L>());
	}
	
	public FactoredCost<D, L> makeFactoredCostInstance(String genericFactoredCostName) {
		return this.genericFactoredCosts.get(genericFactoredCostName).clone(this, this.dataTools.getParameterEnvironment());
	}

	public boolean addGenericFactoredCost(FactoredCost<D, L> factoredCost) {
		this.genericFactoredCosts.put(factoredCost.getGenericName(), factoredCost);
		return true;
	}
}