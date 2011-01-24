package weka.classifiers.trees.nig;

import weka.classifiers.trees.j48.C45ModelSelection;
import weka.core.Instances;

public class NIGModelSelection extends C45ModelSelection {

	/**
	 * for serialization
	 */
	private static final long serialVersionUID = 2947119352591363685L;
	
	private double [] userWeights;

	public double[] getUserWeights() {
		return userWeights;
	}

	public void setUserWeights(double[] userWeights) {
		this.userWeights = userWeights;
	}

	public NIGModelSelection(int minNoObj, Instances allData, double[] userWeights) {
		super(minNoObj, allData);
		this.setUserWeights(userWeights);
	}

}
