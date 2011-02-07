package weka.classifiers.trees.nig;

import weka.classifiers.trees.j48.*;

public class BinNIGSplit extends BinC45Split {
	/**
	 * for serialization
	 */
	private static final long serialVersionUID = -4553804111549603386L;
	
	private double userWeight;

	/**
	 * @return the userWeight
	 */
	public double getUserWeight() {
		return userWeight;
	}

	/**
	 * @param userWeight the userWeight to set
	 */
	public void setUserWeight(double userWeight) {
		this.userWeight = userWeight;
	}

	public BinNIGSplit(int attIndex, int minNoObj, double sumOfWeights, double userWeight) {
		
		super(attIndex, minNoObj, sumOfWeights);
		this.userWeight = userWeight;
	}

	/* (non-Javadoc)
	 * @see weka.classifiers.trees.j48.BinC45Split#infoGain()
	 */
	@Override
	public double infoGain() {
		// TODO Auto-generated method stub
		return userWeight*super.infoGain();
	}

	

}
