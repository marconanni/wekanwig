package weka.classifiers.trees.nig;

import weka.classifiers.trees.j48.C45Split;

/**
 * 
 * @author Marco Nanni
 * @version 1.0
 * this class extends the C45Split, more specifically it returns an infoGain that
 * is calculated in the same way c45 does, but multiplied by a weight specified by the user
 */
public class NIGSplit extends C45Split {
	
	/**for serialization**/
	private static final long serialVersionUID = 682734619045338718L;
	
	/** the user weight, must be in the rang 0..1**/
	protected double userWeight;
	
	public double getUserWeight() {
		return userWeight;
	}

	public void setUserWeight(double userWeight) {
		this.userWeight = userWeight;
	}
	
	/**
	 * initializes the split model
	 * @param userWeight the user weight for the current split, must be in the range 0..1
	 * @throws Exception if userWeight is not in the range 0..1
	 */
	public NIGSplit(int attIndex, int minNoObj, double sumOfWeights, double userWeight) throws Exception {		
		super(attIndex, minNoObj, sumOfWeights);
		if (!NIGSplit.checkUserWeight(userWeight))
			throw new Exception("The userWeight must be in the range 0..1, "+userWeight+"is not valid");
		this.setUserWeight(userWeight);
	}

	/**
	 * initializes the split model putting the user weight value at 1
	 */
	public NIGSplit(int attIndex, int minNoObj, double sumOfWeights) {		
		super(attIndex, minNoObj, sumOfWeights);
		this.setUserWeight(1);
	}

	
	/**
	 * Checks if the weight is in the range 0..1
	 * @param userWeight the user weight to check
	 * @return true if the weight is in the range 0..1, false otherwise
	 * 
	 */
	public static boolean checkUserWeight(double userWeight){
		return (userWeight>=0 && userWeight <=1);
	}

	
	@Override
	/**
	 * returns the Nanni info Gain
	 */
	public double infoGain() {
		
		return (this.getUserWeight()*super.infoGain());
	}
	
	 

}
