package weka.classifiers.trees.nig;

import java.util.Enumeration;

import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.C45Split;
import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.NoSplit;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Class for selecting the (best) NIGSplit available for a given dataset
 * 
 * @author Marco Nanni
 * 
 * @version 1.0
 *
 */
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
	

	/**
	   * Initializes the split selection method with the given parameters.
	   *
	   * @param minNoObj minimum number of instances that have to occur in at least two
	   * subsets induced by split
	   * @param allData FULL training dataset (necessary for
	   * selection of split points).
	   * all user weights are initialized to 1
	   * @param userWeights the user Weights used for calculating the Nanni InfoGain.
	   */
	
	public NIGModelSelection(int minNoObj, Instances allData, double[] userWeights) {
		super(minNoObj, allData);
		this.setUserWeights(userWeights);
	}
	
	
	/**
	   * Initializes the split selection method with the given parameters.
	   *
	   * @param minNoObj minimum number of instances that have to occur in at least two
	   * subsets induced by split
	   * @param allData FULL training dataset (necessary for
	   * selection of split points).
	   * all user weights are initialized to 1
	   */
	public NIGModelSelection(int minNoObj, Instances allData) {
		super(minNoObj, allData);
		userWeights = new double [super.getM_allData().numAttributes()];
		for (int i = 0; i < userWeights.length; i++) {
			userWeights[i]=1;
		}
	}

	/**
	   * Selects NIG type best split for the given dataset.
	   */
	  public  ClassifierSplitModel selectModel(Instances data){

	    double minResult;
	    double currentResult;
	    NIGSplit [] currentModel;
	    NIGSplit bestModel = null;
	    NoSplit noSplitModel = null;
	    double averageInfoGain = 0;
	    int validModels = 0;
	    boolean multiVal = true;
	    Distribution checkDistribution;
	    Attribute attribute;
	    double sumOfWeights;
	    int i;
	    
	    try{

	      // Check if all Instances belong to one class or if not
	      // enough Instances to split.
	      checkDistribution = new Distribution(data);
	      noSplitModel = new NoSplit(checkDistribution);
	      if (Utils.sm(checkDistribution.total(),2*super.getM_minNoObj()) ||
		  Utils.eq(checkDistribution.total(),
			   checkDistribution.perClass(checkDistribution.maxClass())))
		return noSplitModel;

	      // Check if all attributes are nominal and have a 
	      // lot of values.
	      if (super.getM_allData() != null) {
		Enumeration enu = data.enumerateAttributes();
		while (enu.hasMoreElements()) {
		  attribute = (Attribute) enu.nextElement();
		  if ((attribute.isNumeric()) ||
		      (Utils.sm((double)attribute.numValues(),
				(0.3*(double)super.getM_allData().numInstances())))){
		    multiVal = false;
		    break;
		  }
		}
	      } 

	      currentModel = new NIGSplit[data.numAttributes()];
	      sumOfWeights = data.sumOfWeights();

	      // For each attribute.
	      for (i = 0; i < data.numAttributes(); i++){
		
		// Apart from class attribute.
		if (i != (data).classIndex()){
		  
		  // Get models for current attribute.
		  currentModel[i] = new NIGSplit(i,super.getM_minNoObj(),sumOfWeights,userWeights[i]);
		  currentModel[i].buildClassifier(data);
		  
		  // Check if useful split for current attribute
		  // exists and check for enumerated attributes with 
		  // a lot of values.
		  if (currentModel[i].checkModel())
		    if (super.getM_allData() != null) {
		      if ((data.attribute(i).isNumeric()) ||
			  (multiVal || Utils.sm((double)data.attribute(i).numValues(),
						(0.3*(double)super.getM_allData().numInstances())))){
			averageInfoGain = averageInfoGain+currentModel[i].infoGain();
			validModels++;
		      } 
		    } else {
		      averageInfoGain = averageInfoGain+currentModel[i].infoGain();
		      validModels++;
		    }
		}else
		  currentModel[i] = null;
	      }
	      
	      // Check if any useful split was found.
	      if (validModels == 0)
		return noSplitModel;
	      averageInfoGain = averageInfoGain/(double)validModels;

	      // Find "best" attribute to split on.
	      minResult = 0;
	      for (i=0;i<data.numAttributes();i++){
		if ((i != (data).classIndex()) &&
		    (currentModel[i].checkModel()))
		  
		  // Use 1E-3 here to get a closer approximation to the original
		  // implementation.
		  if ((currentModel[i].infoGain() >= (averageInfoGain-1E-3)) &&
		      Utils.gr(currentModel[i].gainRatio(),minResult)){ 
		    bestModel = currentModel[i];
		    minResult = currentModel[i].gainRatio();
		  } 
	      }

	      // Check if useful split was found.
	      if (Utils.eq(minResult,0))
		return noSplitModel;
	      
	      // Add all Instances with unknown values for the corresponding
	      // attribute to the distribution for the model, so that
	      // the complete distribution is stored with the model. 
	      bestModel.distribution().
		  addInstWithUnknown(data,bestModel.attIndex());
	      
	      // Set the split point analogue to C45 if attribute numeric.
	      if (super.getM_allData() != null)
		bestModel.setSplitPoint(super.getM_allData());
	      return bestModel;
	    }catch(Exception e){
	      e.printStackTrace();
	    }
	    return null;
	  }

}
