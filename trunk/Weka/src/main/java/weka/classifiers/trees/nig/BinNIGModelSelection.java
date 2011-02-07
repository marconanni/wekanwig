package weka.classifiers.trees.nig;

import java.util.Enumeration;

import weka.classifiers.trees.j48.*;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;

public class BinNIGModelSelection extends BinC45ModelSelection {

	/**
	 * for serialization
	 */
	private static final long serialVersionUID = -1105790669619196916L;
	
	private double [] userWeights;

	/**
	 * @return the userWeights
	 */
	public double[] getUserWeights() {
		return userWeights;
	}

	/**
	 * @param userWeights the userWeights to set
	 */
	public void setUserWeights(double[] userWeights) {
		this.userWeights = userWeights;
	}

	public BinNIGModelSelection(int minNoObj, Instances allData, double[] userWeights) {
		super(minNoObj, allData);
		this.userWeights = userWeights;
	}

	/* (non-Javadoc)
	 * @see weka.classifiers.trees.j48.BinC45ModelSelection#selectModel(weka.core.Instances)
	 */
	@Override
	public ClassifierSplitModel selectModel(Instances data) {
		double minResult;
	    double currentResult;
	    BinNIGSplit [] currentModel;
	    BinNIGSplit bestModel = null;
	    NoSplit noSplitModel = null;
	    double averageInfoGain = 0;
	    int validModels = 0;
	    boolean multiVal = true;
	    Distribution checkDistribution;
	    double sumOfWeights;
	    int i;
	    
	    try{

	      // Check if all Instances belong to one class or if not
	      // enough Instances to split.
	      checkDistribution = new Distribution(data);
	      noSplitModel = new NoSplit(checkDistribution);
	      if (Utils.sm(checkDistribution.total(),2*getM_minNoObj()) ||
		  Utils.eq(checkDistribution.total(),
			   checkDistribution.perClass(checkDistribution.maxClass())))
		return noSplitModel;

	      // Check if all attributes are nominal and have a 
	      // lot of values.
	      Enumeration enu = data.enumerateAttributes();
	      while (enu.hasMoreElements()) {
		Attribute attribute = (Attribute) enu.nextElement();
		if ((attribute.isNumeric()) ||
		    (Utils.sm((double)attribute.numValues(),
			      (0.3*(double)getM_allData().numInstances())))){
		  multiVal = false;
		  break;
		}
	      }
	      currentModel = new BinNIGSplit[data.numAttributes()];
	      sumOfWeights = data.sumOfWeights();

	      // For each attribute.
	      for (i = 0; i < data.numAttributes(); i++){
		
		// Apart from class attribute.
		if (i != (data).classIndex()){
		  
		  // Get models for current attribute.
		  currentModel[i] = new BinNIGSplit(i,getM_minNoObj(),sumOfWeights, userWeights[i]);
		  currentModel[i].buildClassifier(data);
		  
		  // Check if useful split for current attribute
		  // exists and check for enumerated attributes with 
		  // a lot of values.
		  if (currentModel[i].checkModel())
		    if ((data.attribute(i).isNumeric()) ||
			(multiVal || Utils.sm((double)data.attribute(i).numValues(),
					      (0.3*(double)getM_allData().numInstances())))){
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
	      bestModel.setSplitPoint(getM_allData());
	      return bestModel;
	    }catch(Exception e){
	      e.printStackTrace();
	    }
	    return null;
	}
	
	

}
