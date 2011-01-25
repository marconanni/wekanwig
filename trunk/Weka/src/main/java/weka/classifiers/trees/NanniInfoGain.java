package weka.classifiers.trees;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.StringTokenizer;
import java.util.Vector;

import weka.classifiers.trees.j48.BinC45ModelSelection;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.C45PruneableClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.PruneableClassifierTree;
import weka.classifiers.trees.nig.NIGModelSelection;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;


/**
<!-- globalinfo-start -->
* Class for generating a pruned or unpruned Nanni Info Gain decision tree.
* It is an extension of C4.5
*  For more information about c45, see<br/>
* <br/>
* Ross Quinlan (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann Publishers, San Mateo, CA.
* <p/>
<!-- globalinfo-end -->
*
<!-- technical-bibtex-start -->
* BibTeX:
* <pre>
* &#64;book{Quinlan1993,
*    address = {San Mateo, CA},
*    author = {Ross Quinlan},
*    publisher = {Morgan Kaufmann Publishers},
*    title = {C4.5: Programs for Machine Learning},
*    year = {1993}
* }
* </pre>
* <p/>
<!-- technical-bibtex-end -->
*
<!-- options-start -->
* Valid options are: <p/>
* 
* <pre> -U
*  Use unpruned tree.</pre>
* 
* <pre> -C &lt;pruning confidence&gt;
*  Set confidence threshold for pruning.
*  (default 0.25)</pre>
* 
* <pre> -M &lt;minimum number of instances&gt;
*  Set minimum number of instances per leaf.
*  (default 2)</pre>
* 
* <pre> -R
*  Use reduced error pruning.</pre>
* 
* <pre> -N &lt;number of folds&gt;
*  Set number of folds for reduced error
*  pruning. One fold is used as pruning set.
*  (default 3)</pre>
* 
* <pre> -B
*  Use binary splits only.</pre>
* 
* <pre> -S
*  Don't perform subtree raising.</pre>
* 
* <pre> -L
*  Do not clean up after the tree has been built.</pre>
* 
* <pre> -A
*  Laplace smoothing for predicted probabilities.</pre>
* 
* <pre> -Q &lt;seed&gt;
*  Seed for random data shuffling (default 1).</pre>
* 
<!-- options-end -->
*
* @author Marco Nanni
* @version 1.0
*/
public class NanniInfoGain extends J48 {

	/**
	 * for serialization
	 */
	private static final long serialVersionUID = -84784947613788975L;
	
	/**
	 * a vector of weighs, one per attribute, used when the info gain is calculated
	 * for choosing the best split
	 */
	private double[] numericUserWeigts;
	
	/**
	 * A string containing the user weights, the user weights must be in the same number
	 * as the data attributes and separated by ';'. The weights must be in the range 0..1
	 * If the weights are less than the attributes, the remaining attributes have weight 1
	 * If the weights are more than the attributes, the exceeding weights are ignored.
	 */
	private String userWeights = "1";
	
	/**
	   * Generates the classifier.
	   *
	   * @param instances the data to train the classifier with
	   * @throws Exception if classifier can't be built successfully
	   */
	  public void buildClassifier(Instances instances) 
	       throws Exception {

	    ModelSelection modSelection;
	    
	    this.numericUserWeigts = this.parseWeights(userWeights, instances.numAttributes());

	    if (super.getBinarySplits())
	      modSelection = new BinC45ModelSelection(super.getMinNumObj(), instances);
	    else
	      modSelection = new NIGModelSelection(super.getMinNumObj(), instances,numericUserWeigts);
	    if (!getReducedErrorPruning())
	      super.setM_root( new C45PruneableClassifierTree(modSelection, !super.getUnpruned(), getM_CF(),
						    isM_subtreeRaising(), !isM_noCleanup()));
	    else
	      setM_root(new PruneableClassifierTree(modSelection, !isM_unpruned(), getM_numFolds(),
						   !isM_noCleanup(), getM_Seed()));
	    getM_root().buildClassifier(instances);
	    if (getBinarySplits()) {
	      ((BinC45ModelSelection)modSelection).cleanup();
	    } else {
	      ((C45ModelSelection)modSelection).cleanup();
	    }
	  }
	  
	  /**
	   * Returns an enumeration describing the available options.
	   *
	   * Valid options are: <p>
	   * 
	   * -W weights <br>
	   * The list of user weights, one for attribure, separated by ';'
	   * the weights must be in the range 0..1 <p> 
	   *
	   * -U <br>
	   * Use unpruned tree.<p>
	   *
	   * -C confidence <br>
	   * Set confidence threshold for pruning. (Default: 0.25) <p>
	   *
	   * -M number <br>
	   * Set minimum number of instances per leaf. (Default: 2) <p>
	   *
	   * -R <br>
	   * Use reduced error pruning. No subtree raising is performed. <p>
	   *
	   * -N number <br>
	   * Set number of folds for reduced error pruning. One fold is
	   * used as the pruning set. (Default: 3) <p>
	   *
	   * -B <br>
	   * Use binary splits for nominal attributes. <p>
	   *
	   * -S <br>
	   * Don't perform subtree raising. <p>
	   *
	   * -L <br>
	   * Do not clean up after the tree has been built.
	   *
	   * -A <br>
	   * If set, Laplace smoothing is used for predicted probabilites. <p>
	   *
	   * -Q <br>
	   * The seed for reduced-error pruning. <p>
	   *
	   * @return an enumeration of all the available options.
	   */
	  public Enumeration listOptions() {

	    Vector newVector = new Vector(10);
	    
	    newVector.
		addElement(new Option("\t The list of user weights, one for attribure, separated by ';' \n" +
				      "\t the weights must be in the range 0..1",
				      "W", 1, "-W <list of weights>"));

	    newVector.
		addElement(new Option("\tUse unpruned tree.",
				      "U", 0, "-U"));
	    newVector.
		addElement(new Option("\tSet confidence threshold for pruning.\n" +
				      "\t(default 0.25)",
				      "C", 1, "-C <pruning confidence>"));
	    newVector.
		addElement(new Option("\tSet minimum number of instances per leaf.\n" +
				      "\t(default 2)",
				      "M", 1, "-M <minimum number of instances>"));
	    newVector.
		addElement(new Option("\tUse reduced error pruning.",
				      "R", 0, "-R"));
	    newVector.
		addElement(new Option("\tSet number of folds for reduced error\n" +
				      "\tpruning. One fold is used as pruning set.\n" +
				      "\t(default 3)",
				      "N", 1, "-N <number of folds>"));
	    newVector.
		addElement(new Option("\tUse binary splits only.",
				      "B", 0, "-B"));
	    newVector.
	        addElement(new Option("\tDon't perform subtree raising.",
				      "S", 0, "-S"));
	    newVector.
	        addElement(new Option("\tDo not clean up after the tree has been built.",
				      "L", 0, "-L"));
	   newVector.
	        addElement(new Option("\tLaplace smoothing for predicted probabilities.",
				      "A", 0, "-A"));
	    newVector.
	      addElement(new Option("\tSeed for random data shuffling (default 1).",
				    "Q", 1, "-Q <seed>"));

	    return newVector.elements();
	  }
	  
	  /**
	   * Gets the current settings of the Classifier.
	   *
	   * @return an array of strings suitable for passing to setOptions
	   */
	  // controlla che non ci siano problemi con la dimensione dell'array che adesso è di 16 
	  //e non più di 14
	  public String [] getOptions() {
		  Vector <String> v = new Vector<String>();
		  v.add("-W");
		  v.add(this.getUserWeights());
		  v.addAll(Arrays.asList(super.getOptions()));
		  return v.toArray(new String[v.size()]);

	  }
	  
	  /**
	   * Parses a given list of options.
	   * 
	   <!-- options-start -->
	   * Valid options are: <p/>
	   * 
	   * <pre> -U
	   *  Use unpruned tree.</pre>
	   * 
	   * <pre> -W user weights;
	   *  The list of user weights, one for attribure, separated by ';'
	   *  the weights must be in the range 0..1</pre>
	   * 
	   * <pre> -C &lt;pruning confidence&gt;
	   *  Set confidence threshold for pruning.
	   *  (default 0.25)</pre>
	   * 
	   * <pre> -M &lt;minimum number of instances&gt;
	   *  Set minimum number of instances per leaf.
	   *  (default 2)</pre>
	   * 
	   * <pre> -R
	   *  Use reduced error pruning.</pre>
	   * 
	   * <pre> -N &lt;number of folds&gt;
	   *  Set number of folds for reduced error
	   *  pruning. One fold is used as pruning set.
	   *  (default 3)</pre>
	   * 
	   * <pre> -B
	   *  Use binary splits only.</pre>
	   * 
	   * <pre> -S
	   *  Don't perform subtree raising.</pre>
	   * 
	   * <pre> -L
	   *  Do not clean up after the tree has been built.</pre>
	   * 
	   * <pre> -A
	   *  Laplace smoothing for predicted probabilities.</pre>
	   * 
	   * <pre> -Q &lt;seed&gt;
	   *  Seed for random data shuffling (default 1).</pre>
	   * 
	   <!-- options-end -->
	   *
	   * @param options the list of options as an array of strings
	   * @throws Exception if an option is not supported
	   */
	  public void setOptions(String[] options) throws Exception {
		  String tmp = Utils.getOption("W", options);
		  this.setUserWeights(tmp);
		  super.setOptions(options);
	  }
	
	  /**
	   * Returns the tip text for this property
	   * @return tip text for this property suitable for
	   * displaying in the explorer/experimenter gui
	   */
	public String userWeightsTipText(){
		return "  A string containing the user weights, the user weights must be in the same number" +
				"  as the data attributes and separated by ';'. The weights must be in the range 0..1" +
				"  If the weights are less than the attributes, the remaining attributes have weight 1" +
				"  If the weights are more than the attributes, the exceeding weights are ignored.";
	}

	/**
	 * @return the userWeights
	 */
	public String getUserWeights() {
		return userWeights;
	}

	/**
	 * sets the user weights,
	 * 
	 * @param userWeights the userWeights to set
	 */
	public void setUserWeights(String userWeights) {
		this.userWeights = userWeights;
	}
	
	/**
	 * parses a string containing the user weights, the user weights must be in the same number
	 * as the data attributes and separated by ';'. The weights must be in the range 0..1
	 * If the weights are less than the attributes, the remaining attributes have weight 1
	 * If the weights are more than the attributes, the exceeding weights are ignored.
	 * 
	 * @param stringWeights the string containing the weights
	 * @param numberOfAttributes : the number of attributes of the data set
	 * @return a vector of weights
	 * @throws Exception if a weight is not in the range 0..1
	 */
	private double [] parseWeights(String stringWeights, int numberOfAttributes) throws Exception{
		double [] result = new double [numberOfAttributes];
		StringTokenizer st = new StringTokenizer(stringWeights,";");
		for (int k = 0; k <result.length ; k++) {
			if (st.hasMoreTokens()){
				double current = Double.parseDouble(st.nextToken().trim());
				if (current<0 || current >1)
					throw new Exception("The value "+current+" of the weight is not valid, must be in the range 0..1");
				result[k]= current;
			}
			else
				result [k] =1;
		}
		
		return result;
	}
	/**
	   * Main method for testing this class
	   *
	   * @param argv the commandline options
	   */
	  public static void main(String [] argv){
	    runClassifier(new NanniInfoGain(), argv);
	  }
}
