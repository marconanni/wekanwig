package weka.classifiers.trees;

import weka.classifiers.trees.j48.BinC45ModelSelection;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.C45PruneableClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.PruneableClassifierTree;
import weka.classifiers.trees.nig.NIGModelSelection;
import weka.core.Instances;


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
	private double[] userWeigts;
	
	
	/**
	   * Generates the classifier.
	   *
	   * @param instances the data to train the classifier with
	   * @throws Exception if classifier can't be built successfully
	   */
	  public void buildClassifier(Instances instances) 
	       throws Exception {

	    ModelSelection modSelection;	 

	    if (super.getBinarySplits())
	      modSelection = new BinC45ModelSelection(super.getMinNumObj(), instances);
	    else
	      modSelection = new NIGModelSelection(super.getMinNumObj(), instances,userWeigts);
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

}
