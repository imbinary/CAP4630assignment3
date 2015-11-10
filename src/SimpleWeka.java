import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileNotFoundException;
import java.text.DecimalFormat;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.core.FastVector;
import weka.core.Instances;
 
public class SimpleWeka {
	private static DecimalFormat df = new DecimalFormat("0.00"); 
	
	public static BufferedReader readDataFile(String filename) {
	
		BufferedReader inputReader = null;
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
		return inputReader;
	}
 
	public static Evaluation classify(
			Classifier model, Instances dataSet ) throws Exception {
		
		Evaluation evaluation = new Evaluation(dataSet);
		model.buildClassifier(dataSet);

		evaluation.evaluateModel(model, dataSet);
		System.out.println( evaluation.toSummaryString("\nResults:\n", true) );
		
		return evaluation;
	}
 
	public static double calculateAccuracy(FastVector predictions) {
		double correct = 0;
 
		for (int i = 0; i < predictions.size(); i++) {
			NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
			if (np.predicted() == np.actual()) {
				correct++;
			}
		}
		return 100 * correct / predictions.size();
	}
 
	public static void main(String[] args) throws Exception {
		
		// Load data file of instances
		BufferedReader datafile = readDataFile(args[0]);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);

		// Choose your custom perceptron classifier
		Classifier model = new Perceptron();
		
		// Set the classifier options
		String[] options = new String[6];
		options[0] = "-F";
		options[1] = args[0];
		options[2] = "-I";
		options[3] = args[1];
		options[4] = "-L";
		options[5] = args[2];
		model.setOptions(options);
				
		// Run classifier and report results
		FastVector predictions = new FastVector();
		Evaluation validation = classify(model, data);
		predictions.appendElements(validation.predictions());

		// Final score
		double accuracy = calculateAccuracy(predictions);
		System.out.println("Accuracy of "
				+ model.getClass().getSimpleName() + ": " 
				+ df.format(accuracy) + " %\n");
 
		// Show classifier intermediate data
		System.out.println(model.toString() + "\n\n"); 
	}
}
			
