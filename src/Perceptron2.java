/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    Tom Bylander started with:
 *
 *        VotedPerceptron.java
 *        Copyright (C) 1999 Eibe Frank
 *
 *    making modifications appropriate for implementing perceptron
 *    with a margin.  He does not know how the Weka group wants to
 *    handle copyright, so he left the old copyright notice.
 *
 */



import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.Filter;
import weka.core.*;
import java.util.*;

/**
 * Implements the perceptron algorithm with a margin.
 * Globally transforms nominal attributes into binary ones,
 * replaces all missing values, and (not yet implemented)
 * normalizes the attributes.
 * A version of this algorithm (though with a variable
 * increment) is in<p>
 *
 * R. O. Duda and P. E. Hart (1973). <i>Pattern Classification and
 * Scene Analysis</i>.  John Wiley, New York, NY. <p>
 *
 * Valid options are:<p>
 *
 * -I num <br>
 * The number of iterations to be performed. (default 1)<p>
 *
 * -L num <br>
 * The learning rate (not yet implemented). (default 1)<p>
 *
 * -S num <br>
 * The seed for the random number generator. (default 1)<p>
 *
 * This is the author of VotedPercetron.java
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * This is who started modifying it to do perceptron with a margin.
 * @author Tom Bylander (bylander@cs.utsa.edu)
 * @version $Revision: 0.1 $
 */
public class Perceptron2 extends Classifier implements OptionHandler {

    /** The number of iterations */
    private int m_NumIterations = 1;

    /** The actual number of alterations */
    private int m_K = 0;

    /** The weights of the perceptron */
    private double[] m_Weights = null;

    /** (not yet implemented) The bias weight of the perceptron. */

    /** The training instances */
    private Instances m_Train = null;

    /** Seed used for shuffling the dataset */
    private int m_Seed = 1;

    /** The filter used to make attributes numeric. */
    private NominalToBinary m_NominalToBinary;

    /** The filter used to get rid of missing values. */
    private ReplaceMissingValues m_ReplaceMissingValues;

    /** (not yet implemented) The filter used to normalize attributes. */
    /** Consider weka.filters.attributes.unsupervised.Normalize */

    /** (not yet implemented) A variable for the learning rate. */

    /**
     * Returns a string describing this classifier
     * @return a description of the classifier suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
        return "Implements the perceptron2 algorithm with a margin."
                + " Globally transforms nominal attributes into binary ones,"
                + " replaces all missing values, and (not yet implemented)"
                + " normalizes the attributes."
                + " A version of this algorithm (though with a variable"
                + " increment) is in\n\n"
                + "R. O. Duda and P. E. Hart (1973). <i>Pattern Classification and"
                + "Scene Analysis</i>.  John Wiley, New York, NY.";
    }

    /**
     * Returns an enumeration describing the available options.
     * -L option is not yet implemented
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {

        Vector newVector = new Vector(4);

        newVector.addElement(new Option("\tThe number of iterations to be performed.\n"
                + "\t(default 1)",
                "I", 1, "-I <int>"));
        newVector.addElement(new Option("\tThe seed for the random number generation.\n"
                + "\t(default 1)",
                "S", 1, "-S <int>"));

        return newVector.elements();
    }

    /**
     * Parses a given list of options. Valid options are:<p>
     *
     * -I num <br>
     * The number of iterations to be performed. (default 1)<p>
     *
     * -L num <br>
     * The learning rate (not yet implemented). (default 1)<p>
     *
     * -S num <br>
     * The seed for the random number generator. (default 1)<p>
     *
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

        String iterationsString = Utils.getOption('I', options);
        if (iterationsString.length() != 0) {
            m_NumIterations = Integer.parseInt(iterationsString);
        } else {
            m_NumIterations = 1;
        }
        String seedString = Utils.getOption('S', options);
        if (seedString.length() != 0) {
            m_Seed = Integer.parseInt(seedString);
        } else {
            m_Seed = 1;
        }
    }

    /**
     * Gets the current settings of the classifier.
     * -L option is not yet implemented
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {

        String[] options = new String [8];
        int current = 0;

        options[current++] = "-I"; options[current++] = "" + m_NumIterations;
        options[current++] = "-S"; options[current++] = "" + m_Seed;
        while (current < options.length) {
            options[current++] = "";
        }
        return options;
    }

    /**
     * Learns the perceptron.
     *
     * @exception Exception if something goes wrong during building
     */
    public void buildClassifier(Instances insts) throws Exception {

        if (insts.checkForStringAttributes()) {
            throw new UnsupportedAttributeTypeException("Cannot handle string attributes!");
        }
        if (insts.numClasses() > 2) {
            throw new Exception("Can only handle two-class datasets!");
        }
        if (insts.classAttribute().isNumeric()) {
            throw new UnsupportedClassTypeException("Can't handle a numeric class!");
        }

        // Filter data
        m_Train = new Instances(insts);
        m_Train.deleteWithMissingClass();
        m_NominalToBinary = new NominalToBinary();
        m_NominalToBinary.setInputFormat(m_Train);
        m_Train = Filter.useFilter(m_Train, m_NominalToBinary);

        m_ReplaceMissingValues = new ReplaceMissingValues();
        m_ReplaceMissingValues.setInputFormat(m_Train);
        m_Train = Filter.useFilter(m_Train, m_ReplaceMissingValues);

    /* (not yet implemented) Normalize the attributes. */

        /** Randomize training data */
        m_Train.randomize(new Random(m_Seed));

        /** (not yet implemented) Make space for and initialize the weights */

        /** Compute perceptron */
        m_K = 0;
        for (int it = 0; it < m_NumIterations; it++) {
            for (int i = 0; i < m_Train.numInstances(); i++) {
                Instance inst = m_Train.instance(i);
                double prediction = makePrediction(inst);
                int classValue = (int) inst.classValue();

	/* (not yet implemented) classValue will be 0 or 1 */
	/* Needs to be converted to -1 or 1 */

	/* (not yet implemented) a function to compute the error */
      //          if (error(prediction, classValue) > 0) {
	  /* (not yet implemented) a function to update the weights */
      //              update(inst, classValue);
                    m_K++;
      //          }
            }

      /* should break if no updates this epoch */

        }
    }

    /**
     * Outputs the distribution for the given output.
     *
     * Pipes output of perceptron through sigmoid function.
     * @param inst the instance for which distribution is to be computed
     * @return the distribution
     * @exception Exception if something goes wrong
     */
    public double[] distributionForInstance(Instance inst) throws Exception {

        // Filter instance
        m_NominalToBinary.input(inst);
        m_NominalToBinary.batchFinished();
        inst = m_NominalToBinary.output();

        m_ReplaceMissingValues.input(inst);
        m_ReplaceMissingValues.batchFinished();
        inst = m_ReplaceMissingValues.output();

    /* (not yet implemented) need to normalize this instance */

    /* (not yet implemented) get prediction */
        double prediction = makePrediction(inst);

    /* Maybe (-prediction) should be (+prediction) */
        double[] result = new double[2];
        result[1] = 1 / (1 + Math.exp(-prediction));
        result[0] = 1 - result[1];

        return result;
    }

    /**
     * (not yet implemented)
     * Returns textual description of classifier.
     *
     */
    public String toString() {

        return "Perceptron: toString not yet implemented";
    }

  /* (not yet implemented) TipText and getter and setter for learning rate */


    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String numIterationsTipText() {
        return "Number of iterations to be performed.";
    }

    /**
     * Get the value of NumIterations.
     *
     * @return Value of NumIterations.
     */
    public int getNumIterations() {

        return m_NumIterations;
    }

    /**
     * Set the value of NumIterations.
     *
     * @param v  Value to assign to NumIterations.
     */
    public void setNumIterations(int v) {

        m_NumIterations = v;
    }

    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String seedTipText() {
        return "Seed for the random number generator.";
    }

    /**
     * Get the value of Seed.
     *
     * @return Value of Seed.
     */
    public int getSeed() {

        return m_Seed;
    }

    /**
     * Set the value of Seed.
     *
     * @param v  Value to assign to Seed.
     */
    public void setSeed(int v) {

        m_Seed = v;
    }

    /**
     * (not yet implemented) Compute a prediction for the instance.
     * Don't include the class attribute in the dot product.
     * Don't forget the bias weight.
     */
    private double makePrediction(Instance inst) throws Exception {
    return 0.0;
    }

    /**
     * Main method.
     */
    public static void main(String[] argv) {

        try {
            System.out.println(Evaluation.evaluateModel(new Perceptron(), argv));
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }
}