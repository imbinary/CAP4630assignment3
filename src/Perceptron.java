/**
 * UCF
 * CAP4630 Fall 2015 Assignment 3
 * William Orem and Austin Pantoja
 */

import weka.core.Utils;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.OptionHandler;
import weka.filters.Filter;
import weka.classifiers.Classifier;

import java.text.DecimalFormat;
import java.util.Enumeration;
import java.util.Random;


public class Perceptron extends Classifier implements OptionHandler {

    Instances instances;
    int iterations;
    double learningRate;
    String file;
    int alterations;
    double weights[];	//Bias value stored at weight[0]

	static final long serialVersionUID = 1L;


    public Perceptron() {
        super();
        alterations=0;
    }


	@SuppressWarnings("unchecked")
    @Override
    public void buildClassifier(Instances data) throws Exception {
        instances = new Instances(data);
        //clean data
        instances.deleteWithMissingClass();
        // set initial weight to random
        initWeights(instances.firstInstance().dataset());

        for (int j = 0; j < iterations; j++){
            System.out.print("Iteration " + j + ": ");
            Enumeration<Instance> enu = instances.enumerateInstances();
            while (enu.hasMoreElements()) {
                Instance inst = enu.nextElement();
                // do prediction
                if(prediction(inst) != inst.classValue()) {
                    System.out.print("0");
                    alterations++;
                    updateWeights(inst);
                }
                else
                    System.out.print("1");
                }
            System.out.println();
        }
    }


	@SuppressWarnings("unchecked")
    public void updateWeights(Instance data){
        Enumeration<Attribute> a = data.enumerateAttributes();
        double delta=1;
        if(data.classValue()==0.0)
            delta = -1;
        while (a.hasMoreElements()) {
            Attribute att = a.nextElement();
            if(att.index()==data.classIndex())
                continue;
            weights[att.index()+1] = weights[att.index()+1] + (learningRate * delta * data.value(att));
        }
        weights[0] = weights[0] + (learningRate * delta);
    }


	@SuppressWarnings("unchecked")
    public double prediction(Instance inst){
        double sum=0;
        int i = inst.classIndex();
        Enumeration<Attribute> a = inst.enumerateAttributes();
        while (a.hasMoreElements()) {
            Attribute att =  a.nextElement();
            if(att.index() == inst.classIndex())
                continue;
            sum += inst.value(att) * weights[att.index()+1];
        }
        sum += weights[0];
        if(sum > 0) return 1;
        return 0;
    }



	@SuppressWarnings("unchecked")
    public void initWeights(Instances data){
        Enumeration<Instance> enu = data.enumerateInstances();
        Instance inst;
        if ( enu.hasMoreElements() ) {
            inst = enu.nextElement();
            weights = new double[inst.numAttributes()];
            for (int i = 0; i < inst.numAttributes(); i++)
                weights[i] = Math.random();
        }
    }


    @Override
    public String toString() {
        DecimalFormat df = new DecimalFormat("#0.00");
        StringBuffer buffer = new StringBuffer();
        buffer.append("Source file: " + file + "\n\n");
        buffer.append("Number of iterations: " + iterations + "\n");
        buffer.append("Learning rate: " + learningRate + "\n");
        buffer.append("Total # weight updates =  " + alterations+"\n\n");
        buffer.append("Final weights:\n");
        for(double w : weights)
            buffer.append(df.format(w) + "\n");
        return buffer.toString();
    }


	@SuppressWarnings("unchecked")
    public double[] distributionForInstance(Instance inst) throws Exception {
        Enumeration<Attribute> a = inst.enumerateAttributes();
        while (a.hasMoreElements()) {
            Attribute att = a.nextElement();
            att.setWeight(weights[att.index()]);

        }
        double pred = prediction(inst);
        double[] result = new double[2];
        result[0]=1-pred;
        result[1]=1-result[0];
        return result;
    }


    public void setOptions(String[] options) throws Exception {
        String iters = Utils.getOption('I', options);
        if (iters.length() != 0)
            iterations = Integer.parseInt(iters);
		else
            iterations = 1;

        String learn = Utils.getOption('L', options);
        if (learn.length() != 0)
            learningRate = Double.parseDouble(learn);
		else
            learningRate = 1.0;

        file = Utils.getOption('F', options);
    }

}
