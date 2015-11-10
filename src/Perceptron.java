import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils;

import java.util.Enumeration;
/**
 * Created by chris on 11/7/15.
 */


public class Perceptron extends weka.classifiers.Classifier implements weka.core.OptionHandler {

    Instances m_Instances;
    int m_NumAttributes;
    int m_Bias = 1;
    int m_NumIterations;
    double m_Learn;
    String m_File;
    int m_alterations;
    int m_NumWeights;
    double[] weights;

    public Perceptron() {
        super();
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {


        m_Instances = new Instances(data);
        m_NumAttributes = data.numAttributes();
        //clean data
        m_Instances.deleteWithMissingClass();
        m_NumWeights = m_Instances.numClasses()+1;
        weights = new double[m_NumWeights];
        for(int k = 0; k<m_NumWeights;k++)
            weights[k]=5; //initial weight
        for (int j = 0; j < m_NumIterations; j++){
            System.out.print("Iteration " + j + " ");
            Enumeration enu = m_Instances.enumerateInstances();
            while (enu.hasMoreElements()) {
                Instance inst = (Instance) enu.nextElement();
                Instances g = inst.dataset();
                Enumeration e = g.enumerateInstances();
                //while (e.hasMoreElements()) {
                    Instance h = (Instance) e.nextElement();
                    int n = h.numValues();
                    for (int i = 0; i < n; i++) {
                        System.out.print(h.value(i) );
                    }
                    System.out.print("-");
                //}
            }
            System.out.println();
        }
    }


    @Override
    public String toString() {

        StringBuffer buffer = new StringBuffer();
        buffer.append("Source file: " + m_File + "\n\n");
        buffer.append("Number of iterations: " + m_NumIterations + "\n");
        buffer.append("Learning rate: " + m_Learn + "\n");
        buffer.append("Total # weight updates =  " + "\n\n");



    //    -2.68
      //  0.17
        //8.29
        buffer.append("Final weights:\n");
        return buffer.toString();

    }

    public void distributionForInstance(){

    }

    public void setOptions(String[] options) throws Exception {

        String iterations = Utils.getOption('I', options);
        if (iterations.length() != 0) {
            m_NumIterations = Integer.parseInt(iterations);
        } else {
            m_NumIterations = 1;
        }
        String learn = Utils.getOption('L', options);
        if (learn.length() != 0) {
            m_Learn = Double.parseDouble(learn);
        } else {
            m_Learn = 1.0;
        }

        m_File = Utils.getOption('F', options);

    }

    public double classifyInstance(Instance inst){
        return 1;
    }

}
