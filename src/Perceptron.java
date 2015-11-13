import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils;
import weka.filters.Filter;

import java.text.DecimalFormat;
import java.util.Enumeration;
import java.util.Random;

/**
 * Created by chris on 11/7/15.
 */


public class Perceptron extends weka.classifiers.Classifier implements weka.core.OptionHandler {

    Instances m_Instances;
    boolean start = true;
    int m_Bias = 1;
    int m_NumIterations;
    double m_Learn;
    String m_File;
    int m_alterations;
    Random rand;
    double weights[];

    public Perceptron() {
        super();
        rand = new Random();
        m_alterations=0;

    }

    @Override
    public void buildClassifier(Instances data) throws Exception {


        m_Instances = new Instances(data);

        //clean data
        m_Instances.deleteWithMissingClass();

        // set initial weight to random
        initWeights(m_Instances.firstInstance().dataset());

        for (int j = 0; j < m_NumIterations; j++){
            System.out.print("Iteration " + j + " ");
            Enumeration enu = m_Instances.enumerateInstances();
            while (enu.hasMoreElements()) {
                Instance inst = (Instance) enu.nextElement();
                // do prediction
                if(prediction(inst) != inst.classValue()) {
                    System.out.print("0");
                    m_alterations++;
                    updateWeights(inst);
                }
                else
                    System.out.print("1");
                }
            System.out.println();
        }
    }




    public void updateWeights(Instance data){
        Enumeration a = data.enumerateAttributes();
        double delta=1;
        if(data.classValue()==0.0)
            delta = -1;
        while (a.hasMoreElements()) {
            Attribute att = (Attribute) a.nextElement();
            if(att.index()==data.classIndex())
                continue;
            weights[att.index()+1] = weights[att.index()+1] + (m_Learn * delta * data.value(att));
        }
        weights[0] = weights[0] + (m_Learn * delta * m_Bias);
    }

    public double prediction(Instance inst){
        double sum=0;
        int i = inst.classIndex();
       // System.out.print(i+ " | ");
        Enumeration a = inst.enumerateAttributes();
        while (a.hasMoreElements()) {
            Attribute att = (Attribute) a.nextElement();
            if(att.index()==inst.classIndex())
                continue;
            sum += inst.value(att) * weights[att.index()+1];
            //System.out.print(inst.value(att)+ " ");
        }
        sum += m_Bias * weights[0];
        if(sum >0)
            return 1;
        return 0;
    }



    public void initWeights(Instances data){
        Enumeration enu = data.enumerateInstances();
        Instance inst;
        if ( enu.hasMoreElements() ) {
            inst = (Instance) enu.nextElement();

            weights = new double[inst.numAttributes()];
            start = false;
            for (int i = 0; i < inst.numAttributes(); i++)
                weights[i] = rand.nextDouble();

        }
    }

    @Override
    public String toString() {
        DecimalFormat df = new DecimalFormat("#0.00");
        StringBuffer buffer = new StringBuffer();
        buffer.append("Source file: " + m_File + "\n\n");
        buffer.append("Number of iterations: " + m_NumIterations + "\n");
        buffer.append("Learning rate: " + m_Learn + "\n");
        buffer.append("Total # weight updates =  " + m_alterations+"\n\n");


        buffer.append("Final weights:\n");
        for(double w : weights)
            buffer.append(df.format(w) + "\n");
        return buffer.toString();

    }

    public double[] distributionForInstance(Instance inst) throws Exception {

        Enumeration a = inst.enumerateAttributes();
        while (a.hasMoreElements()) {
            Attribute att = (Attribute) a.nextElement();
            att.setWeight(weights[att.index()]);

        }

        double[] result = new double[2];
        double p = prediction(inst);
        result[0]=1-p;
        result[1]=1-result[0];
        //System.out.print(result[0]+ " "+ result[1]+ " ");
        return result;
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


}
