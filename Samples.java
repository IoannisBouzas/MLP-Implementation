import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;



class Samples{
    

    private double[][] trainingData = new double[4000][2];
    private double[][] testingData = new double[4000][2];
    private int[] targets = new int[8000];
    private Random rand  = new Random(123);
    private int maxSeed = 1;
    private int minSeed = -1;
    
    
    public void initializeSamples(){
        try{
            PrintWriter mapWriter = new PrintWriter("dataMap.txt", "UTF-8");
            PrintWriter trainingWriter = new PrintWriter("trainingData.txt", "UTF-8");
            PrintWriter testingWriter = new PrintWriter("testingData.txt", "UTF-8");


            for(int i = 0; i < 4000; i++){
                for(int j = 0; j < 2; j++){
                    trainingData[i][j] = minSeed + (maxSeed - minSeed) * rand.nextDouble();
                    
                }
                int targetTraining = classify(trainingData[i][0], trainingData[i][1]);
                

                targets[i] = targetTraining;
                

                trainingWriter.println(Arrays.toString(trainingData[i]) + "," + targetTraining);
                                

                mapWriter.println(Arrays.toString(trainingData[i]) + "," + targetTraining);
                   
            }

            for(int i = 0; i < 4000; i++){
                for(int j = 0; j < 2; j++){
                    testingData[i][j] = minSeed + (maxSeed - minSeed) * rand.nextDouble();
                }
                int targetTesting   = classify(testingData[i][0], testingData[i][1]);
                
                targets[i + 4000] = targetTesting;
                
                testingWriter.println(Arrays.toString(testingData[i]) + "," + targetTesting);                
                
                mapWriter.println(Arrays.toString(testingData[i]) + "," + targetTesting);   
            }

            mapWriter.close();
            trainingWriter.close();
            testingWriter.close();


        }catch(IOException e){
            System.out.println("Error while writting in the file");
        }
    }


    private int classify(double x1, double x2) {
        if (Math.pow(x1 - 0.5, 2) + Math.pow(x2 - 0.5, 2) < 0.2) {
            if (x1 > 0.5)
                return 1;
            if (x1 < 0.5)
                return 2;
        }
        if (Math.pow(x1 + 0.5, 2) + Math.pow(x2 + 0.5, 2) < 0.2) {
            if (x1 > -0.5)
                return 1;
            if (x1 < -0.5)
                return 2;
        }
        if (Math.pow(x1 - 0.5, 2) + Math.pow(x2 + 0.5, 2) < 0.2) {
            if (x1 > 0.5)
                return 1;
            if (x1 < 0.5)
                return 2;
        }
        if (Math.pow(x1 + 0.5, 2) + Math.pow(x2 - 0.5, 2) < 0.2) {
            if (x1 > -0.5)
                return 1;
            if (x1 < -0.5)
                return 2;
        }
        if (x1 > 0)
            return 3;
        if (x1 < 0)
            return 4;
        return 0; // Default category
    }


    public static void main (String[] args) {
        Samples test = new Samples();
        test.initializeSamples();
    }


}