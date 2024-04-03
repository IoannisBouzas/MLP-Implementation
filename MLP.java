import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import java.io.File;
import java.io.FileNotFoundException;

import java.util.Scanner;

public class MLP {
    private static int numInputs = 2; // Number of inputs
    private static int numClasses = 4; // Number of classes
    private int numOfFirstHiddenLayer; // Number of neurons in the first hidden layer
    private int numOfSecondHiddenLayer; // Number of neurons in the second hidden layer
    private int numOfThirdHiddenLayer; // Number of neurons in the third hidden layer
    private int[] hiddenLayers; // Number of neurons in each hidden layer  
    private String activationFunction; // Activation function for the hidden layers
    private double learningRate; // Learning rate for the network

    private double[][] neurons; // Neurons for each layer
    private double[][][] weights; // Weights for each neuron
    private double[][] biases; // Biases for each neuron

    

    public MLP(int numOfFirstHiddenLayer, int numOfSecondHiddenLayer, int numOfThirdHiddenLayer, String activationFunction , double learningRate) {
        this.numOfFirstHiddenLayer = numOfFirstHiddenLayer;
        this.numOfSecondHiddenLayer = numOfSecondHiddenLayer;
        this.numOfThirdHiddenLayer = numOfThirdHiddenLayer;
        this.activationFunction = activationFunction;
        this.learningRate = learningRate;
        this.hiddenLayers = new int[]{this.numOfFirstHiddenLayer, this.numOfSecondHiddenLayer, this.numOfThirdHiddenLayer};

        // Initialize the neurons and weights
        neurons = new double[hiddenLayers.length + 2][]; // +2 for the input and output layers
        weights = new double[hiddenLayers.length + 1][][]; // +1 for the output layer
        biases = new double[hiddenLayers.length + 1][]; // +1 for the output layer

    

        // Initialize the neurons for each layer
        neurons[0] = new double[numInputs]; // Input layer
        for (int i = 0; i < hiddenLayers.length; i++) {
            neurons[i + 1] = new double[hiddenLayers[i]]; // Hidden layers
        }
        neurons[neurons.length - 1] = new double[numClasses]; // Output layer
        

        // Initialize weights and biases for each layer
        for (int i = 0; i < hiddenLayers.length + 1; i++) {
            
            int inputSize = i > 0 ? hiddenLayers[i - 1] : numInputs;
            int outputSize = i < hiddenLayers.length ? hiddenLayers[i] : numClasses;

            weights[i] = new double[outputSize][inputSize];
            biases[i] = new double[outputSize];
            
            // Initialize weights and biases with random values between -1 and 1
            
            for (int j = 0; j < outputSize; j++) {
                biases[i][j] = 2 * Math.random() - 1;
                
                for (int k = 0; k < inputSize; k++) {
                    weights[i][j][k] = 2 * Math.random() - 1;
               
                }
            }
        }
    }   



    // Activation function
    public double activate(double x, String function) {
        switch (function) {
            case "relu":
                return Math.max(0, x);
            case "sigmoid":
                return 1 / (1 + Math.exp(-x));
            case "tanh":
                return Math.tanh(x);
            default:
                throw new IllegalArgumentException("Unknown activation function: " + function);
        }
    }

    // Derivative of the activation function
    public double activateDerivative(double x, String function) {
        switch (function) {
            case "relu":
                return x > 0 ? 1 : 0;
            case "sigmoid":
                double sigmoid = 1 / (1 + Math.exp(-x));
                return sigmoid * (1 - sigmoid);
            case "tanh":
                double tanh = Math.tanh(x);
                return 1 - tanh * tanh;
            default:
                throw new IllegalArgumentException("Unknown activation function: " + function);
        }
    }

    

    // Forward pass function
    public double[] forward(String hiddenActivation , double... inputs) {
        double[] layerOutput = inputs; // Start with the input layer

        // For each layer
        for (int i = 0; i < weights.length; i++) {
            double[] newLayerOutput = new double[weights[i].length];

            // For each neuron in the current layer
            for (int j = 0; j < weights[i].length; j++) {
                double weightedSum = biases[i][j]; // Start with the bias

                // For each input to the neuron
                for (int k = 0; k < weights[i][j].length; k++) {
                    weightedSum += layerOutput[k] * weights[i][j][k];
                }

                // Apply the activation function
                String function = (i < weights.length - 1) ? hiddenActivation : "sigmoid"; // Use activation functions for hidden layers and sigmoid for the output layer
                newLayerOutput[j] = activate(weightedSum, function);
            }

            layerOutput = newLayerOutput; // The output of this layer is the input to the next layer
            
        }
        return layerOutput; // The output of the final layer is the output of the network
    }



    // Backpropagation function
    public void backprop(double... targets) {
        
        
        // Calculate the error of the output layer
        double[] outputErrors = new double[neurons[neurons.length - 1].length];
        for (int i = 0; i < neurons[neurons.length - 1].length; i++) {
            outputErrors[i] = targets[i] - neurons[neurons.length - 1][i];
        }
    
        // Calculate the gradient of the output layer
        double[] outputGradient = new double[neurons[neurons.length - 1].length];
        for (int i = 0; i < neurons[neurons.length - 1].length; i++) {
            outputGradient[i] = outputErrors[i] * activateDerivative(neurons[neurons.length - 1][i], "sigmoid");
            
        }
    
        // Update the weights and biases of the output layer
        for (int i = 0; i < weights[weights.length - 1].length; i++) {
            for (int j = 0; j < weights[weights.length - 1][i].length; j++) {
                weights[weights.length - 1][i][j] += learningRate * outputGradient[i] * neurons[neurons.length - 2][j];
            
            }
            biases[biases.length - 1][i] += learningRate * outputGradient[i];
            
        }
    
        // Backpropagate the error to the hidden layers
        for (int i = weights.length - 2; i >= 0; i--) {
            double[] hiddenErrors = new double[neurons[i + 1].length];
            for (int j = 0; j < neurons[i + 1].length; j++) {
                hiddenErrors[j] = 0;
                for (int k = 0; k < neurons[i + 2].length; k++) {
                    hiddenErrors[j] += outputErrors[k] * weights[i + 1][k][j];
                }
            }
    
            // Calculate the gradient of the hidden layer
            double[] hiddenGradient = new double[neurons[i + 1].length];
            for (int j = 0; j < neurons[i + 1].length; j++) {
                hiddenGradient[j] = hiddenErrors[j] * activateDerivative(neurons[i + 1][j], activationFunction);
               
            }
    
            // Update the weights and biases of the hidden layer
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    weights[i][j][k] += learningRate * hiddenGradient[j] * neurons[i][k];
            
                }
                biases[i][j] += learningRate * hiddenGradient[j];

            }
    
            // The error of this layer is the input to the next layer
            outputErrors = hiddenErrors;
        }
    }

    

    // Training function
    public void train(String filePath, int B, double threshold) {
        double prevError = Double.MAX_VALUE;
        double error = 0;
        int epoch = 0;
        int minEpochs = 700;
                
        while(epoch < minEpochs || Math.abs(prevError - error) >= threshold){

            try {
                File file = new File(filePath);
                Scanner scanner = new Scanner(file);

                // Read the training data
                List<double[]> examples = new ArrayList<>();
                while (scanner.hasNextLine()) {
                    String line = scanner.nextLine();
                    String[] parts = line.split(",");
                    double[] example = new double[parts.length];
                    for (int i = 0; i < parts.length; i++) {
                        String part = parts[i];
                        part = part.replace("[", "").replace("]", "");
                        example[i] = Double.parseDouble(part);
                    }
                    examples.add(example);
                    
                }
                scanner.close();

                // Training loop
                while (true) {
                    Collections.shuffle(examples); // Shuffle the examples
                    prevError = error;
                    error = 0;

                    // Process each mini-batch
                    for (int i = 0; i < examples.size(); i += B) {
                       
                        // Process each example in the mini-batch
                        for (int j = i; j < i + B && j < examples.size(); j++) {
                            double[] example = examples.get(j);
                            double[] inputs = Arrays.copyOfRange(example, 0, numInputs);
                            double[] target = Arrays.copyOfRange(example, numInputs, example.length);
                            double[] targets = new double[numClasses];
                            if (target[0] == 1) {
                                targets[0] = 1;
                            } else if (target[0] == 2) {
                                targets[1] = 1;
                            } else if (target[0] == 3) {
                                targets[2] = 1;
                            } else if (target[0] == 4) {
                                targets[3] = 1;
                            }
                            
                            forward(activationFunction , inputs);
                            backprop(targets);
                        }
                        
                        
                    }

                    // Compute the total training error
                    for (double[] example : examples) {
                        double[] target = Arrays.copyOfRange(example, numInputs, example.length);
                        double[] targets = new double[numClasses];
                        if (target[0] == 1) {
                            targets[0] = 1;
                        } else if (target[0] == 2) {
                            targets[1] = 1;
                        } else if (target[0] == 3) {
                            targets[2] = 1;
                        } else if (target[0] == 4) {
                            targets[3] = 1;
                        }
                        double[] outputs = forward(activationFunction , example);
                        for (int i = 0; i < outputs.length; i++) {
                            error += Math.pow(targets[i] - outputs[i], 2);
                        }
                    }
                    error /= (examples.size() + 1e-10);

                    System.out.println("Epoch: " + epoch + ", Error: " + error);

                    // Break the loop if the difference in error is less than the threshold
                    // and the number of epochs is at least 700
                    if (epoch >= minEpochs && Math.abs(prevError - error) < threshold) {
                        break;
                    }

                    epoch++;
                }
            } catch (FileNotFoundException e) {
                System.out.println("An error occurred.");
                e.printStackTrace();
            }
        }
    }


    public void test(String filePath) {
        int correct = 0;
        int total = 0;

        try {
            File file = new File(filePath);
            Scanner scanner = new Scanner(file);

            // Read the test data
            List<double[]> examples = new ArrayList<>();
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] parts = line.split(",");
                double[] example = new double[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    String part = parts[i];
                    part = part.replace("[", "").replace("]", "");
                    example[i] = Double.parseDouble(part);
                }
                examples.add(example);
            }
            scanner.close();

            // Test each example
            for (double[] example : examples) {
                double[] inputs = Arrays.copyOfRange(example, 0, numInputs);
                double[] target = Arrays.copyOfRange(example, numInputs, example.length);
                double[] targets = new double[numClasses];
                if (target[0] == 1) {
                    targets[0] = 1;
                } else if (target[0] == 2) {
                    targets[1] = 1;
                } else if (target[0] == 3) {
                    targets[2] = 1;
                } else if (target[0] == 4) {
                    targets[3] = 1;
                }
                double[] outputs = forward(activationFunction , inputs);
                for (int i = 0; i < outputs.length; i++) {
                    if (Math.round(outputs[i]) == targets[i]) {
                        correct++;
                    }
                }
                total++;
            }

            // Print the percentage of correct decisions
            System.out.println("Generalization capacity: " + (double) correct / total * 100 + "%");
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    
    }

    public static void main(String[] args) {
        // Initialize the MLP with your desired parameters
        MLP mlp = new MLP(3, 3, 3, "sigmoid", 0.01);
    
        // Define the batch size and the threshold for training
        int batchSize = 5;
        double threshold = 0.01;
    
        // Train the MLP with the data from trainingData.txt
        mlp.train("trainingData.txt", batchSize, threshold);
    
        // Test the MLP with the data from testData.txt
        mlp.test("testingData.txt");
    }



}