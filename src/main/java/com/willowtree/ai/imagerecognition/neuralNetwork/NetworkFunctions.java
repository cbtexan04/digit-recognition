package com.willowtree.ai.imagerecognition.neuralNetwork;

import java.util.Arrays;

public class NetworkFunctions {
    public static double[][] softmax(double[][] z) {
        // Implement the softmax activation function element-wise
        // This is used for the output layer in multi-class classification
        double[][] result = new double[z.length][z[0].length];
        for (int i = 0; i < z.length; i++) {
            double sumExp = 0.0;
            for (int j = 0; j < z[i].length; j++) {
                result[i][j] = Math.exp(z[i][j]);
                sumExp += result[i][j];
            }
            for (int j = 0; j < z[i].length; j++) {
                result[i][j] /= sumExp;
            }
        }
        return result;
    }

    public static double[][] sigmoid(double[][] z) {
        // Implement the sigmoid activation function element-wise for 2D arrays
        double[][] result = new double[z.length][z[0].length];
        for (int i = 0; i < z.length; i++) {
            for (int j = 0; j < z[i].length; j++) {
                result[i][j] = 1.0 / (1.0 + Math.exp(-z[i][j]));
            }
        }
        return result;
    }

    public static double[] sigmoid(double[] original) {
        double[] z = minMaxScaling(original);
        double[] result = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            double sigmoidValue = 1.0 / (1.0 + Math.exp(-z[i]));
            result[i] = sigmoidValue;
        }
        return result;
    }

    private static double[] minMaxScaling(double[] values) {
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        // Find the minimum and maximum values
        for (double value : values) {
            if (value < min) {
                min = value;
            }
            if (value > max) {
                max = value;
            }
        }

        // Apply Min-Max scaling
        double[] scaledValues = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            scaledValues[i] = (values[i] - min) / (max - min);
        }

        return scaledValues;
    }

    public static double[] sigmoidGradient(double[] original) {
        // Attempt to better normalize the data
        double[] z = minMaxScaling(original);

        double[] sigmoidZ = sigmoid(z);
        boolean anyBelowPoint5 = Arrays.stream(sigmoidZ).filter(e -> e < 0.5).count() > 0;
        if (anyBelowPoint5) {
            System.out.println("Debug found some below 0.5");
        }


        double[] result = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            double r = sigmoidZ[i] * (1.0 - sigmoidZ[i]);
            result[i] = r;
            if (r > 0.5) {
                System.out.println("Debug: "+  r);
            }
        }

        return result;
    }
}
