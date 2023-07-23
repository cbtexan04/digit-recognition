package com.willowtree.ai.imagerecognition.neuralNetwork;

import com.willowtree.ai.imagerecognition.config.NetworkConfig;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Getter
@Setter
@Slf4j
@Service
public class NeuralNetworkPredictor {
    private NetworkConfig networkConfig;

    @Autowired
    public NeuralNetworkPredictor(NetworkConfig networkConfig) {
        this.networkConfig = networkConfig;
    }

    private double[] predict(double[] input) {
        // Perform forward propagation to make predictions for the input
        RealMatrix inputMatrix = MatrixUtils.createRowRealMatrix(input);

        // Compute activations for the hidden layer
        RealMatrix hiddenLayerInput = inputMatrix.multiply(MatrixUtils.createRealMatrix(networkConfig.getWeightsInputHidden()));
        RealMatrix hiddenLayerOutput = MatrixUtils.createRealMatrix(NetworkFunctions.sigmoid(hiddenLayerInput.getData()));

        // Compute activations for the output layer
        RealMatrix outputLayerInput = hiddenLayerOutput.multiply(MatrixUtils.createRealMatrix(networkConfig.getWeightsHiddenOutput()));
        RealMatrix outputLayerOutput = MatrixUtils.createRealMatrix(NetworkFunctions.softmax(outputLayerInput.getData()));

        // Convert the output to a 1D array (column vector)
        return outputLayerOutput.getColumnVector(0).toArray();
    }

    public int solveFromPixels(double[] arr) {
        double[] prediction = predict(arr);

        int maxIndex = 0;
        double maxVal = prediction[0];
        for (int i = 1; i < prediction.length; i++) {
            if (prediction[i] > maxVal) {
                maxVal = prediction[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}
