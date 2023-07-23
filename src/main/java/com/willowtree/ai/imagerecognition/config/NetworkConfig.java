package com.willowtree.ai.imagerecognition.config;

import lombok.AccessLevel;
import lombok.Data;
import lombok.Setter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.Random;

@Component
@Data
public class NetworkConfig {
    @Value("${config.toggles.enable-training}")
    private boolean trainingEnabled;

    @Value("${config.network.epochs}")
    private int epochs;

    @Value("${config.network.learning-rate}")
    private double learningRate;

    @Value("${config.network.hidden-layer-size}")
    private int hiddenSize;

    @Value("${config.files.training-file-name}")
    private String trainingFileName;

    @Value("${config.files.weights-file-name}")
    private String weightsFileName;

    // 28x28, this should NOT be configurable
    @Setter(AccessLevel.NONE)
    private final int inputSize = 784;

    // 10 digits to choose from, this should NOT be configurable
    @Setter(AccessLevel.NONE)
    private final int outputSize = 10;

    // NeuralNetworkService will inject these values during post construct
    private double[][] weightsInputHidden;
    private double[][] weightsHiddenOutput;

    public double[][] getWeightsInputHidden() {
        if (weightsInputHidden == null) {
            this.setWeightsInputHidden(generateDummyWeights(inputSize, hiddenSize));
        }
        return this.weightsInputHidden;
    }

    public double[][] getWeightsHiddenOutput() {
        if (weightsHiddenOutput == null) {
            this.setWeightsHiddenOutput(generateDummyWeights(hiddenSize, outputSize));
        }
        return this.weightsHiddenOutput;
    }

    private double[][] generateDummyWeights(int numRows, int numCols) {
        // Generate random dummy weights (as placeholders)
        // For actual training, replace this with your training process
        Random random = new Random();
        double[][] dummyWeights = new double[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                dummyWeights[i][j] = random.nextDouble();
            }
        }
        return dummyWeights;
    }
}
