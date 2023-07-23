package com.willowtree.ai.imagerecognition.neuralNetwork;

import com.willowtree.ai.imagerecognition.config.NetworkConfig;
import com.willowtree.ai.imagerecognition.services.FileSystemService;
import com.willowtree.ai.imagerecognition.services.models.ImageLabelPair;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;

@Getter
@Setter
@Slf4j
@Service
public class NeuralNetworkTrainer {
    private NetworkConfig networkConfig;
    private FileSystemService fileSystemService;

    @Autowired
    public NeuralNetworkTrainer(NetworkConfig networkConfig, FileSystemService fileSystemService) {
        this.networkConfig = networkConfig;
        this.fileSystemService = fileSystemService;
    }

    public void train(List<ImageLabelPair> trainingData, int epochs, double learningRate) throws IOException, URISyntaxException {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            for (int pairCount = 0; pairCount < trainingData.size(); pairCount++) {
                ImageLabelPair pair = trainingData.get(pairCount);
                double[] input = pair.getPixels();
                int label = pair.getLabel();

                // Convert input and label to matrices for matrix operations
                RealMatrix inputMatrix = MatrixUtils.createRowRealMatrix(input);
                RealMatrix targetOutput = MatrixUtils.createRowRealMatrix(new double[]{
                        label == 0 ? 1 : 0,
                        label == 1 ? 1 : 0,
                        label == 2 ? 1 : 0,
                        label == 3 ? 1 : 0,
                        label == 4 ? 1 : 0,
                        label == 5 ? 1 : 0,
                        label == 6 ? 1 : 0,
                        label == 7 ? 1 : 0,
                        label == 8 ? 1 : 0,
                        label == 9 ? 1 : 0
                });

                // Forward propagation
                RealMatrix hiddenLayerInput = inputMatrix.multiply(MatrixUtils.createRealMatrix(networkConfig.getWeightsInputHidden()));
                RealMatrix hiddenLayerOutput = MatrixUtils.createRealMatrix(NetworkFunctions.sigmoid(hiddenLayerInput.getData()));

                RealMatrix outputLayerInput = hiddenLayerOutput.multiply(MatrixUtils.createRealMatrix(networkConfig.getWeightsHiddenOutput()));
                RealMatrix outputLayerOutput = MatrixUtils.createRealMatrix(NetworkFunctions.softmax(outputLayerInput.getData()));

                // Calculate the loss (cross-entropy loss)
                double[] predictedProbabilities = outputLayerOutput.getRow(0);
                double loss = -Math.log(predictedProbabilities[label]);
                totalLoss += loss;

                // Backpropagation
                RealMatrix outputDelta = outputLayerOutput.subtract(targetOutput);
                RealMatrix hiddenError = outputDelta.multiply(MatrixUtils.createRealMatrix(networkConfig.getWeightsHiddenOutput()).transpose());
                double[] hiddenDelta = new double[networkConfig.getHiddenSize()];
                for (int i = 0; i < networkConfig.getHiddenSize(); i++) {
                    double sum = 0.0;
                    for (int j = 0; j < networkConfig.getOutputSize(); j++) {
                        sum += hiddenError.getEntry(0, j) * NetworkFunctions.sigmoidGradient(hiddenLayerInput.getData()[0])[i];
                    }
                    hiddenDelta[i] = sum;
                }

                // Update weights for hidden layer input/output
                for (int i = 0; i < networkConfig.getWeightsHiddenOutput().length; i++) {
                    for (int j = 0; j < networkConfig.getWeightsHiddenOutput()[i].length; j++) {
                        networkConfig.getWeightsHiddenOutput()[i][j] -= hiddenLayerOutput.getEntry(0, i) * outputDelta.getEntry(0, j) * learningRate;
                    }
                }

                for (int i = 0; i < networkConfig.getWeightsInputHidden().length; i++) {
                    for (int j = 0; j < networkConfig.getWeightsInputHidden()[i].length; j++) {
                        networkConfig.getWeightsInputHidden()[i][j] -= inputMatrix.getEntry(0, i) * hiddenDelta[j] * learningRate;
                    }
                }

                log.trace("Pair count: {}, Epoch: {}", pairCount, epoch);
            }

            // Incremental save
            log.debug("Saving weights to filesystem");
            fileSystemService.saveWeights(networkConfig.getWeightsFileName(), networkConfig.getWeightsInputHidden(), networkConfig.getWeightsHiddenOutput());

            double averageLoss = totalLoss / trainingData.size();
            log.info("Epoch " + epoch + " - Average Loss: " + averageLoss);
        }
    }
}
