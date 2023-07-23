package com.willowtree.ai.imagerecognition.services;

import com.willowtree.ai.imagerecognition.config.NetworkConfig;
import com.willowtree.ai.imagerecognition.services.models.ImageLabelPair;
import com.willowtree.ai.imagerecognition.neuralNetwork.NeuralNetworkTrainer;
import com.willowtree.ai.imagerecognition.services.models.WeightData;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.io.*;
import java.net.URISyntaxException;
import java.util.List;

@Service
@Slf4j
@Getter
public class NeuralNetworkService {
    private final FileSystemService fileSystemService;
    private final NetworkConfig networkConfig;
    private final NeuralNetworkTrainer neuralNetworkTrainer;

    @Autowired
    public NeuralNetworkService(
            FileSystemService fileSystemService,
            NetworkConfig networkConfig,
            NeuralNetworkTrainer neuralNetworkTrainer
    ) {
        this.fileSystemService = fileSystemService;
        this.networkConfig = networkConfig;
        this.neuralNetworkTrainer = neuralNetworkTrainer;
    }

    @PostConstruct
    public void trainNeuralNetwork() throws IOException, URISyntaxException {
        try {
            WeightData weightData = fileSystemService.loadWeightsFromResources(networkConfig.getWeightsFileName());
            networkConfig.setWeightsInputHidden(weightData.getWeightsInputHidden());
            networkConfig.setWeightsHiddenOutput(weightData.getWeightsHiddenOutput());
            log.info("Loaded weights from file");
            return;
        } catch (IOException | ClassNotFoundException | NullPointerException e) {
            log.warn("Could not load weights. Processing...");
        }

        if (!getNetworkConfig().isTrainingEnabled()) {
            log.info("Training disabled; Not further adjusting weights");
            return;
        }

        // Fetch training data from csv
        log.info("Loading csv data from filesystem");
        List<ImageLabelPair> trainingData = fileSystemService.readTrainingDataFromCSV(networkConfig.getTrainingFileName());

        // Training will set new weights directly
        log.info("Starting training based on data with {} training sets", trainingData.size());
        neuralNetworkTrainer.train(trainingData, networkConfig.getEpochs(), networkConfig.getLearningRate());

        // Save off the new weights
        log.info("Saving off weights to filesystem");
        fileSystemService.saveWeights(networkConfig.getWeightsFileName(), networkConfig.getWeightsInputHidden(), networkConfig.getWeightsHiddenOutput());
    }
}
