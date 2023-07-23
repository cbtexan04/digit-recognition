package com.willowtree.ai.imagerecognition.controllers;

import com.willowtree.ai.imagerecognition.neuralNetwork.NeuralNetworkPredictor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
@Slf4j
public class DigitRecognitionController {
    private final NeuralNetworkPredictor neuralNetwork;


    @Autowired
    public DigitRecognitionController(NeuralNetworkPredictor neuralNetworkPredictor) {
        this.neuralNetwork = neuralNetworkPredictor;
    }

    @PostMapping("/predict")
    public int predictDigit(@RequestBody double[] input) {
        int prediction = neuralNetwork.solveFromPixels(input);
        log.info("{}", prediction);
        return prediction;
    }
}
