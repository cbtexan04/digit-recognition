package com.willowtree.ai.imagerecognition.services.models;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.RequiredArgsConstructor;

@Data
@AllArgsConstructor
public class WeightData {
    private double[][] weightsInputHidden;
    private double[][] weightsHiddenOutput;
}
