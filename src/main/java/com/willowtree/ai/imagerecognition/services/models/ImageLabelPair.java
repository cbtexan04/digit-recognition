package com.willowtree.ai.imagerecognition.services.models;

import lombok.Getter;

@Getter
public class ImageLabelPair {
    private final double[] pixels;
    private final int label;

    public ImageLabelPair(double[] pixels, int label) {
        this.pixels = pixels;
        this.label = label;
    }
}