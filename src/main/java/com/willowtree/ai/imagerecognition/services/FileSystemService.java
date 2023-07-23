package com.willowtree.ai.imagerecognition.services;

import com.willowtree.ai.imagerecognition.services.models.ImageLabelPair;
import com.willowtree.ai.imagerecognition.services.models.WeightData;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;

import java.io.*;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

@Service
public class FileSystemService {
    private final ResourceLoader resourceLoader;

    FileSystemService(ResourceLoader resourceLoader) {
        this.resourceLoader = resourceLoader;
    }

    public void saveWeights(String fileName, double[][] weightsInputHidden, double[][] weightsHiddenOutput) throws IOException, URISyntaxException {
        // Get the destination URL in the resources folder
        URL resourceURL = getClass().getResource("/" + fileName);
        Path destinationPath = Paths.get("src/main/resources/" + fileName);

        if (resourceURL == null) {
            // The resource file does not exist, create it in the resources folder
            Files.createDirectories(destinationPath.getParent());
        }

        resourceURL = destinationPath.toUri().toURL();

        // Write the weights to the file in the resources folder
        try (OutputStream outputStream = new FileOutputStream(new File(resourceURL.toURI()))) {
            // Create an ObjectOutputStream to write the weights matrices to the file
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(outputStream);
            // Write the weights matrices to the file
            objectOutputStream.writeObject(weightsInputHidden);
            objectOutputStream.writeObject(weightsHiddenOutput);
        }
    }

    public WeightData loadWeightsFromResources(String fileName) throws IOException, ClassNotFoundException {
        // Get the input stream from the resources folder
        try (InputStream inputStream = getClass().getResourceAsStream("/" + fileName);
             ObjectInputStream objectInputStream = new ObjectInputStream(inputStream)) {

            // Read the weights matrices from the input stream
            double[][] weightsInputHidden = (double[][]) objectInputStream.readObject();
            double[][] weightsHiddenOutput = (double[][]) objectInputStream.readObject();

            return new WeightData(weightsInputHidden, weightsHiddenOutput);
        }
    }

    public List<ImageLabelPair> readTrainingDataFromCSV(String csvFilePath) throws IOException {
        List<ImageLabelPair> trainingData = new ArrayList<>();

        try {
            // Load the resource from the classpath
            Resource resource = resourceLoader.getResource("classpath:" + csvFilePath);

            // Process the CSV content
            BufferedReader reader = new BufferedReader(new InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8));
            String line;
            boolean firstLineSkipped = false; // Flag to track if the first line is skipped
            while ((line = reader.readLine()) != null) {
                if (!firstLineSkipped) {
                    firstLineSkipped = true;
                    continue; // Skip the first line (header)
                }

                String[] parts = line.split(",");

                // Assuming the label is in the first column and pixel values start from the second column
                int label = Integer.parseInt(parts[0]);

                double[] pixels = new double[parts.length - 1];
                for (int i = 1; i < parts.length; i++) {
                    pixels[i - 1] = Double.parseDouble(parts[i]);
                }

                trainingData.add(new ImageLabelPair(pixels, label));
            }
        } catch (IOException e) {
            e.printStackTrace();
            throw e;
        }

        return trainingData;
    }

}
