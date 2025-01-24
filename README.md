

# Handwritten Text Detection OCR Model using CNN + RNN with LSTM

## Overview
This project implements an Optical Character Recognition (OCR) model for handwritten text detection. The architecture combines Convolutional Neural Networks (CNNs) for feature extraction with Recurrent Neural Networks (RNNs) using Long Short-Term Memory (LSTM) units for sequential text recognition. The model is designed to process handwritten text images and convert them into machine-readable text.

## Features
- **CNN for Feature Extraction**: The CNN layers extract spatial features from input images.
- **RNN with LSTM for Sequence Modeling**: The LSTM layers handle sequential dependencies in text.
- **CTC Loss Function**: Connectionist Temporal Classification (CTC) loss is used to align predicted sequences with ground truth labels without requiring pre-segmented data.
- **End-to-End Training**: The model is trained to directly map input images to corresponding text labels.

## Requirements
The following libraries and frameworks are required to run the project:

- Python 3.8+
- TensorFlow 2.x
- NumPy
- OpenCV
- Matplotlib
- scikit-learn
- pandas

Install dependencies using the command:
```bash
pip install -r requirements.txt
```

## Dataset
The model is trained and tested on handwritten text datasets. Popular options include:
- IAM Handwriting Database
- MNIST for handwriting digits
- Custom datasets (if available)

Ensure the dataset is preprocessed into image-text pairs.

## Preprocessing
1. **Image Resizing**: Normalize all input images to a fixed size (e.g., 128x32 pixels).
2. **Grayscale Conversion**: Convert images to grayscale for simplicity.
3. **Normalization**: Normalize pixel values to the range [0, 1].
4. **Text Tokenization**: Encode ground truth text into sequences of integers for training.

## Model Architecture
1. **Input Layer**: Accepts preprocessed image input.
2. **CNN Layers**: Convolutional and pooling layers extract spatial features.
3. **RNN Layers**: LSTM layers model sequential text data.
4. **CTC Layer**: Decodes predicted sequences into readable text.

## Training
1. Split the dataset into training and testing sets.
2. Compile the model using the Adam optimizer and CTC loss.
3. Train the model using the training set and validate it on the testing set.

Example training script:
```python
model.compile(optimizer='adam', loss=tf.keras.backend.ctc_batch_cost)
model.fit(train_generator, validation_data=val_generator, epochs=50)
```

## Evaluation
The model is evaluated using the Character Error Rate (CER) and Word Error Rate (WER). Use the testing set to compute these metrics and validate model performance.

## Usage
1. Prepare input images for prediction.
2. Load the trained model.
3. Pass images to the model for text detection.

Example usage:
```python
predictions = model.predict(test_images)
for pred in predictions:
    print(decode_prediction(pred))
```

## Results
The model achieves high accuracy on handwritten text datasets, with low CER and WER values. Results can vary based on the dataset quality and preprocessing techniques.

## Future Work
- Improve accuracy by experimenting with deeper architectures or additional preprocessing.
- Fine-tune the model on domain-specific datasets.
- Extend support for multilingual text detection.

## References
1. Connectionist Temporal Classification (CTC): Graves et al.
2. TensorFlow Documentation: https://www.tensorflow.org/
3. IAM Handwriting Database: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

## License
This project is licensed under the MIT License. Feel free to use and modify the code for educational and research purposes.

