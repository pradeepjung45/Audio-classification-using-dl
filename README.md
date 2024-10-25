                                                                UrbanSound8K Audio Classification
This project focuses on audio classification using the UrbanSound8K dataset. The objective is to classify audio samples into different sound classes such as dog bark, siren, car horn, and more, using a deep learning model built with TensorFlow and Keras.

                                                                 Table of Contents
Project Overview
Dataset
Preprocessing
Feature Extraction
Model Architecture
Training
Results
Installation
Usage
Contributing
License
Project Overview
This project applies machine learning techniques to classify urban sound samples. The UrbanSound8K dataset contains 8,732 labeled sound excerpts (≤ 4s) of urban sounds, categorized into 10 classes such as air conditioner, dog bark, children playing, and others.

                                                             Dataset
The dataset used is the UrbanSound8K dataset, which contains audio samples classified into 10 different urban sound categories. You can download the dataset and place it in the Downloads/UrbanSound8k directory.

                                                              Classes in the dataset:

Air Conditioner
Car Horn
Children Playing
Dog Bark
Drilling
Engine Idling
Gun Shot
Jackhammer
Siren
Street Music
Preprocessing
The audio samples are preprocessed using the librosa library:

Sample rate: 22,050 Hz
Audio features are extracted using Mel-Frequency Cepstral Coefficients (MFCCs), which are popular in speech and audio processing tasks.
Feature Extraction
For each audio sample, 40 MFCC features are extracted. These features are then averaged to obtain a fixed-length feature vector for every sample.

                                                        python
                                                        Copy code
def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features
Model Architecture
A neural network model is built using TensorFlow and Keras. The architecture consists of:

Input layer: 40 features (MFCCs)
Hidden layers: Dense layers with ReLU activation and dropout for regularization
Output layer: 10 neurons (softmax activation) for 10 sound classes
python
Copy code
model = Sequential()
model.add(Dense(100, input_dim=40, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
Training
The model is trained for 100 epochs using the Adam optimizer and categorical crossentropy loss.

                                                python
Copy code
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
num_epochs = 100
num_batch_size = 32
checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.keras', verbose=1, save_best_only=True)

start = datetime.now()
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
duration = datetime.now() - start
print("Training completed in time: ", duration)
Results
After training, the model achieves an accuracy of around 75% on the test dataset.

                                            python
                                            Copy code
test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy[1]}")
Installation
To set up the project locally, follow these steps:

Clone the repository:

                                             bash
                                             Copy code
git clone https://github.com/yourusername/urban-sound-classification.git
Install dependencies:

                                              bash
                                              Copy code
pip install -r requirements.txt
Download and extract the UrbanSound8K dataset to the Downloads/UrbanSound8k/ directory.

                                             Usage
Run the script to extract features and train the model:

                                            bash
                                            Copy code
python train_model.py
Once the model is trained, it will be saved in the saved_models/ directory. You can use the trained model to classify new audio files.

                                          Contributing
If you want to contribute to this project, feel free to submit a pull request or open an issue.

                                          License
This project is licensed under the MIT License. See the LICENSE file for more details.

