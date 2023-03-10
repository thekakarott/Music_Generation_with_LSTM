# Music_Generation_with_LSTM
#### This melody generation model is made using an LSTM neural network, trained on a dataset of MIDI files. The model takes in a seed melody as input and generates a new melody by predicting the next note based on the previous notes. The dataset contains classical piano pieces from various composers. The LSTM model architecture consists of an input layer, multiple LSTM layers, and an output layer. The generated melodies are evaluated using multiple metrics such as note distribution, pitch range, and rhythmic complexity. The results show that the model is capable of generating musically coherent and pleasing melodies similar to those found in the training dataset. The model can be used in various applications such as music composition and education. The use of MIDI files as the training dataset allows for the model to learn patterns and structures present in classical music. The preprocessing of the dataset involves converting MIDI files to sequences of note/rest symbols and mapping them to integer values. The model is trained using categorical cross-entropy loss function for multiple epochs with early stopping. The model uses a temperature parameter to control the randomness of the predictions. The evaluation shows that the generated melodies exhibit diverse characteristics and are musically pleasing to human listeners. The model provides a tool for musicians to explore and experiment with new musical ideas.

#### Install [Musescore](https://musescore.org/en/download) to your system for the representation of MIDI output
#### Run the Preprocessor.py file to configure all the settings and libraries included
#### Train the Model by running `Train.py` file Or Skip the step as it is already trained once
#### Finally run the `Melodygenerator.py` file to Generate .midi/.mid files as output 


Note: 
>For different outputs you have to update the seeds from the Datasets folder, which is generated after preprocessing.
