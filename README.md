# Music_Generation_with_LSTM
#### Main objective of this is to propose an algorithm which can be used to generate musical notes  using  Recurrent  Neural  Networks  (RNN)
principally Long  Short-Term  Memory  (LSTM)  networks.  A  model  is designed  to  execute  this  algorithm  where  data  is  represented with the help of musical instrument digital interface (MIDI) file format  for  easier  access  and  better  understanding. Preprocessing of data before feeding it into the model, revealing methods to read, process and prepare MIDI files for input are also discussed. 
The model used in this is used to learn the sequences  of  polyphonic  musical  notes  over  a  single-layered LSTM network. The model must have the potential to recall past details  of  a  musical sequence  and  its  structure  for  better learning. 
Description  of  layered  architecture  used  in LSTM model  and  its  intertwining  connections  to  develop  a  neural network is presented in this work.This paper imparts a peek view of distributions of weights and biases in every layer of the model along with a precise representation of losses and accuracy at  each  step  and  batches.  
When  the  model  was  thoroughly analyzed, it produced stellar results in composing new melodies.

#### Install Musescore to your system for the representation of MIDI output
#### Run the Preprocessor.py file to configure all the settings and libraries included
#### Train the Model by running Train.py file Or Skip the step as it is already trained once
#### Finally run the Melodygenerator.py file to Generate .midi/.mid files as output (for different outputs you have to update the seeds from the Datasets folder, which is generated after preprocessing)
