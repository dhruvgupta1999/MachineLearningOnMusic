# Explo_Musical_chord_detection
This is my project code on detecting musical chords played on guitar using a fully-connected neural network

# Objective:
To detect what chords are to be played to a song 



# What we have accomplished so far:

1) Created our own dataset of over 160 chord samples and 30 musical note samples which comprises of movable chords,open chords and barre chords on multiple guitars

2) worked with the short-term fourier transform function to produce chromagrams
	What is a chromagram?
	Chromagram is a tool that aids in visually analysing sound as well as running processing algorithm on it.
	It can be thought of as a 2d array with rows as frequncy bins and columns as time slices
	Each cell contains the magnitude of the waveform at that coordinate.

3)implemented a pitch classifier  (taking A4 at 440 hz , used the geometric mean as boundaries)

	What is a pitch classifier?
	classifies audio of different frequencies into their respective musical note class 

4) used music theory concepts and python to detect an approximate scale and likely notes to be played to a given song:
		*found most occuring notes and selected best fit onto a musical mode to get the set 
		of notes and chords that are most likely to fit onto the audio sample
		*best fit was found on the major scale formula 

5) noise-cleaning for the above audio-features was implemented 
		*removed sounds with magnitude below half the rms magnitude of the audio sample											
		
		*Filtering by nearest-neighbors.
		Each data point (e.g, spectrogram column) is replaced by aggregating its nearest neighbors in feature space.



6) Used feed-forward Neural Network implemented on tensorflow and theano to train an the dataset and make predictions 
	
	The highest accuracy recorded about 97% and a median accuracy of about 87%

7) Have implemented seperation of harmonic and percussive elements of music:
	*The method is based on the assumption that harmonic components exhibit horizontal lines on the spectrogram while the percussive sounds are evident as vertical lines. Used Non-Linear filters applied to the spectrogram in order to filter out these components.

8) implemented estimate of the tempo using confidence interval measure.
In this method, periodicity analysis is carried out by analysing filterbank energies on the percussive component of the audio sample obtained above.

Description of our neural net:

	Input: uses the output of pitch classifier as its input, thus requiring a modest 12 features

	Hidden Layers: two hidden layers having same number of nodes

	Output : 12 possible classes (All major chords)

	regularisation: dropout layer on the first hidden layer and l2 regularisation on the second hidden layer 

	accuracy: were able to obtain 100% accuracy on train set at approximately 300 epochs
			  were able to obtain >97% accuracy (at 28 neurons per hidden layer) at approximately 1500 epochs 



# Description of files:

1) all.csv and Train1.csv contain train data of all sorts of chords:
	0:N ,1:C_maj , .... 13:C_min

2)final_preprocessing.py worked on the beatle chord dataset to extract feautures and write to a .csv file

3)final_explo_ml_model.py extracts features from the .csv created above and uses a keras model to detect the corresponding chord.
