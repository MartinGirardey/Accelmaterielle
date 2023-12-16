##############################
#        Requirements        #
##############################

torch
torchvision
torchmetrics
matplotlib
numpy

##############################
#    General organisation    #
##############################

  MaterialAcceleration_GIRARDEY_LIORET
     |_ model_own
	 |_ UNet_own.py
     |_ model_pyimagesearch
         |_ config.py
         |_ dataset.py
         |_ UNet.py
     |_ output
         |_ binary_plot.pth
	 |_ binary_unet.png
	 |_ multiclass_plot.png
	 |_ multiclass_unet.pth
     |_ predict.py
     |_ train.py
     |_ README.txt

##############################
#  Description of the files  #
##############################

  - UNet.py : UNet neural network as inspired by the original paper [], we found this PyTorch implementation on [] and adapted it a bit to our specific situation. It is called in train.py and predict.py.

  - UNet_own.py : A first try to implement by ourselves the UNet model. It worked but was slower and less adaptable than the one we found on Internet, so we chose to abandon it.

  - config.py : Configuration file where we can choose what kind of problem we're facing (binary or multiclass), the different training parameters (number of epochs, size of the batches, number and nature of layers for the UNet model, learning rate...) and the different paths (datasets, output...).

  - dataset.py : Here are defined the two kind of dataset we use in this project (binary and multiclass datasets) as Pytorch map-style datasets. Those two datasets are used to load data in train.py and predict.py.

  - predict.py : Algorithm to make a complete benchmark on a given number of examples. We implemented different function to load the data, format it (in order to get binary mask.s thanks to threshold.s which can be modified), plot it and different metrics (ROC curve, accuracy, DICE and Jaccard index, F1 score, Average Precision, Confusion matrix...). We made a function to optimize the threshold for binary segmentation according to a given index (DICE, accuracy, Jaccard or F1). See the following "How to evaluate a model ?" section for more informations.

  - train.py : Python algorithm to train our UNet neural network with respect to the parameters set in the configuration file config.py (number of epochs, batch size, initial learning rate...). Normaly you don't have to modify anything in this file to make a training. See the following "How to train the network ?" section for more informations.

  - xxxxx_plot.png : Curve of training and testing loss made during the training of the model (binary or multiclass).

  - xxxxx_unet.pth : PyTorch save file where is saved the trained UNet model (binary or multiclass), this files is loaded in predict.py (and in train.py if we want to continue the training)

##############################
# How to train the network ? #
##############################

1. Configure the training parameters within the config.py file :
	- TRAINING_TYPE = "BINARY" or "MULTICLASS"   
	- CONTINUE_TRAINING = True or False        : Shall we continue a training or start a new one
	- INTERPOLATE = True or False              : Do the model has to interpolate the output to the input size ?
	- ENCODER_CHANNELS & DECODER_CHANNELS      : Define the architecture of the UNet network with the number of convolution layers
	- INPUT_IMAGE_WIDTH & INPUT_IMAGE_HEIGHT   : For optimization, we often consider smaller images (original ones resized), here you can specify the desired size in input of the model
	- OUTPUT_IMAGE_WIDTH & OUTPUT_IMAGE_HEIGHT : Because we often don't have enough memory available (depending on batch size), we can reduce it by reducing the size of the images we consider
	- INIT_LR                                  : Initial learning rate
	- NUM_EPOCHS                               : Number of epochs
	- BATCH_SIZE                               : Size of a batch
	- TEST_SPLIT = between 0 and 1             : Amount of data reserved for testing purpose
	- SPLIT_SEED                               : Seed for the random splitting of the data, allows to start in the same conditions from one test to another if desired
	- xxxxxx_DATASET_PATH                      : Path to the datasets (binary and multiclass)
	- BASE_OUTPUT                              : Base path to create the output files

2. Launch train.py script and wait for it to finish (a message is written every time an epoch is completed)

3. The trained model is saved every time the test loss is improved, it is saved in the BASE_OUTPUT folder, along with an image of a graph showing the evolution of the train and test loss during the training (if you successfully completed all the epochs of the training)

Rk : In the multiclass case (5 classes), only 4 classes are trained. When predicting, a threshold on the probabilities obtained allows to generate the fifth class (the most common one).

##############################
#  How to evaluate a model ? #
##############################

1. Open the predict.py file and configure the prediction (variables to be modified just after the "if __name__ == '__main__':" line) :
	- nbExamples                                                       : Number of examples to consider
	- randomSeed                                                       : Seed for the random picking of the examples among the test set
	- selection                                                        : List of selected indices to print among the nbExamples
	- threshold                                                        : Threshold to be considered to separate the two classes in the binary case, and to determine the fifth class in the multiclass case
	- plot = True or False                                             : Plot things or not (image, masks, predicted masks, metrics...)
	- computeMetrics = True or False                                   : Compute the metrics or not (Accuracy, Dice, Jaccard, Average Precision, F1, ROC curve, Confusion matrix...) 
	- findBestThreshold = True or False                                : For binary segmentation only, allows to find the best threshold (according to a given index)
	- findBestThresholdArgs = (start, stop, nb thresholds, index used) : Parameters for the research of the best threshold
		- [start, stop] : interval considered
		- nb thresholds : number of thresholds considered among the interval
		- index used : 'dice', 'jaccard', 'f1' or 'accuracy' to set the index we want to take for optimizing the threshold ('dice', 'jaccard' and 'f1' gives approximately the same results, 'accuracy' best threshold is generally quite higher)

2. Launch the predict.py script, depending on the parameters you entered, you should obtain :
	- (if computeMetrics=True) metric values for each example and mean value
	- (if findBestThreshold) values of best threshold for each considered example, and mean best threshold (which can then be sent
	- (if plot=True) nbExamples windows with different plots :
		- image, ground truth mask and predicted mask (single dimension mask which is built from the 4 masks obtained at the output of the network)
		- (if computeMetrics=True) confusion matrix, error map and ROC curve