Requirements :


General organisation :

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

Description of the files(open them to get more informations):
  - UNet.py : UNet neural network as inspired by the original paper [], we found this PyTorch implementation on [] and adapted it a bit to our specific situation. It is called in train.py and predict.py.

  - UNet_own.py : A first try to implement by ourselves the UNet model. It worked but was slower and less adaptable than the one we found on Internet, so we chose to abandon it.

  - config.py : Configuration file where we can choose what kind of problem we're facing (binary or multiclass), the different training parameters (number of epochs, size of the batches, number and nature of layers for the UNet model, learning rate...) and the different paths (datasets, output...).

  - dataset.py : Here are defined the two kind of dataset we use in this project (binary and multiclass datasets) as Pytorch map-style datasets. Those two datasets are used to load data in train.py and predict.py.

  - predict.py : 

  - train.py : Python algorithm to train our UNet neural network in 

  - xxxxx_plot.png : Curve of training and testing loss made during the training of the model (binary or multiclass).

  - xxxxx_unet.pth : PyTorch save file where is saved the trained UNet model (binary or multiclass), this files is loaded in predict.py (and in train.py if we want to continue the training)