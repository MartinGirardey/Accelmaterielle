# Import the necessary packages
from torch.utils.data import Dataset
import cv2
import numpy as np
import model_pyimagesearch.config as config
import torch

class SegmentationBinaryDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		# Store the image and mask filepaths, and augmentation transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms

	def __len__(self):
		# Return the number of total samples contained in the dataset
		return len(self.imagePaths)

	def __getitem__(self, idx):
		# Grab the image path from the current index
		imagePath = self.imagePaths[idx]

		# Load the image from disk, swap its channels from BGR to RGB, and read the associated mask from disk in
		# grayscale mode
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.maskPaths[idx], 0)

		# Check to see if we are applying any transformations
		if self.transforms is not None:
			# Apply the transformations to both the image and its mask
			image = self.transforms(image)
			mask = self.transforms(mask)

		# Return a tuple of the image and its mask
		return (image, mask)

class SegmentationMulticlassDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms, classValues=config.MASK_VALUES):
		# Store the image and mask filepaths, and augmentation transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.classValues = classValues
		self.transforms = transforms

	def __len__(self):
		# Return the number of total samples contained in the dataset
		return len(self.imagePaths)

	def __getitem__(self, idx):
		# Grab the image path from the current index
		imagePath = self.imagePaths[idx]

		# Load the image from disk, swap its channels from BGR to RGB, and read the associated mask from disk in
		# grayscale mode
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mono_mask = cv2.imread(self.maskPaths[idx], 0)
		mask = [np.zeros((mono_mask.shape[0], mono_mask.shape[1])) for i in range(len(self.classValues))]
		for i in range(len(self.classValues)):
			mask[i] = (mono_mask == self.classValues[i]) * self.classValues[i]

		# Check to see if we are applying any transformations
		if self.transforms is not None:
			# Apply the transformations to both the image and its mask
			image = self.transforms(image)
			for i in range(len(self.classValues)):
				mask[i] = self.transforms(mask[i])

		# Return a tuple of the image and its mask
		return (image, mask)