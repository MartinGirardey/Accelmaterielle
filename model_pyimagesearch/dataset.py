# Import the necessary packages
from torch.utils.data import Dataset
import cv2
import numpy as np
import model_pyimagesearch.config as config
import torch, torchvision

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
		# Load the image from disk, swap its channels from BGR to RGB, and read the associated mask from disk in
		# grayscale mode
		image = torchvision.io.read_image(self.imagePaths[idx]) / 255.
		mask = torchvision.io.read_image(self.maskPaths[idx], torchvision.io.ImageReadMode.GRAY)
		mask = mask - torch.min(mask)
		mask = mask / torch.max(mask)

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
		image = torchvision.io.read_image(imagePath) / 255.
		mono_mask = torchvision.io.read_image(self.maskPaths[idx], torchvision.io.ImageReadMode.GRAY)
		mask = [torch.zeros((mono_mask.shape[0], mono_mask.shape[1])) for i in range(len(self.classValues)-1)]

		for i in range(len(mask)):
			mask[i] = (mono_mask == self.classValues[i]).type(torch.FloatTensor)

		# Check to see if we are applying any transformations
		if self.transforms is not None:
			# Apply the transformations to both the image and its mask
			image = self.transforms(image)
			for i in range(len(mask)):
				mask[i] = self.transforms(mask[i])
			mask = torch.cat(mask, dim=0)

		# Return a tuple of the image and its mask
		return (image, mask)