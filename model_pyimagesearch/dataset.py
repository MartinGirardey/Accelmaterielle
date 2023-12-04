# Import the necessary packages
from torch.utils.data import Dataset
import cv2
import numpy as np

class SegmentationBinaryDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms, maskThreshold=255/2):
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
	def __init__(self, imagePaths, maskPaths, transforms, maskThreshold=255/2):
		# Store the image and mask filepaths, and augmentation transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
		self.maskThreshold = maskThreshold

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