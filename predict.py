# Import the necessary packages
import random
from model_pyimagesearch import config
from torchmetrics.classification import Dice, JaccardIndex, PrecisionRecallCurve, ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import torch, torchvision
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from torchvision import transforms
from model_pyimagesearch.dataset import SegmentationBinaryDataset, SegmentationMulticlassDataset
from torch.nn import functional as F

def metrics(target, preds, threshold, ax=[]):
    dice = None ; jaccard = None ; pr_curve = None
    if config.TRAINING_TYPE == "BINARY":
        dice = Dice(threshold=threshold)
        jaccard = JaccardIndex(task="binary", threshold=threshold)
        pr_curve = PrecisionRecallCurve(task='binary')
        confusion_matrix = ConfusionMatrix(task='binary', threshold=threshold, normalize='true')
        conf_matrix = confusion_matrix(preds, target).numpy()
    # elif config.TRAINING_TYPE == "MULTICLASS":
    #     dice = Dice(num_classes=config.NB_CLASSES)
    #     jaccard = JaccardIndex(task="multiclass", num_classes=config.NB_CLASSES)

    print("\n###################  EVALUATION  ###################")
    print("   Dice = {:.4f}".format(dice(preds, target)))
    print("   Jaccard = {:.4f}".format(jaccard(preds, target)))
    print("####################################################\n")

    isAx = True
    fig = None
    if len(ax) == 0:
        fig, ax = plt.subplots(1, 2, figsize=(10,4))
        isAx = False

    ax[0].matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax[0].text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='medium')
    ax[0].set_xlabel('Predictions', fontsize="medium")
    ax[0].set_ylabel('Actuals', fontsize="medium")
    ax[0].set_title('Confusion Matrix', fontsize="large")

    precision, recall, thresholds = pr_curve(preds, target)
    pr_curve.plot(score=True, ax=ax[1])

    if not isAx:
        fig.tight_layout()
        fig.show()

    return precision, recall, thresholds

def plot_one_mask(image, mask, pred, ax=[]):
    # Initialize our figure
    isAx = True
    fig = None
    if len(ax) == 0:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
        isAx = False

    # Plot the original image, its mask, and the predicted mask
    ax[0].imshow(image)
    ax[1].imshow(mask, cmap='gray', vmin=0, vmax=255)
    ax[2].imshow(pred, cmap='gray', vmin=0, vmax=255)

    # Set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Ground Truth mask")
    ax[2].set_title("Predicted mask")

    if not isAx:
        fig.tight_layout()
        fig.show()

def plot_multi_masks(image, masks, preds):
    # Initialize our figure
    figure, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))

    # Plot the original image, its mask, and the predicted mask
    for i in range(5):
        ax[0][i].imshow(masks[i], cmap='gray', vmin=0, vmax=1)
    for i in range(5):
        ax[1][i].imshow(preds[i], cmap='gray', vmin=0, vmax=1)

    # Set the layout of the figure and display it
    figure.tight_layout()
    figure.show()

def make_predictions(model, data, threshold=config.PRED_THRESHOLD, plot=True, computeMetrics=True):
    # Set model to evaluation mode
    model.eval()

    # Turn off gradient tracking
    with torch.no_grad():
        (image, multiMask) = data

        mask = None
        if config.TRAINING_TYPE == "BINARY":
            mask = multiMask.unsqueeze(0)
        # elif config.TRAINING_TYPE == "MULTICLASS":
        #     mask = np.ones(multiMask.size()[1:]) * (-1)
        #     for i in range(multiMask.size(0)):
        #         mask[multiMask[i,:,:] == 1.] = config.PLOT_MASK_VALUES[i]
        #     fMask = torch.zeros(multiMask.size()[1:])
        #     fMask[mask == -1] = 1.
        #     multiMask = torch.cat((multiMask, fMask.unsqueeze(0)), dim=0)
        #     mask[mask == -1] = config.PLOT_MASK_VALUES[-1]

        # Make the prediction, pass the results through the sigmoid function, and convert the result to a NumPy array
        origPredMask = model(image.unsqueeze(0))
        if config.TRAINING_TYPE == "BINARY":
            origPredMask = F.sigmoid(origPredMask)
        # elif config.TRAINING_TYPE == "MULTICLASS":
        #     origPredMask = F.softmax(origPredMask, dim=0)

        predMask = origPredMask.detach().clone()
        # predMask[predMask < threshold] = 0.
        pred = torch.zeros((predMask.shape[1:]))
        if config.TRAINING_TYPE == "BINARY":
            # Filter out the weak predictions and convert them to integers
            print("Max : {} ; Min : {}".format(torch.max(predMask), torch.min(predMask)))
            pred = (predMask > threshold) # For binary segmentation
        # elif config.TRAINING_TYPE == "MULTICLASS":
        #     # Generating the predicted mask (for each pixel, the mask with the highest probability)
        #     pred[:,:] = torch.argmax(predMask, dim=0)
        #     pred[torch.sum(predMask, dim=0) == 0.] = -1
        #     for i in range(config.NB_CLASSES-1):
        #         # print("{} : {} ; {}".format(config.DICT_MASKS[i], torch.max(predMask[i,:,:]), torch.min(predMask[i,:,:])))
        #         pred[pred == i] = config.PLOT_MASK_VALUES[i]
        #     pred[pred == -1] = config.PLOT_MASK_VALUES[-1]
        #
        #     # Regenerating the masks
        #     multiPred = torch.zeros((predMask.size(0)+1, predMask.size(1), predMask.size(2)))
        #     for i in range(config.NB_CLASSES):
        #         multiPred[i, pred == config.PLOT_MASK_VALUES[i]] = 1.

        mask = mask.type(torch.IntTensor)
        pred = pred.type(torch.IntTensor)
        # multiMask = multiMask.type(torch.IntTensor)
        # multiPred = multiPred.type(torch.IntTensor)

        # Compute the metrics
        # if config.TRAINING_TYPE == "MULTICLASS":
        #     plot_multi_masks(image.swapdims(0,1).swapdims(1,2), multiMask, multiPred)
        #     metrics(multiPred.unsqueeze(0), multiMask.unsqueeze(0))

        # Plots and metrics computing
        if plot and computeMetrics:
            fig, ax = plt.subplots(2, 3, figsize=(10,8))
            plot_one_mask(image.swapdims(0,1).swapdims(1,2), mask.squeeze()*255, pred.squeeze()*255, ax[0])
            metrics(mask.squeeze(0), origPredMask.squeeze(0), threshold, ax[1])
            fig.show()
        elif plot and not computeMetrics:
            fig, ax = plt.subplots(1, 3, figsize=(10,4))
            plot_one_mask(image.swapdims(0,1).swapdims(1,2), mask.squeeze()*255, pred.squeeze()*255, ax)
            fig.show()
        elif not plot and computeMetrics:
            fig, ax = plt.subplots(1, 3, figsize=(10, 4))
            metrics(mask.squeeze(0), origPredMask.squeeze(0), threshold, ax)
            fig.show()

        # Compute and show the metrics : Dice and Jaccard index, Precision-Recall curve, Confusion matrix
        # if computeMetrics:
        #     if config.TRAINING_TYPE == "BINARY":
        #         metrics(mask.squeeze(0), origPredMask.squeeze(0), threshold)

        # if plot:
        #     plot_one_mask(image.swapdims(0, 1).swapdims(1, 2), mask.squeeze() * 255, pred.squeeze() * 255)

if __name__ == '__main__':
    nbExamples = 1 # Number of examples to analyze
    randomSeed = 1 # Seed for the random selection of those examples
    plot = True # Plot the image and the masks
    computeMetrics = True # Compute and plot the metrics (Precision - Recall curve & Confusion Matrix)

    # Load the image and mask filepaths in a sorted manner
    imagePaths = None ; maskPaths = None
    if config.TRAINING_TYPE == 'BINARY':
        imagePaths = sorted(list(paths.list_images(config.IMAGE_BINARY_DATASET_PATH)))
        maskPaths = sorted(list(paths.list_images(config.MASK_BINARY_DATASET_PATH)))
    elif config.TRAINING_TYPE == 'MULTICLASS':
        imagePaths = sorted(list(paths.list_images(config.IMAGE_MULTICLASS_DATASET_PATH)))
        maskPaths = sorted(list(paths.list_images(config.MASK_MULTICLASS_DATASET_PATH)))

    # Partition the data into training and testing sets
    split = train_test_split(imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=config.SPLIT_SEED)
    (_, testImages) = split[:2]
    (_, testMasks) = split[2:]

    # Select the data to show according to the given seed
    np.random.seed(randomSeed)
    testImages = np.random.choice(testImages, nbExamples)
    np.random.seed(randomSeed)
    testMasks = np.random.choice(testMasks, nbExamples)

    # Define transformations
    transforms = transforms.Compose([transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH),
                                                       interpolation=transforms.InterpolationMode.NEAREST)])
    # Create the train and test datasets
    dataset = None
    if config.TRAINING_TYPE == 'BINARY':
        dataset = SegmentationBinaryDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms)
    elif config.TRAINING_TYPE == 'MULTICLASS':
        dataset = SegmentationMulticlassDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms)

    # Load our model from disk and flash it to the current device
    print("[INFO] Load up model...")
    if config.TRAINING_TYPE == 'BINARY':
        unet = torch.load(config.BINARY_MODEL_PATH).to(config.DEVICE)
    elif config.TRAINING_TYPE == 'MULTICLASS':
        unet = torch.load(config.MULTICLASS_MODEL_PATH).to(config.DEVICE)

    # Showing the
    for i in range(len(testImages)):
        make_predictions(unet, dataset[i], threshold=0.45, plot=plot, computeMetrics=computeMetrics)

    plt.show()
