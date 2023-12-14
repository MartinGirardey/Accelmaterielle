# Import the necessary packages
import random
from model_pyimagesearch import config
from torchmetrics.classification import Accuracy, Dice, JaccardIndex, AveragePrecision, F1Score, ConfusionMatrix, PrecisionRecallCurve, ROC
import matplotlib.pyplot as plt
import numpy as np
import torch, torchvision
import os
from torchvision import transforms
from model_pyimagesearch.dataset import SegmentationBinaryDataset, SegmentationMulticlassDataset
from torch.nn import functional as F
import glob

def metrics(target, preds, threshold, ax=[], plot=True):
    dice = None ; jaccard = None ; pr_curve = None
    if config.TRAINING_TYPE == "BINARY":
        accuracy = Accuracy(task='binary', threshold=threshold)
        dice = Dice(threshold=threshold)
        jaccard = JaccardIndex(task="binary", threshold=threshold)
        ap = AveragePrecision(task='binary', average=None)
        f1 = F1Score(task='binary', threshold=threshold)
        roc_curve = ROC(task='binary')
        confusion_matrix = ConfusionMatrix(task='binary', threshold=threshold, normalize='true')
        conf_matrix = confusion_matrix(preds, target).numpy()
    # elif config.TRAINING_TYPE == "MULTICLASS":
    #     dice = Dice(num_classes=config.NB_CLASSES)
    #     jaccard = JaccardIndex(task="multiclass", num_classes=config.NB_CLASSES)

    isAx = True
    fig = None
    if len(ax) == 0:
        fig, ax = plt.subplots(1, 2, figsize=(10,4))
        isAx = False

    if plot:
        ax[0].matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax[0].text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='medium')
        ax[0].set_xlabel('Predictions', fontsize="medium")
        ax[0].set_ylabel('Actuals', fontsize="medium")
        ax[0].set_title('Confusion Matrix', fontsize="large")

        roc_curve.update(preds, target)
        roc_curve.plot(score=True, ax=ax[2])

        if not isAx:
            fig.tight_layout()
            fig.show()

    return accuracy(preds, target).item(), dice(preds, target).item(), jaccard(preds, target).item(), ap(preds, target).item(), f1(preds, target).item()

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
        ax[0][i].imshow(masks[i,:,:], cmap='gray', vmin=0, vmax=1)
        ax[0][i].set_title("GT : {}".format(config.DICT_MASKS[i]))
    for i in range(5):
        ax[1][i].imshow(preds[i,:,:], cmap='gray', vmin=0, vmax=1)
        ax[1][i].set_title("Predicted : {}".format(config.DICT_MASKS[i]))

    # Set the layout of the figure and display it
    figure.tight_layout()
    figure.show()

def find_best_binary_threshold(target, preds, start=0., stop=1., n_thresholds=50, index='dice'):
    thresholds = np.linspace(start, stop, n_thresholds)
    best_threshold = 0
    best_score = -np.inf
    for i in range(n_thresholds):
        c_index = None
        if index == 'dice':
            c_index = Dice(threshold=thresholds[i])
        elif index == 'jaccard':
            c_index = JaccardIndex(task='binary', threshold=thresholds[i])
        elif index == 'f1':
            c_index = F1Score(task='binary', threshold=thresholds[i])
        else:
            c_index = Accuracy(task='binary', threshold=thresholds[i])

        score = c_index(preds, target)
        if score > best_score:
            best_score = score
            best_threshold = thresholds[i]

    return best_threshold

def make_predictions(model, data, threshold=config.PRED_THRESHOLD, plot=True, computeMetrics=True):
    # Set model to evaluation mode
    model.eval()

    # Turn off gradient tracking
    with torch.no_grad():
        (image, multiMask) = data

        # Group every masks in a single image and add the mask we didn't consider during training (landing-zones)
        mask = None
        if config.TRAINING_TYPE == "BINARY":
            multiMask = multiMask.unsqueeze(0)
            mask = multiMask * 255
        elif config.TRAINING_TYPE == "MULTICLASS":
            mask = torch.ones(multiMask.size()[1:]) * (-1)
            for i in range(multiMask.size(0)):
                mask[multiMask[i,:,:] == 1.] = config.PLOT_MASK_VALUES[i]
            fMask = torch.zeros(multiMask.size()[1:])
            fMask[mask == -1] = 1.
            multiMask = torch.cat((multiMask, fMask.unsqueeze(0)), dim=0)
            mask[mask == -1] = config.PLOT_MASK_VALUES[-1]

        # Make the prediction, pass the results through the sigmoid function, and convert the result to a NumPy array
        multiPred = model(image.unsqueeze(0))
        if config.TRAINING_TYPE == "BINARY":
            multiPred = F.sigmoid(multiPred)
        elif config.TRAINING_TYPE == "MULTICLASS":
            # multiPred = F.softmax(multiPred, dim=0)
            multiPred = F.sigmoid(multiPred)

        pred = torch.zeros((multiPred.shape[2:]))
        predMask = None
        if config.TRAINING_TYPE == "BINARY":
            # Filter out the weak predictions and convert them to integers
            # print("Max : {} ; Min : {}".format(torch.max(multiPred), torch.min(multiPred)))
            pred[:,:] = (multiPred[0,0,:,:] > threshold) * 255 # For binary segmentation
        elif config.TRAINING_TYPE == "MULTICLASS":
            predMask = multiPred.detach().clone()[0,:,:,:]
            predMask[predMask < threshold] = 0.
            # Generating the predicted mask (for each pixel, the mask with the highest probability)
            pred[:,:] = torch.argmax(predMask, dim=0)
            pred[torch.sum(predMask, dim=0) == 0.] = -1
            for i in range(config.NB_CLASSES-1):
                # print("{} : {} ; {}".format(config.DICT_MASKS[i], torch.max(predMask[i,:,:]), torch.min(predMask[i,:,:])))
                pred[pred == i] = config.PLOT_MASK_VALUES[i]
            pred[pred == -1] = config.PLOT_MASK_VALUES[-1]

            # Regenerating the masks (in 0-1 values)
            predMask = torch.zeros((config.NB_CLASSES, predMask.size(1), predMask.size(2)))
            for i in range(config.NB_CLASSES):
                predMask[i, pred == config.PLOT_MASK_VALUES[i]] = 1.

        mask = mask.type(torch.IntTensor)
        multiMask = multiMask.type(torch.IntTensor)
        pred = pred.type(torch.IntTensor)

        # Plots and metrics computing
        if plot and computeMetrics:
            fig, ax = plt.subplots(2, 3, figsize=(10,8))
            plot_one_mask(image.swapdims(0,1).swapdims(1,2), mask.squeeze(), pred.squeeze(), ax[0])
            if config.TRAINING_TYPE == 'MULTICLASS':
                plot_multi_masks(image.swapdims(0,1).swapdims(1,2), multiMask, predMask)
            accuracy, dice, jaccard, ap, f1 = metrics(multiMask, multiPred, threshold, ax[1])
            fig.show()
            return accuracy, dice, jaccard, ap, f1

        elif plot and not computeMetrics:
            fig, ax = plt.subplots(1, 3, figsize=(10,4))
            plot_one_mask(image.swapdims(0,1).swapdims(1,2), mask.squeeze(), pred.squeeze(), ax)
            if config.TRAINING_TYPE == 'MULTICLASS':
                plot_multi_masks(image.swapdims(0,1).swapdims(1,2), multiMask, predMask)
            fig.show()

        elif not plot and computeMetrics:
            accuracy, dice, jaccard, ap, f1 = metrics(multiMask, multiPred, threshold, plot=False)
            return accuracy, dice, jaccard, ap, f1

        return multiMask, multiPred

if __name__ == '__main__':
    nbExamples = 5 # Number of examples to analyze among the test set
    randomSeed = 42 # Seed for the random selection of those examples
    # threshold = 0.47 # Best for accuracy
    # threshold = 0.18 # Best for Dice, Jaccard and F1
    threshold = 0.4
    plot = True # Plot the image and the masks
    computeMetrics = True # Compute and plot the metrics (Precision - Recall curve & Confusion Matrix)
    findBestThreshold = True
    # Start, stop, number of values to consider and index to use (dice, jaccard, f1 or accuracy, accuracy by default)
    findBestThresholdArgs = (0., 1., 50, 'f1')

    # Load the image and mask filepaths in a sorted manner
    imagePaths = None ; maskPaths = None
    if config.TRAINING_TYPE == 'BINARY':
        imagePaths = sorted(list(glob.glob(config.IMAGE_BINARY_DATASET_PATH + "/*.png")))
        maskPaths = sorted(list(glob.glob(config.MASK_BINARY_DATASET_PATH + "/*.png")))
    elif config.TRAINING_TYPE == 'MULTICLASS':
        imagePaths = sorted(list(glob.glob(config.IMAGE_MULTICLASS_DATASET_PATH + "/*.png")))
        maskPaths = sorted(list(glob.glob(config.MASK_MULTICLASS_DATASET_PATH + "/*.png")))

    # Define transformations
    transforms = transforms.Compose([transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH),
                                                       interpolation=transforms.InterpolationMode.NEAREST)])

    # Loading the dataset
    dataset = None
    if config.TRAINING_TYPE == 'BINARY':
        dataset = SegmentationBinaryDataset(imagePaths=imagePaths, maskPaths=maskPaths, transforms=transforms)
    elif config.TRAINING_TYPE == 'MULTICLASS':
        dataset = SegmentationMulticlassDataset(imagePaths=imagePaths, maskPaths=maskPaths, transforms=transforms)

    # Partition of the data into training and testing sets
    trainSize = int((1-config.TEST_SPLIT)*len(dataset))
    _, testSet = torch.utils.data.random_split(dataset, [trainSize, len(dataset)-trainSize],
                                               generator=torch.Generator().manual_seed(config.SPLIT_SEED))

    # Picking few data to plot and analyze
    np.random.seed(randomSeed)
    testIndex = np.random.choice(range(0,len(testSet)), nbExamples)

    # Load our model from disk and flash it to the current device
    print("[INFO] Load up model...")
    if config.TRAINING_TYPE == 'BINARY':
        unet = torch.load(config.BINARY_MODEL_PATH, map_location=torch.device(config.DEVICE)).to(config.DEVICE)
    elif config.TRAINING_TYPE == 'MULTICLASS':
        unet = torch.load(config.MULTICLASS_MODEL_PATH, map_location=torch.device(config.DEVICE)).to(config.DEVICE)

    # Looking for the threshold.s which maximize the results according to a score (DICE, Jaccard, F1 or Accuracy)
    best_threshold = config.PRED_THRESHOLD
    if findBestThreshold:
        if config.TRAINING_TYPE == 'BINARY':
            thresholds = []
            for ind in testIndex:
                multiMask, multiPred = make_predictions(unet, testSet[ind], plot=False, computeMetrics=False)
                thresholds.append(find_best_binary_threshold(multiMask.squeeze(0), multiPred.squeeze(0),
                                                       start=findBestThresholdArgs[0], stop=findBestThresholdArgs[1],
                                                       n_thresholds=findBestThresholdArgs[2],
                                                       index=findBestThresholdArgs[3]))
            best_threshold = sum(thresholds)/len(thresholds) # Generally around 0.35 for DICE, Jaccard and F1, around 0.5 for Accuracy
            print("Best thresholds      = {}".format(thresholds))
            print("Best thresholds mean = {}".format(best_threshold))

    else:
        # Ploting and analyzing the few data one by one
        accuracies, dices, jaccards, aps, f1s = [], [], [], [], []
        # testIndex = [0, 0]
        # thresholds = [0.34, 0.5]
        # i = 0
        for ind in testIndex:
            if computeMetrics:
                accuracy, dice, jaccard, ap, f1 = make_predictions(unet, testSet[ind], threshold=threshold, plot=plot, computeMetrics=computeMetrics)
                accuracies.append(accuracy)
                dices.append(dice)
                jaccards.append(jaccard)
                aps.append(ap)
                f1s.append(f1)
            elif plot:
                make_predictions(unet, testSet[ind], threshold=threshold, plot=plot, computeMetrics=computeMetrics)
            # i += 1

        if len(dices) > 0:
            print("###################### EVALUATION ######################")
            print("Accuracies               = {}".format(accuracies))
            print("Mean accuracy            = {}".format(sum(accuracies)/len(accuracies)))
            print("Dice indexs              = {}".format(dices))
            print("Dice mean                = {}".format(sum(dices)/len(dices)))
            print("Jaccard indexs           = {}".format(jaccards))
            print("Jaccard mean             = {}".format(sum(jaccards)/len(jaccards)))
            print("Average-Precision indexs = {}".format(aps))
            print("Average-Precision mean   = {}".format(sum(aps)/len(aps)))
            print("F1 scores                = {}".format(f1s))
            print("F1 mean                  = {}".format(sum(f1s)/len(f1s)))
            print("########################################################")

        if plot:
            plt.show()
