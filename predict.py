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
from sklearn.metrics import confusion_matrix

def metrics(target, preds, threshold, targetMono=None, maskMono=None, predMask=None, ax=[], plot=True):
    dice = None ; jaccard = None ; pr_curve = None
    if config.TRAINING_TYPE == "BINARY":
        accuracy = Accuracy(task='binary', threshold=threshold)
        dice = Dice(threshold=threshold)
        jaccard = JaccardIndex(task="binary", threshold=threshold)
        ap = AveragePrecision(task='binary', average=None)
        f1 = F1Score(task='binary', threshold=threshold)
        roc_curve = ROC(task='binary')
        conf_matrix = ConfusionMatrix(task='binary', threshold=threshold, normalize='true')
        c_matrix = conf_matrix(preds, target).numpy()

    elif config.TRAINING_TYPE == "MULTICLASS":
        accuracy = Accuracy(task='multiclass', num_classes=config.NB_CLASSES)
        jaccard = JaccardIndex(task="multiclass", num_classes=config.NB_CLASSES)
        ap = AveragePrecision(task='multiclass', num_classes=config.NB_CLASSES, average=None)
        f1 = F1Score(task='multiclass', num_classes=config.NB_CLASSES)
        roc_curve = ROC(task='multiclass', num_classes=config.NB_CLASSES-1)
        # confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=config.NB_CLASSES,
        #                                    normalize='true')
        # conf_matrix = confusion_matrix(predMask, target).numpy()
        # c_matrix = confusion_matrix(targetMono.squeeze(), maskMono.squeeze(), normalize='true')

    if config.TRAINING_TYPE != 'MULTICLASS':
        isAx = True
        fig = None
        if len(ax) == 0:
            fig, ax = plt.subplots(1, 3, figsize=(10,4))
            isAx = False

        if plot:
            ax[0].matshow(c_matrix, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(c_matrix.shape[0]):
                for j in range(c_matrix.shape[1]):
                    ax[0].text(x=j, y=i, s=round(c_matrix[i, j], 4), va='center', ha='center', size='medium')
            ax[0].set_xlabel('Predictions', fontsize="medium")
            ax[0].set_ylabel('Actuals', fontsize="medium")
            ax[0].set_title('Confusion Matrix', fontsize="large")

            roc_curve.update(preds, target)
            roc_curve.plot(score=True, ax=ax[2])

            if not isAx:
                fig.tight_layout()
                fig.show()

    if config.TRAINING_TYPE == "BINARY":
        return accuracy(preds, target).item(), dice(preds, target).item(), jaccard(preds, target).item(), \
               ap(preds, target).item(), f1(preds, target).item()
    elif config.TRAINING_TYPE == "MULTICLASS":
        return accuracy(predMask, target).item(), jaccard(predMask, target).item(), \
               ap(predMask.unsqueeze(0), targetMono), f1(predMask, target).item()

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

def find_best_binary_threshold(target, preds, start=0., stop=1., n_thresholds=50, index='accuracy'):
    thresholds = np.linspace(start, stop, n_thresholds)
    best_threshold = 0
    best_score = -np.inf
    for i in range(n_thresholds):
        c_index = None
        if index != 'mean':
            if index == 'dice':
                c_index = Dice(threshold=thresholds[i])
            elif index == 'jaccard':
                c_index = JaccardIndex(task='binary', threshold=thresholds[i])
            elif index == 'f1':
                c_index = F1Score(task='binary', threshold=thresholds[i])
            else:
                c_index = Accuracy(task='binary', threshold=thresholds[i])

            score = c_index(preds, target).item()
            if score > best_score:
                best_score = score
                best_threshold = thresholds[i]
        else:
            return (find_best_binary_threshold(target, preds, start, stop, n_thresholds, index='dice') + \
                   find_best_binary_threshold(target, preds, start, stop, n_thresholds, index='accuracy')) / 2

    return best_threshold

def find_best_multiclass_threshold(model, data, start=0., stop=1., n_thresholds=50, index='accuracy'):
    thresholds = np.linspace(start, stop, n_thresholds)
    best_threshold = 0
    best_score = -np.inf
    for i in range(n_thresholds):
        c_index = None
        if index != 'mean':
            if index == 'jaccard':
                c_index = JaccardIndex(task='multiclass', num_classes=config.NB_CLASSES)
            elif index == 'f1':
                c_index = F1Score(task='multiclass', num_classes=config.NB_CLASSES)
            else:
                c_index = Accuracy(task='multiclass', num_classes=config.NB_CLASSES)

            multiMask, _, predMask = make_predictions(model, data, plot=False, computeMetrics=False)
            score = c_index(predMask, multiMask).item()
            if score > best_score:
                best_score = score
                best_threshold = thresholds[i]
        else:
            return (find_best_multiclass_threshold(target, preds, start, stop, n_thresholds, index='jaccard') + \
                   find_best_multiclass_threshold(target, preds, start, stop, n_thresholds, index='accuracy')) / 2

    return best_threshold

# multiMask : target masks
# multiPred : predicted masks
# mask : monochannel target mask
# pred : monochannel predicted mask
def make_predictions(model, data, threshold=config.PRED_THRESHOLD, plot=True, computeMetrics=True):
    # Set model to evaluation mode
    model.eval()

    # Turn off gradient tracking
    with torch.no_grad():
        (image, multiMask) = data

        # Make the prediction, pass the results through the sigmoid function, and convert the result to a NumPy array
        multiPred = model(image.unsqueeze(0))
        if config.TRAINING_TYPE == "BINARY":
            multiPred = F.sigmoid(multiPred)
        elif config.TRAINING_TYPE == "MULTICLASS":
            # multiPred = F.softmax(multiPred, dim=0)
            multiPred = F.sigmoid(multiPred)

        if not config.INTERPOLATE:
            multiMask = torchvision.transforms.CenterCrop([multiPred.size(2), multiPred.size(3)])(multiMask)

        # Group every masks in a single image and add the mask we didn't consider directly during training (landing-zones)
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

        # plt.figure()
        # plt.imshow(multiPred.squeeze(), cmap='gray')
        # plt.title("Predicted probability map for binary case")
        # plt.show()

        # plt.imshow(image.swapdims(0,1).swapdims(1,2))
        # plt.title("Original image")
        #
        # plt.figure()
        # fig, ax = plt.subplots(2, 2, figsize=(5,5))
        # for i in range(4):
        #     ax[i//2][i%2].imshow(multiPred[0, i,:,:], cmap='gray', vmin=0, vmax=1)
        #     ax[i//2][i%2].set_title("Probability map for class {}".format(config.DICT_MASKS[i]))
        # plt.show()

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

        # plt.figure()
        # # multiMask_bis = torchvision.transforms.CenterCrop([388, 276])(multiMask)
        # pred_bis = F.interpolate(pred.swapdims(0,1).unsqueeze(0).unsqueeze(0), [388, 276], mode='nearest')
        # pred_bis = pred_bis.squeeze(0)
        #
        # cropMask = torch.zeros([config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH])
        # center_x = int(config.INPUT_IMAGE_WIDTH / 2) - 1
        # center_y = int(config.INPUT_IMAGE_HEIGHT / 2) - 1
        # cropMask[center_y-int(multiMask.size(2)/2):center_y+int(multiMask.size(2)/2),
        #         center_x-int(multiMask.size(3)/2):center_x+int(multiMask.size(3)/2)] = multiMask[0,0,:,:]
        # plt.imshow(cropMask, cmap='gray', vmin=0, vmax=1)

        mask = mask.type(torch.IntTensor)
        multiMask = multiMask.type(torch.IntTensor)
        pred = pred.type(torch.IntTensor)

        # Plots and metrics computing
        if plot and computeMetrics:
            fig, ax = plt.subplots(2, 3, figsize=(10,8))
            plot_one_mask(image.swapdims(0,1).swapdims(1,2), mask.squeeze(), pred.squeeze(), ax[0])
            if config.TRAINING_TYPE == 'MULTICLASS':
                plot_multi_masks(image.swapdims(0,1).swapdims(1,2), multiMask, predMask)
            error = (mask == pred) * 255
            ax[1][1].imshow(error.squeeze(), cmap='RdYlGn', vmin=0, vmax=255)
            ax[1][1].set_title("Error map")
            fig.show()

            if config.TRAINING_TYPE == 'BINARY':
                # return accuracy, dice, jaccard, ap, f1
                return metrics(multiMask, multiPred, threshold, ax=ax[1])
            elif config.TRAINING_TYPE == 'MULTICLASS':
                return metrics(multiMask, multiPred, threshold, targetMono=mask.unsqueeze(0), maskMono=mask,
                               predMask=predMask, ax=ax[1])

        elif plot and not computeMetrics:
            fig, ax = plt.subplots(1, 3, figsize=(10,4))
            plot_one_mask(image.swapdims(0,1).swapdims(1,2), mask.squeeze(), pred.squeeze(), ax)
            if config.TRAINING_TYPE == 'MULTICLASS':
                plot_multi_masks(image.swapdims(0,1).swapdims(1,2), multiMask, predMask)
            fig.show()

        elif not plot and computeMetrics:
            accuracy, dice, jaccard, ap, f1 = metrics(multiMask, multiPred, threshold, plot=False)
            return accuracy, dice, jaccard, ap, f1

        return multiMask, multiPred, predMask

if __name__ == '__main__':
    nbExamples = 25 # N  umber of examples to analyze among the test set
    # 20 examples with seed 42 : 2,5 for shadow problem, 13 for precise prediction, 15 & 16 for many details non-classification
    # 20 : not bad with SGD no interpolation
    # 20 examples with seed 16 : 1 for misprediction, 7 is a mess, 13 for persons, 20 for car with persons (and interpolation)
    # 20 examples with seed 32 : 6 good for interpolation, 18 for many people (interpolation)
    selection = [] # Select the data you want to plot among the nbExamples ones (empty to study all examples)
    randomSeed = 5 # Seed for the random selection of those examples (42 for above selection)
    # threshold = 0.438 # Best for accuracy
    # threshold = 0.271 # Best for Dice, Jaccard and F1
    # threshold =
    threshold = 0.2 # Threshold to be considered is findBestThreshold = False
    plot = False # Plot the image and the masks
    computeMetrics = True # Compute and plot the metrics (Precision - Recall curve & Confusion Matrix)
    findBestThreshold = True

    # Start, stop, number of values to consider and index to use (dice, jaccard, f1, accuracy or mean (= mean of dice and accuracy)
    # -> accuracy by default)
    findBestThresholdArgs = (0., 1., 50, 'mean')

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
    testIndex = np.random.choice(range(0,len(testSet)), size=nbExamples, replace=False)

    if len(selection) > 0:
        testIndex = [testIndex[i] for i in selection]

    # Load our model from disk and flash it to the current device
    print("[INFO] Load up model...")
    if config.TRAINING_TYPE == 'BINARY':
        unet = torch.load(config.BINARY_MODEL_PATH, map_location=torch.device(config.DEVICE)).to(config.DEVICE)
    elif config.TRAINING_TYPE == 'MULTICLASS':
        unet = torch.load(config.MULTICLASS_MODEL_PATH, map_location=torch.device(config.DEVICE)).to(config.DEVICE)

    # Looking for the threshold.s which maximize the results according to a score (DICE (only for binary), Jaccard, F1 or Accuracy)
    thresholds = []
    if findBestThreshold:
        thresholds = []
        if config.TRAINING_TYPE == 'BINARY':
            for ind in testIndex:
                multiMask, multiPred, _ = make_predictions(unet, testSet[ind], plot=False, computeMetrics=False)
                thresholds.append(find_best_binary_threshold(multiMask.squeeze(0), multiPred.squeeze(0),
                                                       start=findBestThresholdArgs[0], stop=findBestThresholdArgs[1],
                                                       n_thresholds=findBestThresholdArgs[2],
                                                       index=findBestThresholdArgs[3]))
        elif config.TRAINING_TYPE == 'MULTICLASS':
            for ind in testIndex:
                thresholds.append(find_best_multiclass_threshold(unet, testSet[ind], start=findBestThresholdArgs[0],
                                                 stop=findBestThresholdArgs[1], n_thresholds=findBestThresholdArgs[2],
                                                 index=findBestThresholdArgs[3]))
        mean_threshold = sum(thresholds)/len(thresholds) # Generally around 0.35 for DICE, Jaccard and F1, around 0.5 for Accuracy
        print("Best thresholds      = {}".format(thresholds))
        print("Best thresholds mean = {}".format(mean_threshold))
    else:
        thresholds = [threshold for i in range(nbExamples)]

    if plot or computeMetrics:
        # Ploting and analyzing the few data one by one
        accuracies, dices, jaccards, aps, f1s = [], [], [], [], []
        for i in range(len(testIndex)):
            if computeMetrics:
                if config.TRAINING_TYPE == 'BINARY':
                    accuracy, dice, jaccard, ap, f1 = make_predictions(unet, testSet[testIndex[i]], threshold=thresholds[i],
                                                                   plot=plot, computeMetrics=computeMetrics)
                elif config.TRAINING_TYPE == 'MULTICLASS':
                    accuracy, jaccard, ap, f1 = make_predictions(unet, testSet[testIndex[i]], threshold=thresholds[i],
                                                                   plot=plot, computeMetrics=computeMetrics)
                accuracies.append(accuracy)
                if config.TRAINING_TYPE == 'BINARY':
                    dices.append(dice)
                jaccards.append(jaccard)
                aps.append(ap)
                f1s.append(f1)
            elif plot:
                make_predictions(unet, testSet[testIndex[i]], threshold=thresholds[i], plot=plot,
                                 computeMetrics=computeMetrics)

        if len(dices) > 0:
            print("###################### EVALUATION ######################")
            # print("Accuracies               = {}".format(accuracies))
            print("Mean accuracy            = {}".format(sum(accuracies)/len(accuracies)))
            print("Variance accuracy        = {}".format(np.std(np.array(accuracies))))
            if(config.TRAINING_TYPE == 'BINARY'):
                # print("Dice indexs              = {}".format(dices))
                print("Dice mean                = {}".format(sum(dices)/len(dices)))
                print("Dice variance            = {}".format(np.std(np.array(dices))))
            # print("Jaccard indexs           = {}".format(jaccards))
            print("Jaccard mean             = {}".format(sum(jaccards)/len(jaccards)))
            print("Jaccard variance         = {}".format(np.std(np.array(jaccards))))
            # print("Average-Precision indexs = {}".format(aps))
            print("Average-Precision mean   = {}".format(sum(aps)/len(aps)))
            print("AP variance              = {}".format(np.std(np.array(aps))))
            # print("F1 scores                = {}".format(f1s))
            print("F1 mean                  = {}".format(sum(f1s)/len(f1s)))
            print("F1 variance              = {}".format(np.std(np.array(f1s))))
            print("########################################################")

        if plot:
            plt.show()