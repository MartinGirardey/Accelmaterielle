# Import the necessary packages
from model_pyimagesearch.dataset import SegmentationBinaryDataset, SegmentationMulticlassDataset
from model_pyimagesearch.UNet import UNet
from model_pyimagesearch import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision.transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import glob

if __name__ == '__main__':
    # Load the image and mask filepaths in a sorted manner
    imagePaths = None ; maskPaths = None
    if config.TRAINING_TYPE == 'BINARY':
        imagePaths = sorted(list(glob.glob(config.IMAGE_BINARY_DATASET_PATH + "/*.png")))
        maskPaths = sorted(list(glob.glob(config.MASK_BINARY_DATASET_PATH + "/*.png")))
    elif config.TRAINING_TYPE == 'MULTICLASS':
        imagePaths = sorted(list(glob.glob(config.IMAGE_MULTICLASS_DATASET_PATH + "/*.png")))
        maskPaths = sorted(list(glob.glob(config.MASK_MULTICLASS_DATASET_PATH + "/*.png")))

    # Define transformations
    # transforms = None
    transforms = torchvision.transforms.Resize((config.OUTPUT_IMAGE_HEIGHT, config.OUTPUT_IMAGE_WIDTH),
                                                       interpolation=torchvision.transforms.InterpolationMode.NEAREST)

    dataset = None
    if config.TRAINING_TYPE == 'BINARY':
        dataset = SegmentationBinaryDataset(imagePaths=imagePaths, maskPaths=maskPaths, transforms=transforms)
    elif config.TRAINING_TYPE == 'MULTICLASS':
        dataset = SegmentationMulticlassDataset(imagePaths=imagePaths, maskPaths=maskPaths, transforms=transforms)

    # Partition of the data into training and testing sets
    trainSize = int((1-config.TEST_SPLIT)*len(dataset))
    trainDS, testDS = torch.utils.data.random_split(dataset, [trainSize, len(dataset)-trainSize],
                                               generator=torch.Generator().manual_seed(config.SPLIT_SEED))

    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # # Plotting the different masks (multiclass case)
    # image, mask = trainDS[1]
    # plt.subplot(2,3,1)
    # plt.imshow(image.swapdims(0,1).swapdims(1,2))
    # for i in range(4):
    #     plt.subplot(2,3,i+2)
    #     plt.imshow(mask[i,:,:])
    # plt.show()

    # Create the training and test data loaders (on cuda, for num_workers, we consider the value 2 * number of GPU used)
    trainLoader = DataLoader(trainDS, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                             num_workers=8)
    testLoader = DataLoader(testDS, shuffle=False, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                             num_workers=8)

    # Set or load the UNet model
    unet = None
    if config.CONTINUE_TRAINING: # Load UNet model to continue training
        if config.TRAINING_TYPE == 'BINARY':
            unet = torch.load(config.BINARY_MODEL_PATH).to(config.DEVICE)
        elif config.TRAINING_TYPE == 'MULTICLASS':
            unet = torch.load(config.MULTICLASS_MODEL_PATH).to(config.DEVICE)
    else: # Train a model from scratch
        if config.TRAINING_TYPE == 'BINARY':
            unet = UNet(config.ENCODER_CHANNELS, config.DECODER_CHANNELS, config.NB_CLASSES, retainDim=config.INTERPOLATE)\
                .to(config.DEVICE)
        elif config.TRAINING_TYPE == 'MULTICLASS':
            unet = UNet(config.ENCODER_CHANNELS, config.DECODER_CHANNELS, config.NB_CLASSES-1, retainDim=config.INTERPOLATE)\
                .to(config.DEVICE)

    # Initialize loss function and optimizer
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=config.INIT_LR)

    # Initialize a dictionary to store training history
    H = {"train_loss": [], "test_loss": []}

    # Loop over epochs
    print("[INFO] Training the network...")
    startTime = time.time()
    minTestLoss = np.inf
    for e in range(config.NUM_EPOCHS):
        # Set the model in training mode
        unet.train()

        # Initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0

        # Loop over the training set
        for (i, (x, y)) in enumerate(trainLoader):
            # Send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            x = F.interpolate(x, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH), mode='nearest')

            # Perform a forward pass and calculate the training loss
            pred = unet(x)
            if not config.INTERPOLATE:
                y = torchvision.transforms.CenterCrop([pred.size(2), pred.size(3)])(y)
                loss = lossFunc(pred, y)
            else:
                loss = lossFunc(pred, y)

            # First, zero out any previously accumulated gradients, then perform backpropagation, and then update model
            # parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Add the loss to the total training loss so far
            totalTrainLoss += loss

        # Switch off autograd
        with torch.no_grad():
            # Set the model in evaluation mode
            unet.eval()

            # Loop over the validation set
            for (i, (x, y)) in enumerate(testLoader):
                # Send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                # Make the predictions and calculate the validation loss
                pred = unet(x)
                if not config.INTERPOLATE:
                    y = torchvision.transforms.CenterCrop([pred.size(2), pred.size(3)])(y)
                    loss = lossFunc(pred, y)
                else:
                    loss = lossFunc(pred, y)

                totalTestLoss += loss

        # Calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss * (len(trainDS) + len(testDS)) / len(trainDS)
        avgTestLoss = totalTestLoss * (len(trainDS) + len(testDS)) / len(testDS)

        # Update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

        # Print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e+1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.6f}".format(avgTrainLoss, avgTestLoss))

        if avgTestLoss < minTestLoss:
            print("Test loss {} -> {} : saving model...".format(minTestLoss, avgTestLoss))
            if config.TRAINING_TYPE == 'BINARY':
                torch.save(unet, config.BINARY_MODEL_PATH)
            elif config.TRAINING_TYPE == 'MULTICLASS':
                torch.save(unet, config.MULTICLASS_MODEL_PATH)
            minTestLoss = avgTestLoss

    # Display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] Total time taken to train the model: {:.2f}s".format(endTime - startTime))

    # Plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    if config.TRAINING_TYPE == 'BINARY':
        plt.savefig(config.BINARY_PLOT_PATH)
    elif config.TRAINING_TYPE == 'MULTICLASS':
        plt.savefig(config.MULTICLASS_PLOT_PATH)