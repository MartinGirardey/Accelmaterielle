# Import the necessary packages
from model_pyimagesearch.dataset import SegmentationBinaryDataset, SegmentationMulticlassDataset
from model_pyimagesearch.UNet import UNet
from model_pyimagesearch import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os

if __name__ == '__main__':
    binary = False

    # Load the image and mask filepaths in a sorted manner
    imagePaths = None ; maskPaths = None
    if binary:
        imagePaths = sorted(list(paths.list_images(config.IMAGE_BINARY_DATASET_PATH)))
        maskPaths = sorted(list(paths.list_images(config.MASK_BINARY_DATASET_PATH)))
    else:
        imagePaths = sorted(list(paths.list_images(config.IMAGE_MULTICLASS_DATASET_PATH)))
        maskPaths = sorted(list(paths.list_images(config.MASK_MULTICLASS_DATASET_PATH)))

    # Partition the data into training and testing splits using 85% of the data for training and the remaining 15% for
    # testing
    split = train_test_split(imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=42)

    # Unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    # Write the testing image paths to disk so that we can use then when evaluating/testing our model
    print("[INFO] Saving testing image paths...")
    f = None
    if binary:
        f = open(config.BINARY_TEST_PATHS, "w")
    else:
        f = open(config.MULTICLASS_TEST_PATHS, "w")
    f.write("\n".join(testImages))
    f.close()

    # Define transformations
    transforms = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH),
                                                       interpolation=transforms.InterpolationMode.NEAREST),
                                     transforms.ToTensor()])

    # Create the train and test datasets
    trainDS = None ; testDS = None
    if binary:
        trainDS = SegmentationBinaryDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms)
        testDS = SegmentationBinaryDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms)
    else:
        trainDS = SegmentationMulticlassDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms)
        testDS = SegmentationMulticlassDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms)

    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # Plotting the different masks (multiclass case)
    # image, mask = trainDS[1]
    # plt.subplot(2,3,1)
    # plt.imshow(image.swapdims(0,1).swapdims(1,2))
    # for i in range(5):
    #     plt.subplot(2,3,i+2)
    #     plt.imshow(mask[i,:,:])
    # plt.show()

    # Create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                             num_workers=os.cpu_count())
    testLoader = DataLoader(testDS, shuffle=False, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                             num_workers=os.cpu_count())

    # Load UNet model to continue training
    if binary:
        unet = torch.load(config.BINARY_MODEL_PATH).to(config.DEVICE)
    else:
        unet = torch.load(config.MULTICLASS_MODEL_PATH).to(config.DEVICE)

    # Initialize our UNet model
    # unet = None
    # if binary:
    #     unet = UNet(encChannels=config.ENCODER_CHANNELS, decChannels=config.DECODER_CHANNELS).to(config.DEVICE)
    # else:
    #     unet = UNet(encChannels=config.ENCODER_CHANNELS, decChannels=config.DECODER_CHANNELS, nbClasses=config.NB_CLASSES)\
    #            .to(config.DEVICE)

    # Initialize loss function and optimizer
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=config.INIT_LR)

    # Initialize a dictionary to store training history
    H = {"train_loss": [], "test_loss": []}

    # Loop over epochs
    print("[INFO] Training the network...")
    startTime = time.time()
    minTestLoss = np.inf
    for e in tqdm(range(config.NUM_EPOCHS)):
        # Set the model in training mode
        unet.train()

        # Initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0

        # Loop over the training set
        for (i, (x, y)) in enumerate(trainLoader):
            # Send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            # Perform a forward pass and calculate the training loss
            pred = unet(x)
            loss = lossFunc(pred, y)

            # First, zero out any previously accumulated gradients, then perform backpropagation, and then update model
            # parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Add the loss to the total training loss so far
            totalTrainLoss += loss

            print("Epoch {} : Train batch {} over {}".format(e+1, i+1, len(trainLoader)))

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
                totalTestLoss += lossFunc(pred, y)

                print("Epoch {} : Test batch {} over {}".format(e+1, i+1, len(testLoader)))

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
            if binary:
                torch.save(unet, config.BINARY_MODEL_PATH)
            else:
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
    if binary:
        plt.savefig(config.BINARY_PLOT_PATH)
    else:
        plt.savefig(config.MULTICLASS_PLOT_PATH)
