## Importing Relevant Libraries

import torch
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.lraspp import LRASPPHead
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from natsort import natsorted

## Define Main Function

def main():
    global criterion
    global optimizer
    global model
    global root
    
    root = os.getcwd()
    lr = 0.001
    epochs = 20
    
    print("\nAssign model")
    model = VariableHead(outputchannels = 1).DeepLabV3()
    
    print('\nDefine Loss Function and Optimizer')
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    
    print('\nLoad the data')
    trainval_loader, test_loader = Dataloader().loader()
    
    print('\nTrain the network')
    model = loop(trainval_loader).train_model(epochs)
    
    print('\nRun on test dataset to calculate a Dice Score')
    loop(test_loader).test_model()
    
    return model

## Create Pretrained Models with Trainable Parameters

class VariableHead():
    def __init__(self, outputchannels):
        self.outputchannels = outputchannels
        
    def DeepLabV3(self):
        global model_type 
        model_type = 'DLV3'
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
        # Replace the classifier with different output channels
        model.classifier = DeepLabHead(2048, self.outputchannels)
        # Freeze all layers except the classifier
        for param in model.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    
    def FCN101(self):
        global model_type 
        model_type = 'FCN'
        model = torchvision.models.segmentation.fcn_resnet101(pretrained=True, progress=True)
        # Replace the classifier with different output channels
        model.classifier = FCNHead(2048, self.outputchannels)
        # Freeze all layers except the classifier
        for param in model.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model
    
    def MobileNet(self):
        global model_type
        model_type = 'MBN'
        model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained = True, progress = True)
        # Replace the classifier with different output channels
        model.classifier = LRASPPHead(low_channels = 40, 
                                      high_channels = 960, 
                                      num_classes = self.outputchannels, 
                                      inter_channels = 64)
        # Freeze all layers except the classifier
        for param in model.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model

## Import Dataset of Images and Masks

### Create list of paths in directory

class DatasetList():
    def __init__(self, directory: str):
        self.dir = directory
    
    def __lst__(self):
        d = self.dir
        lst = []
        for path in os.listdir(d):
            full_path = os.path.join(d, path)
            if os.path.isfile(full_path):
                lst.append(full_path)
        return natsorted(lst)

### Create a dataset approriate for loading

class SegmentationDataSet(data.Dataset):
    def __init__(self, inputs: list, targets: list, image_transform_small = None, mask_transform_small = None, image_transform_large = None, mask_transform_large = None):
        self.inputs = inputs
        self.targets = targets
        self.image_transform_small = image_transform_small
        self.mask_transform_small = mask_transform_small
        self.image_transform_large = image_transform_large
        self.mask_transform_large = mask_transform_large
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y = Image.open(input_ID).convert('RGB'), Image.open(target_ID).convert('1')

        # Preprocessing
        w,h = x.size       
        if w > 250:
          x = transforms.functional.five_crop(x,224)
          y = transforms.functional.five_crop(y,224)
          ii = 0
          for image in x:
            im = self.image_transform_large(image)
            mask = self.mask_transform_large(y[ii])
            ii += 1
            return im, mask
        else:
          x = self.image_transform_small(x)
          y = self.mask_transform_small(y)
            
        return x, y

### Load the data using PyTorch

class Dataloader():
    def __init__(self):
        global root

        self.root = root

        self.image_transforms_small = transforms.Compose([transforms.Resize(256),
                                                          transforms.CenterCrop(224),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                                std=[0.229, 0.224, 0.225])])
        
        self.mask_transforms_small = transforms.Compose([transforms.Resize(256),
                                                         transforms.CenterCrop(224),
                                                         transforms.ToTensor()])
        
        self.image_transforms_large = transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                                std=[0.229, 0.224, 0.225])])
        
        self.mask_transforms_large = transforms.Compose([transforms.ToTensor()])
        self.batch_size = 16
        self.shuffle = True
        
    def loader(self):
        entire_set = SegmentationDataSet(DatasetList(self.root + "/Image").__lst__(), 
                                         DatasetList(self.root + "/Mask").__lst__(), 
                                         image_transform_small = self.image_transforms_small, 
                                         mask_transform_small = self.mask_transforms_small,
                                         image_transform_large = self.image_transforms_large, 
                                         mask_transform_large = self.mask_transforms_large)
                
        train_length = int(0.8* len(entire_set))
        
        test_length = len(entire_set) - train_length

        trainval, test = torch.utils.data.random_split(entire_set, [train_length, test_length])
        
        trainval_loader = torch.utils.data.DataLoader(trainval, 
                                                      batch_size = self.batch_size, 
                                                      shuffle = self.shuffle)
        
        test_loader = torch.utils.data.DataLoader(test, 
                                                  batch_size = self.batch_size, 
                                                  shuffle = self.shuffle)
        return trainval_loader, test_loader

## Train the Model

class loop():
    def __init__(self, data_loader):
        global criterion
        global optimizer
        global model
        global model_type
        
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.model_type = model_type
            
    def train(self):        
        # Initialize loss
        running_loss = 0.0
        
        # Iterate over input batches
        for input_batch in iter(self.data_loader):
            inputs, masks = input_batch
            if torch.cuda.is_available():
              inputs = inputs.cuda()
              masks = masks.cuda()
              self.model = self.model.cuda()
            # zero parameter gradients
            optimizer.zero_grad()
            # Set model to training mode
            self.model.train()
            # Forward pass
            outputs = self.model(inputs)
            # Loss calculation
            loss = criterion(outputs['out'], nn.Sigmoid()(masks))
            # backpropagate
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        return running_loss, self.model
    
    def validate(self):
        self.model.eval()
        
        # Initialize metrics
        total_train = 0.0
        correct_train = 0.0
        
        # Iterate over input batches
        for input_batch in iter(self.data_loader):
            inputs, masks = input_batch
            
            if torch.cuda.is_available():
              inputs = inputs.cuda()
              masks = masks.cuda()
              self.model = self.model.cuda()
           
            outputs = self.model(inputs)
            predicted = (nn.Sigmoid()(outputs['out'].data) >= 0.5).float()
            total_train += masks.nelement()
            correct_train += predicted.eq(masks.data).sum().item()
            train_accuracy = 100*correct_train/total_train
        
        return train_accuracy
    
    def test(self):
        self.model.eval()
        
        # Initialize Metrics
        total_dice = 0.0
        total_batch = 0
        
        for input_batch in iter(self.data_loader):
            inputs, masks = input_batch
            if torch.cuda.is_available():
              inputs = inputs.cuda()
              masks = masks.cuda()
              self.model = self.model.cuda()
            total_batch += 1
            outputs = self.model(inputs)
            predicted = (nn.Sigmoid()(outputs['out'].data) >= 0.5).float()
            targets = masks.data
            total_dice += Metrics(predicted, targets).dice_coeff()
        
        return total_dice/total_batch
    
    def train_model(self, num_epochs):
        train_model = {}
        error_count = 0
        for epoch in range(1, num_epochs + 1):
            print('\nEpoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)
            # Training Loop
            train = self.train()
            train_loss = train[0]
            print('\nTraining loss: %9f' %(train_loss))
            # Validation Loop
            valid_accu = self.validate()
            print('\nValidation Accuracy: %9f' %(valid_accu))
            train_model[epoch] = [valid_accu, train[1]]
            
            with open('training data.txt', "a") as file:
              file.writelines('\nEpoch ' + str(epoch) + ' Train Loss: ' + str(train[0]) + ' Valid Accu: ' + str(train_model[epoch][0]))
            
            # Termination Conditions
            current_accu = list(train_model.values())[epoch - 1][0]
            min_accu = min(train_model.values())[0]
            
            if epoch >= 10 and current_accu <= min_accu:
                error_count += 1
            
            elif error_count >= 5:
                print("\n*****Optimization error, restarting training*****\n")
                main()
                break

        best_model = max(train_model.values())[1]
        self.model = best_model
        
        saveload().model_save()
        
        return self.model
    
    def test_model(self):
         # Test Loop
         coeff = self.test()
         print("\nDice Score on Test Data: %5f" %(coeff))
         
         with open('training data.txt', "a") as file:
              file.writelines('\nDice Score on Test Data: ' + str(coeff))

### Save and Load a Model

class saveload():
  def __init__(self):
    global model_type
    global model
    global root

    self.model = model
    self.model_type = model_type

  def model_save(self):
    torch.save(self.model.state_dict(), root + "/" + self.model_type + "_dict.pt")

### Define the metrics used

class Metrics():
    def __init__(self, predicted, targets):
        self.image = predicted.to('cpu')
        self.mask = targets.to('cpu')
    
    def dice_coeff(self):
        im1 = np.asarray(self.image).astype(np.bool)
        im2 = np.asarray(self.mask).astype(np.bool)

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        return 2. * intersection.sum() / (im1.sum() + im2.sum())

main()