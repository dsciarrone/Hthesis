## Importing Relevant Libraries
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as matimg
import numpy as np
import os
from natsort import natsorted
import random

## Define Main Function

def main():
    global criterion
    global optimizer
    global model
    global root
    global patch_size
    global save_pth
    
    root = os.getcwd()
    save_pth = root + '/saves/'
    patch_size = 9
    lr = 0.001
    epochs = 10
    
    with open('log.txt', 'a') as lg:
      lg.writelines("\nAssign model")
      model = classifier().resnet_101()
      lg.writelines("\nModel Assigned")
      
      lg.writelines('\nDefine Loss Function and Optimizer')
      criterion = nn.BCELoss()
      optimizer = optim.Adam(model.parameters(), lr)
      lg.writelines('\nLoss function and optimizer defined')
      
      lg.writelines('\nLoad the data')
      trainval_loader, test_loader = Dataloader().loader()
      lg.writelines('\nData loaded')
      
      #lg.writelines('\nVisualise a batch of training data')
      #visualise()._batch_(trainval_loader)
      
      lg.writelines('\nTrain the network')
      model = loop(trainval_loader).train_model(epochs)
      
      lg.writelines('\nRun on test dataset to calculate test accuracy')
      loop(test_loader).test_model()
      
      lg.writelines('\nVisualise a single test image')
      image, mask, name = visualise().__iter__()
      lg.writelines('\n' + str(name))
      plt.imshow(image)
      plt.show()
      plt.imshow(mask, cmap = 'gray', vmin = 0, vmax = 1)
      plt.show()
      
      lg.writelines('\nRun the trained model on that single test image')
      prediction = Decoder(image).single_test()
      Decoder(image).mask(prediction)

      lg.writelines('\nDice score for prediction to manual mask')
      dice = metrics().dice(mask,prediction)
      lg.writelines(dice)

    return model, trainval_loader, prediction

##Create an Untrained Classifier Network

class classifier():
    def resnet_101(self):
      global model_type
      model_type = 'resnet101'
      model = torchvision.models.resnet101()
      model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2),
               nn.Softmax(1))
      return model

    def alexnet(self):
      global model_type
      model_type = 'alexnet'
      model = torchvision.models.alexnet()
      model.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)
      model.classifier = nn.Sequential(*list(model.classifier) + [nn.Softmax(1)])
      return model

##Import Dataset of Images and Masks
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

class ClassificationDataset(data.Dataset):
    def __init__(self, data, targets, trans=None):
        self.data = data
        self.targets = torch.FloatTensor(targets)
        self.transform = trans
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(x)
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

### Load the data using PyTorch

class Dataloader():
    def __init__(self):
        global root
        global patch_size

        self.transforms = transforms.Compose([transforms.Resize(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                         std=[0.229, 0.224, 0.225])])
        self.batch_size = 16
        self.shuffle = True
        
    def loader(self):
        patches, targets = self.patcher()
        entire_set = ClassificationDataset(patches, targets, trans = self.transforms)
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
    
    def patcher(self):
      a = DatasetList(root + "/Image").__lst__()
      b = DatasetList(root + "/Mask").__lst__() 
      
      pad_size = int(np.floor(patch_size/2))
      patches = []
      targets = np.array([[0.0,0.0]], dtype = 'f')
      num_im = 2

      for ii in range(num_im):
        im = np.array(Image.open(a[ii]).convert('RGB'))

        msk = np.array(Image.open(b[ii]).convert('1'))

        w,h = msk.shape
        W = w - patch_size
        H = h - patch_size

        y = 0
        for y in range(H):
          x = 0
          for x in range(W):
            
            if x == 0 and y == 0:
              if msk[x + pad_size,y + pad_size] == 0:
                targets[0] = [0.0, 1.0]
              else:
                targets[0] = [1.0, 0.0]
            
            if msk[x + pad_size,y + pad_size] == 0:
              patches.append(im[x :x + patch_size, y : y + patch_size,:])
              targets = np.append(targets,[[0.0, 1.0]],axis = 0)
            else:
              patches.append(im[x :x + patch_size, y : y + patch_size,:])
              targets = np.append(targets,[[1.0, 0.0]], axis = 0)
            x += 1
          y += 1
      elements = np.array(random.sample(range(0,len(patches)), 20000))
      return np.array(patches)[elements], targets[elements]

## Visualise a batch of training data

class visualise():
    def __init__(self):
        global root
        self.init = 1

    def imshow(self, img): # defining a function to show an image
        img = img/2+(0.5*self.init) # unnormalize the image
        npimg = np.array(img.permute(1,2,0))
        image = Image.fromarray(npimg)
        image.save(save_pth + "training batch.png")
        plt.imshow(npimg)
        plt.show()
    
    def _batch_(self, data_loader):
        # show a random batch of training images
        dataiter = iter(data_loader)
        images, targets = next(dataiter)
        self.imshow(torchvision.utils.make_grid(images))
        print(torch.argmax(targets, axis = 1))
    
    def __iter__(self):
        im_path = DatasetList(root + "/Image").__lst__()
        msk_path = DatasetList(root + "/Mask").__lst__()
        rnd = random.randint(0,len(msk_path) - 1)
        img = np.array(Image.open(im_path[rnd]).convert('RGB'))
        msk = np.array(Image.open(msk_path[rnd]).convert('1'))

        return img, msk, im_path[rnd]

## Train the Model

class loop():
    def __init__(self, data_loader):
        global criterion
        global optimizer
        global model
        global model_type
        global save_pth

        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
            
    def train(self):        
        # Initialize loss
        running_loss = 0.0        
        # Iterate over the training set
        for batch in iter(self.data_loader):
          inputs, targets = batch
          
          if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
            self.model = self.model.cuda()
          
          self.optimizer.zero_grad()
          # Set model to training mode
          self.model.train()
          # Forward pass
          outputs = self.model(inputs)
          # Loss calculation
          loss = self.criterion(outputs, targets)
          # backpropagate
          loss.backward()
          self.optimizer.step()
          running_loss += loss.item()
        
        return running_loss, self.model
    
    def validate(self):
        self.model.eval()
        
        # Initialize metrics
        total_train = 0.0
        correct_train = 0.0
        
        # Iterate over input batches
        for input_batch in iter(self.data_loader):
            inputs, targets = input_batch
            if torch.cuda.is_available():
              inputs = inputs.cuda()
              targets = targets.cuda()
              self.model = self.model.cuda()
            outputs = self.model(inputs)
            predictions = torch.argmax(outputs, axis = 1)
            truths = torch.argmax(targets, axis = 1)
            total_train += truths.nelement()
            correct_train += predictions.eq(truths).sum().item()
            train_accuracy = 100*correct_train/total_train
        
        return train_accuracy
    
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
         accu = self.validate()
         print("\nAccuracy on Test Data: %5f" %(accu))
         with open('training data.txt', "a") as file:
              file.writelines('Accuracy on Test Data: ' + str(accu))
         return accu

"""### Save and Load a Model"""

class saveload():
  def __init__(self):
    global model_type
    global save_pth
    global model

    self.model = model
    self.model_type = model_type
    self.save_pth = save_pth

  def model_save(self):
    torch.save(self.model.state_dict(), self.save_pth + "/" + self.model_type + "_dict.pt")

  def model_load(self, pth):
    
    if 'resnet101' in self.model_type:
      self.model = classifier().resnet_101()
      self.model.load_state_dict(torch.load(pth))
    
    if 'alexnet' in self.model_type:
      self.model = classifier().alexnet()
      self.model.load_state_dict(torch.load(pth))

    self.model.eval()

    return self.model

## Define Comparison Metrics

class metrics():
  def __init__(self):
    global patch_size
  
  def dice(self, mask, prediction):
    
    ind = int(np.floor(patch_size/2))
    im1 = np.asarray(mask[ind:-ind-1,ind:-ind-1]).astype(np.bool)
    im2 = np.asarray(prediction).astype(np.bool)
    print(im1.shape)
    print(im2.shape)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

## Decode the segmentation

class Decoder():
    def __init__(self, image):
        global model
        global patch_size
        self.img = image
        self.model = model
    
    def OCT(self, msk):
        clean_img = self.img*msk
        npimg = clean_img.numpy()
        plt.imshow(npimg)
        plt.show()
    
    def mask(self, msk):
        plt.imshow(msk, cmap = 'gray', vmin = 0, vmax = 1)
        plt.show()
    
    def single_test(self):
        pad_size = int(np.floor(patch_size/2))
        w,h,c = self.img.shape
        W = w - patch_size
        H = h - patch_size
        prediction = np.zeros((W,H))
        
        y = 0
        for y in range(H):
          x = 0
          for x in range(W):
            patch = Image.fromarray(self.img[x :x + patch_size, y : y + patch_size,:])
            transform = transforms.Compose([transforms.Resize(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                std=[0.229, 0.224, 0.225])])
            patch = transform(patch)

            if torch.cuda.is_available():
              patch = patch.cuda()
              self.model = self.model.cuda()
            self.model.eval()
            pred = torch.argmax(self.model(patch.unsqueeze(0)), axis = 1)
            prediction[x,y] = pred.item()
            
            x += 1
          y += 1
        
        vessel = np.where(prediction == 0)
        background = np.where(prediction == 1)

        prediction[vessel] = 1
        prediction[background] = 0

        return prediction

# Testing
main()