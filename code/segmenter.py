## Importing Relevant Libraries
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from PIL import Image
import numpy as np
import os
from natsort import natsorted

## Define Main Function

def main():
    global root
    global patch_size
    global model
    global model_type
    
    root = os.getcwd()
    patch_size = 9
    model_type = 'classifier'
    model = model_type_input()

    os.mkdir(root + "/Mask_" + model_type)

    decodedir()

    return 

class Decoder():
    def __init__(self, image):
        global patch_size
        global root
        global model

        self.img = image
        self.model = model
    
    def classifier(self):
        w,h,c = self.img.shape
        W = w - patch_size
        H = h - patch_size
        prediction = np.zeros((W,H))
        
        y = 0
        for y in range(H):
          x = 0
          
          with open('log.txt', 'a') as lg:
            lg.write("\n  Running model on line " + str(y + 1) + "/" + str(H))
          
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

        prediction = Image.fromarray(prediction*255)

        return prediction

    def encoder(self):
      self.img = torch.from_numpy(self.img.transpose(2,0,1)).float()
      mini_batch = self.img.unsqueeze(0)
      self.model.eval()
      prediction = self.model(mini_batch)
      prediction = (nn.Sigmoid()(prediction['out'].data) >= 0.5).float()
      prediction = np.squeeze(prediction.cpu().detach().numpy())
      prediction = Image.fromarray(prediction*255)
      
      return prediction

def model_type_input():
  global model_type

  # model_type = input("Choose model type as 'classifier' or 'encoder': ")
  
  #Load trained model from the State Dict
  if model_type == 'classifier':
    model = torchvision.models.resnet101()
    model.fc = nn.Sequential(
              nn.Linear(2048, 128),
              nn.ReLU(inplace=True),
              nn.Linear(128, 2),
              nn.Softmax(1))
    
    model.load_state_dict(torch.load(root + "/state_dicts/resnet101_dict.pt"))
  
  elif model_type == 'encoder':
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    model.classifier = DeepLabHead(2048, 1)

    model.load_state_dict(torch.load(root + "/state_dicts/DLV3_dict.pt"))
  
  else:
    print('Model type incorrectly input')
    model_type_input()
  
  return model

def decodedir():
  global model_type
  global root

  image_pths = DatasetList(root + "/Image").__lst__()

  for pth in image_pths:
    name = os.path.basename(pth)

    name = os.path.splitext(name)[0]

    bsname = name.split("_", 1)[0]
    
    with open('log.txt', 'a') as lg:
      lg.write("\nOpening " + name + "\n")
    
    img = Image.open(pth).convert('RGB')
    npimg = np.array(img)
    
    with open('log.txt', 'a') as lg:
      lg.write("\nDecoding " + name + "\n")
    
    if model_type =='classifier':
      mask = Decoder(npimg).classifier()
    
    if model_type == 'encoder':
      mask = Decoder(npimg).encoder()

    with open('log.txt', 'a') as lg:
      lg.write("\nConverting mask of " + name + " to image\n")
    
    mask = mask.convert('RGB')
    
    with open('log.txt', 'a') as lg:
      lg.write("\nSaving " + bsname + "_mask as .png\n")
    
    mask.save(root + "/Mask_" + model_type + "/" + bsname + "_mask.png")

  return

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

main()