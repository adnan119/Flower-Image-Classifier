import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

#train_on_gpu = torch.cuda.is_available()

def load_data(args = "./flowers" ):
    data_dir = args
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(valid_dir, transform = val_transform)
    test_data = datasets.ImageFolder(test_dir, transform = test_transform)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_data, batch_size=16)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=16)
    
    return trainloader, validloader, testloader

model_architectures = {'densenet161':2208,
                      'resnet152':2048,
                      'densenet121':1024,
                      'densenet201':1920}


def nn_classifier(model_arch = 'resnet152', hidden_units = 512, hidden_activation = nn.ReLU(), 
                  dropout = 0, output_activation = nn.LogSoftmax(dim=1),lr, train_on_gpu = True):
    if model_arch == 'densenet161':
        model = models.densenet161(pretrained=True)        
    elif model_arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif model_arch == 'resnet152':
        model = models.resnet152(pretrained = True)
    elif model_arch == 'densenet201':
        model = models.densenet201(pretrained = True)
    else:
        print("This classifier only accepts densenet121, densenet161 and resnet152 as the pretrained models")
        
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
                          #('adaptive_pool',nn.AdaptiveAvgPool2d((1,1))),
                          #('adaptive_maxpool',nn.AdaptiveMaxPool2d((1,1))),
                          #('flatten',nn.Flatten()),
                          ('batch_norm',nn.BatchNorm1d(model_architectures[model_arch],eps=1e-05, momentum=0.1, affine=True)),
                          ('Dropout',nn.Dropout(dropout)),
                          ('fc1', nn.Linear(model_architectures[model_arch], hidden_units)),
                          ('activation', hidden_activation),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', output_activation)
                          ]))
    if  model_arch == 'resnet152':
        model.fc = classifier
    else:
        model.classifier = classifier
    
    
    if train_on_gpu:
        model.cuda()
    
    criterion = nn.NLLLoss()
    if model_name == 'resnet152':
        optimizer = optim.Adam(model.fc.parameters(), lr=learn_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    lrsheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
    return model, criterion, optimizer, lrsheduler

def training_network(model, criterion, optimizer, lrsheduler, epochs = 15, loader=trainloader):
    model.to('cuda')

    for e in range(epochs):
        training_loss = 0
        validation_loss = 0
        lrsheduler.step(validation_loss)
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()*inputs.size(0)


        model.eval()
        for ii, (inputs, labels) in enumerate(validloader):

            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            validation_loss += loss.item()*inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        training_loss = training_loss/len(trainloader.dataset)
        validation_loss = validation_loss/len(validloader.dataset)
        _correct = (correct*100)/total

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tAccuracy: {:.6f}'.format(
            e, training_loss, validation_loss, _correct))

        if validation_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            validation_loss))

        #to save the check-points
            model.class_to_idx = train_data.class_to_idx
            checkpoint = {'architecture': model_arch,
                    'classifier': model.fc,
                    'class_to_idx': model.class_to_idx,
                    'state_dict': model.state_dict()}

            torch.save(checkpoint, 'model_challenge_2.pt')
            valid_loss_min = validation_loss
            
def load_checkpoint(path="model_challenge_2.pt"):
    """
    Loads deep learning model checkpoint.
    """
    
    # Load the saved file
    checkpoint = torch.load("model_challenge_2.pt")
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.fc = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    
    return model

def process_image(image):
    img_pil = Image.open(image)
   
    image_process = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = image_process(img_pil)
    
    return img_tensor

def predict(image_path, model, topk=5):   
    model.to('cuda:0')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)


ap = argparse.ArgumentParser(description='Train.py')
ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.003)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=15)
ap.add_argument('--arch', dest="arch", action="store", default="resnet152", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=512)
ap.add_argument('--train_on_gpu', default=True, action="store", dest="gpu")

pa = ap.parse_args()
args = pa.data_dir
lr = pa.learning_rate
model_arch = pa.arch
dropout = pa.dropout
hidden_units = pa.hidden_units
epochs = pa.epochs
gpu = pa.train_on_gpu

trainloader, validloader, testloader = load_data(args)


model, optimizer, criterion = nn_classifier(model_arch, hidden_units, nn.ReLU(), dropout, nn.LogSoftmax(dim=1), lr)

training_network(model, criterion, optimizer, lrsheduler, epochs, trainloader)

