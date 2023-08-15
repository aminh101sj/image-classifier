import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

import json

from PIL import Image
import numpy as np

class Machine(object):

    def __load_training_set(self, data_dir):
        self.train_dir = data_dir + '/train'
        self.valid_dir = data_dir + '/valid'
        self.test_dir = data_dir + '/test'    


        self.train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        self.valid_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

        self.test_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])


        # TODO: Load the datasets with ImageFolder
        self.train_datasets = datasets.ImageFolder(self.train_dir, transform=self.train_transforms)
        self.valid_datasets = datasets.ImageFolder(self.valid_dir, transform=self.valid_transforms)
        self.test_datasets = datasets.ImageFolder(self.test_dir, transform=self.test_transforms)

        # TODO: Using the image datasets and the trainforms, define the dataloaders
        self.trainloader = torch.utils.data.DataLoader(self.train_datasets, batch_size=64, shuffle=True)
        self.validloader = torch.utils.data.DataLoader(self.valid_datasets, batch_size=64)
        self.testloader = torch.utils.data.DataLoader(self.test_datasets, batch_size=64)

        self.class_to_idx = self.train_datasets.class_to_idx

    def __load_cat_map(self):
        with open('cat_to_name.json', 'r') as f:
            self.cat_to_name = json.load(f)

    def __load_model(self):

        if self.arch == 'densenet121': 
            self.model = models.densenet121(pretrained=True)
            self.input_n = 1024
            self.fc1_n = 500
        elif self.arch == 'vgg13':
            self.model = models.vgg13(pretrained=True)
            self.input_n = 25088
            self.fc1_n = 4096
        else:
            print("Error: Unsupported Model. Using Densenet121")
            self.model = models.densenet121(pretrained=True)
            self.input_n = 1024
            self.fc1_n = 500


        # Freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False

        self.output_n = 102
        classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(self.input_n, self.fc1_n)),
                                    ('relu', nn.ReLU()),
                                    ('dropout',nn.Dropout(0.3)),
                                    ('fc2', nn.Linear(self.fc1_n, self.hidden_units)),
                                    ('relu', nn.ReLU()),
                                    ('dropout',nn.Dropout(0.3)),
                                    ('fc3', nn.Linear(self.hidden_units, self.output_n)),
                                    ('output', nn.LogSoftmax(dim=1))
                                    ]))

        self.model.classifier = classifier

        self.criterion = nn.NLLLoss()

        self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")

        # Set model to use cpu or gpu
        self.model.to(self.device)

        # Set Optimizer 
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)


    def __init__(self, data_dir='flowers', save_dir='.', epochs=5, arch='densenet121', hidden_units=250, learning_rate=0.003, gpu=None):
        self.epochs = epochs
        self.arch = arch
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.gpu = gpu
        self.save_dir = save_dir

        self.__load_training_set(data_dir)
        self.__load_cat_map()      
        self.__load_model()  


    def train(self):
        running_loss = 0

        self.train_losses, self.test_losses = [], []

        for epoch in range(self.epochs):
            running_loss = 0
            
            for inputs, labels in self.trainloader:
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                logps = self.model.forward(inputs)
                loss = self.criterion(logps, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                        
            else:
                
                test_loss = 0
                accuracy = 0
                self.model.eval()
                with torch.no_grad():
                    for inputs, labels in self.validloader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        logps = self.model.forward(inputs)
                        batch_loss = self.criterion(logps, labels)
                            
                        test_loss += batch_loss.item()
                            
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                        
                self.train_losses.append(running_loss/len(self.trainloader))
                self.test_losses.append(test_loss/len(self.testloader))
                
                print(f"Epoch {epoch+1}/{self.epochs}.. "
                        f"Train loss: {running_loss/len(self.trainloader):.3f}.. "
                        f"Validation loss: {test_loss/len(self.validloader):.3f}.. "
                        f"Validation accuracy: {accuracy/len(self.validloader):.3f}")
                    
                self.model.train()

    def plot(self):
        plt.plot(self.train_losses, label='Training loss')
        plt.plot(self.test_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()

    def test(self):
        test_loss = 0
        test_accuracy = 0
        self.model.eval()

        with torch.no_grad(): 
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logps = self.model.forward(inputs)
                batch_loss = self.criterion(logps, labels)
                            
                test_loss += batch_loss.item()
                            
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                        
                
            print(f"Test accuracy: {test_accuracy/len(self.testloader)*100}%")
                    
        self.model.train()

    def save_chkpth(self):
        checkpoint = {'classifier': {
                'fc1': { 'input': self.input_n, 'output': self.fc1_n },
                'fc2': { 'input': self.fc1_n, 'output': self.hidden_units},
                'fc3': { 'input': self.hidden_units, 'output': self.output_n}
              },
              'arch': self.arch,
              'learning_rate': self.learning_rate,
              'state_dict': self.model.state_dict(),
              'cat_to_name': self.cat_to_name,
              'epochs': self.epochs,
              'optimizer_dict': self.optimizer.state_dict(),
              'class_to_idx': self.class_to_idx
             }

        torch.save(checkpoint, self.save_dir + '/checkpoint-cmd.pth')


    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)

        self.arch = checkpoint['arch']
        print('the arch: ', self.arch)
        if self.arch == 'densenet121': 
            self.model = models.densenet121(pretrained=True)
        elif self.arch == 'vgg13':
            self.model = models.vgg13(pretrained=True)
        else:
            print("Error: Unsupported Model. Using Densenet121")
            self.model = models.densenet121(pretrained=True)
        
        # Freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False
        
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(checkpoint['classifier']['fc1']['input'], checkpoint['classifier']['fc1']['output'])),
                            ('relu', nn.ReLU()),
                            ('dropout',nn.Dropout(0.3)),
                            ('fc2', nn.Linear(checkpoint['classifier']['fc2']['input'], checkpoint['classifier']['fc2']['output'])),
                            ('relu', nn.ReLU()),
                            ('dropout',nn.Dropout(0.3)),
                            ('fc3', nn.Linear(checkpoint['classifier']['fc3']['input'], checkpoint['classifier']['fc3']['output'])),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

        self.model.classifier = classifier
        
        self.model.load_state_dict(checkpoint['state_dict'])
        
        self.arch = checkpoint['arch']
        self.learning_rate = checkpoint['learning_rate']
        self.cat_to_name = checkpoint['cat_to_name']
        self.epochs = checkpoint['epochs']
        
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=0.003)
        self.optimizer.load_state_dict(state_dict=checkpoint['optimizer_dict'])
        
        self.class_to_idx = checkpoint['class_to_idx']
        
        
    def process_image(self,image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        with Image.open(image) as im:
            # shortest side is 256 pixels and keep ratio
            if im.width > im.height:
                (width, height) = (int((im.width*256)/im.height), 256)
            else:
                (width, height) = (256, int((im.height*256)/im.width))
            

            im_resized = im.resize((width, height))
            
            # calculate crop
            left = (width-244)/2
            upper = (height-244)/2
            right = (width+244)/2
            lower = (height+244)/2
            
            im_crop = im_resized
            im_crop = im_resized.crop((left, upper, right, lower))
            
            # Normalize
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            
            np_image = np.array(im_crop)/255
            
            np_image = (np_image - mean)/std
            
            # Fix color channels
            np_image = np_image.transpose(2,0,1)
            
        return np_image

    def imshow(self, image, ax=None, title=None):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
        
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.transpose((1, 2, 0))
        
        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        
        plt.show()

        return ax

    def predict(self, image_path, topk=5, gpu=None):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        
        # TODO: Implement the code to predict the class from an image file
        device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        
        image = torch.from_numpy(self.process_image(image_path))
        image = torch.unsqueeze(image,0).to(device).float()
        
        self.model.to(device)

        self.model.eval()
        logps = self.model.forward(image)
        
        
        ps = torch.exp(logps)
        
        top_ps, top_idx = ps.topk(k=topk, dim=1)

        list_ps = top_ps.tolist()[0]
        list_idx = top_idx.tolist()[0]
        classes = []
        self.model.train()
        
        index_to_class = dict(map(reversed, self.class_to_idx.items()))

        for x in list_idx:
            classes.append(index_to_class[x])
        
        return list_ps, classes



def main(type):
    if type == 'test':
        m = Machine()
        print("Loading...")
        m.load_checkpoint('checkpoint-cmd.pth')
        print("Show image...")
        m.imshow(m.process_image('./flowers/test/1/image_06743.jpg'))
        print("Predicting...")
        p,c = m.predict('./flowers/test/1/image_06743.jpg')
        print(p,c)
        m.imshow(m.process_image('./flowers/test/2/image_05133.jpg'))
        p,c = m.predict('./flowers/test/2/image_05133.jpg')
        print(p,c)

        fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

        ax1.axis('off')

        labels = []
        for idx in c:
            labels.append(m.cat_to_name[idx])

        ax2.barh(labels, p)
        ax2.set_yticks(labels)
        plt.show()

    else:
        m = Machine(arch='densenet112', epochs=5, hidden_units=200)
        print('Training...')
        m.train()
        print("Plotting...")
        m.plot()
        print('Testing...')
        m.test()
        print('Saving...')
        m.save_chkpth()


if __name__ == '__main__': main('test')