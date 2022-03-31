import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as funct
import numpy as np


class conv_network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.device='cuda'
        self.model_path='leaf_model.pth'
        
        self.conv_one=torch.nn.Conv2d(1,12,2).to(self.device)
        self.conv_two=torch.nn.Conv2d(12,24,2).to(self.device)
        self.conv_three=torch.nn.Conv2d(24,12,2).to(self.device)
        # self.conv_four=torch.nn.Conv2d(12,10,2).to(self.device)
        
        self.test_input=torch.zeros(12,1,28,28).to(self.device)
        self.line_size=self.convs(self.test_input)
        
        self.line_one=nn.Linear(self.line_size,512).to(self.device)
        self.line_two=nn.Linear(512,10).to(self.device)
        
        if self.device=='cuda':
            self.test_input.to('cpu')
            del self.test_input
            torch.cuda.empty_cache()
        
        
        self.optimizer=optim.Adam(self.parameters(),lr=.0001)
        self.criterion=nn.MSELoss() #.to(self.device)
        
    # def load_model(self):
    #     torch.load(self.model_path)
        
    
    def convs(self,x):
        x=funct.max_pool2d(funct.relu(self.conv_one(x)),(2,2))
        x=funct.max_pool2d(funct.relu(self.conv_two(x)),(2,2))
        x=funct.max_pool2d(funct.relu(self.conv_three(x)),(2,2))
        # x=funct.max_pool2d(funct.relu(self.conv_four(x)),(2,2))
        
        
        size=x.shape
        total=1
        
        for s in size:
            total*=s
        
        self.zero_grad()
        return total
        


    def forward(self,x):
        x=funct.max_pool2d(funct.relu(self.conv_one(x)),(2,2))
        x=funct.max_pool2d(funct.relu(self.conv_two(x)),(2,2))
        x=funct.max_pool2d(funct.relu(self.conv_three(x)),(2,2))
        # x=funct.max_pool2d(funct.relu(self.conv_four(x)),(2,2))
        
        x=torch.flatten(x)
        x=funct.relu(self.line_one(x))
        x=self.line_two(x)
        x=funct.softmax(x,dim=0)
        return x
    
    
    def test_on_image(self,image,target):
        
        init_fet=12
        new_shape=[init_fet]+[i for i in image.shape]
        net_input=np.zeros(new_shape)
        
        for i in range(init_fet):
            net_input[i]=image
            
        image=torch.FloatTensor(net_input).to(self.device)
        target=torch.FloatTensor(target).to(self.device)
        
        look=self.forward(image)
        # print(look)
        return torch.argmax(look),torch.argmax(target)
        
    
    
    def train_on_image(self,image,target):
        
        init_fet=12
        new_shape=[init_fet]+[i for i in image.shape]
        net_input=np.zeros(new_shape)
        
        for i in range(init_fet):
            net_input[i]=image
            
        image=torch.FloatTensor(net_input).to(self.device)
        target=torch.FloatTensor(target).to(self.device)
        
        self.zero_grad()
        
        look=self.forward(image)
        loss=self.criterion(look,target)
        # print(torch.argmax(look).item(),torch.argmax(target).item(),loss)
        
        loss.backward()
        self.optimizer.step()
        
        
    def toggle_device(self):
        
        if self.device=='cpu':
            self.device='cuda'
            
            self.conv_one.to('cuda')
            self.conv_two.to('cuda')
            self.conv_three.to('cuda')
            self.conv_four.to('cuda')
            self.line_one.to('cuda')
            self.line_two.to('cuda')
            
            
            
            
        
        elif self.device=='cuda':
            self.device='cpu'
        
            self.conv_one.to('cpu')
            self.conv_two.to('cpu')
            self.conv_three.to('cpu')
            self.conv_four.to('cpu')
            self.line_one.to('cpu')
            self.line_two.to('cpu')
        
            torch.cuda.empty_cache()
        
        
        
        # self.criterion.to(self.device)
        
    
    
    def load_model(self):
        dict=torch.load('hand_digit.pth')
        self.load_state_dict(dict)
        self.eval()
        
        
    def save(self):
        torch.save(self.state_dict(),'hand_digit.pth')
        
        
    
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        