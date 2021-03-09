import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
import numpy as np
import argparse
import random
#import resnet
#import compression_resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--hid_dim', type=int, default=32, help='dimension of encoded vector')
parser.add_argument('--epoch', type=int, default=200, help='epoch')
parser.add_argument('--batch', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='leaerning rate')
#parser.add_argument('-split', type = str, default = '4', help='1,2,3,4,5')
#parser.add_argument('-load', type=str)
#parser.add_argument('-bit', type=int, default=8, help='bit_num')

#parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                   # help='input batch size for training (default: 64)')
#parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                   # help='input batch size for testing (default: 1000)')
#parser.add_argument('--epochs', type=int, default=80, metavar='N',
#                    help='number of epochs to train (default: 14)')
#parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    #help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
#parser.add_argument('--no-cuda', action='store_true', default=False,
#                    help='disables CUDA training')
parser.add_argument('--channel_noise', type=float, default=0.1)
#parser.add_argument('--intermediate_dim', type=int, default=64)
parser.add_argument('--penalty', type=float, default=1e-2)
parser.add_argument('--thr_ratio', type=float, default=1e-2)
parser.add_argument('--init_gamma', type=float, default=0)
#parser.add_argument('--prune_epoch', type=int, default=100)
args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = args.epoch
print('-------',num_epochs)
batch_size = args.batch
learning_rate = args.lr

#random.seed(1)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform_test)
test_loader_this  = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x*self.weight

class split_resnet4(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.args = args
        self.hidden_channel = args.hid_dim

        self.prep = nn.Sequential(
                    nn.Conv2d(3,64,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(64),
                    nn.ReLU()
                    )
        self.layer1 = nn.Sequential(
                    nn.Conv2d(64,128,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
                    )
        self.layer1_res = nn.Sequential(
                    nn.Conv2d(128,128,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128,128,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(128),
                    nn.ReLU()
                    )
        self.layer2 = nn.Sequential(
                    nn.Conv2d(128,256,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2)
                    )
        self.layer3 = nn.Sequential(
                    nn.Conv2d(256,512,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
                    )

        self.layer3_res = nn.Sequential(
                    nn.Conv2d(512,512,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Conv2d(512,512,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(512),
                    nn.ReLU()
                    )
        self.classifier1 = nn.Sequential(
                    nn.MaxPool2d(kernel_size = 4, stride = 4, padding = 0, dilation = 1, ceil_mode = False),
                    Flatten()
                    )
        self.classifier2 = nn.Sequential(
                    nn.Linear(512,10,bias = False),
                    Mul(0.125)
                    )
        self.args = args

        self.hidden_channel = args.hid_dim

        self.encoder1 = nn.Sequential(
                        nn.Conv2d(512,4,kernel_size = 3,stride = 1, padding = 1, bias = False),
                        nn.BatchNorm2d(4),
                        nn.ReLU()
                        )
        self.encoder2 = nn.Sequential(
                        nn.Linear(64,64)
                        )
        self.encoder2_2 = nn.Linear(64,self.hidden_channel)

        self.decoder1 = nn.Linear(self.hidden_channel,64)
        self.decoder1_2 = nn.Sequential(
                        nn.Linear(64,64),
                        nn.ReLU()
                        )
        self.decoder2 = nn.Sequential(
                        nn.Conv2d(4,512,kernel_size = 3,stride = 1, padding = 1, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU()
                        )

        self.Tanh = nn.Tanh()



        self.mask = torch.ones((args.hid_dim)).to(device)
        self.mask_dim = 0
        self.pruned_valued_vector = torch.zeros((args.hid_dim)).to(device)

        self.posterior_mu = nn.Parameter(torch.ones(args.hid_dim))
        init_var = 0.01
        self.posterior_mu.data.normal_(args.init_gamma, init_var)
        #self.posterior_mu.data[-1].normal_(1, init_var)
        #self.posterior_mu.requires_grad = False

        

    def forward(self, x, epoch):
        
        x = self.prep(x)
        iden = self.layer1(x)
        x = self.layer1_res(iden)
        x = x + iden
        x = self.layer2(x)
        x = self.layer3(x)

        #------------------
        

        #B = x.size()[0]
        #x = torch.reshape(x,(B,-1))

        B = x.size()[0]
        x = self.encoder1(x)
        x = torch.reshape(x,(B,4*4*4))
        x = self.encoder2(x)
        # normalized R^(n) -> R^(n-1)
        x_norm2 = torch.norm(x,dim=1)
        x = 64 * (x.permute(1,0)/(x_norm2+1e-12)).permute(1,0) 

        l2_norm_squared = torch.sum(self.encoder2_2.weight.pow(2),dim = 1) + self.encoder2_2.bias.pow(2)
        l2_norm = l2_norm_squared.pow(0.5)
        encoder_weight = (self.encoder2_2.weight.permute(1,0) / (l2_norm+1e-12)).permute(1,0)
        encoder_bias = self.encoder2_2.bias / (l2_norm+1e-12)

        #x = torch.tanh(x) # constrain the amplitude from the last layer

        x = F.linear(x, encoder_weight, encoder_bias)

        encoded_feature = torch.tanh(x * self.posterior_mu)
        #if epoch == 5 or epoch ==6:
        

        clamp_feature = torch.clamp(torch.abs(encoded_feature),min = 1e-4) * torch.sign(encoded_feature.detach())

        KL = self.KL_log_uniform(self.args.channel_noise**2/(clamp_feature.pow(2)))
        x = clamp_feature + torch.randn_like(clamp_feature) * self.args.channel_noise

        #print('min',torch.min(torch.abs(encoded_feature)).item(),torch.min(torch.abs(clamp_feature)).item())

        if self.training:
            pass
            #if epoch > 90:
            #    x = x * self.get_mask(self.args.thr_ratio)
            #else:
            #    pass
        else:
            x = x * self.get_mask(self.args.thr_ratio)

        x = F.relu(self.decoder1(x))
        x = self.decoder1_2(x)
        x = torch.reshape(x,(B,4,4,4))
        decoded_feature = self.decoder2(x)


        #-------------------
        x = self.layer3_res(decoded_feature)
        x = x + decoded_feature
        x = self.classifier1(x)
        output = self.classifier2(x)

        
        return output, KL

    def KL_log_uniform(self,alpha_squared):
        #alpha_squared = (sigma/mu)^(2)
        k1 = 0.63576
        k2 = 1.8732
        k3 = 1.48695

        batch_size = alpha_squared.size(0)

        KL_term = k1 * F.sigmoid(k2 + k3 * torch.log(alpha_squared)) - 0.5 * F.softplus(-1 * torch.log(alpha_squared)) - k1

        return - torch.sum(KL_term) / batch_size

    def get_mask(self, threshold):

        #alpha = F.linear(torch.abs(self.posterior_mu), self.upper_tri_matrix)
        alpha = torch.abs(self.posterior_mu.detach())
        hard_mask = (alpha > threshold).float()
        return hard_mask

#model = compression_resnet.split_resnet4(args).to(device)
model = split_resnet4(args).to(device)

#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=30, gamma=args.gamma)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)  
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# Start training
def train(model=model):
    
    #flag = 0
    test_acc = 0
    #save_mask_dim = 0
    prune_dim = 0
    test_acc2 = 0
    pruned_number2 = 0
    saved_model1 = {}
    saved_model2 = {}

    for epoch in range(num_epochs):

        print('----epoch:{}----'.format(epoch))
        if (epoch)%10 == 0:
            data_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            model.train()
            output, KL = model(x,epoch)
            criterion = nn.CrossEntropyLoss()
            criterion = criterion.to(device)
            loss1 = criterion(output, y)
            if epoch <= 20:
                loss = loss1
            else:
                anneal_ratio = min(1,(epoch - 20)/20)
                loss = loss1 + args.penalty * KL * anneal_ratio
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.posterior_mu.data = torch.clamp(torch.abs(model.posterior_mu.data),min = 1e-4) * torch.sign(model.posterior_mu.data)

            #print('-----------')
            #model.posterior_mu[0].grad.data = 1
            #print('grad',model.posterior_mu.grad)
            #print('mu',model.posterior_mu)




            #print('mu_grad',model.posterior_mu.grad.data)
            #print(model.posterior_mu.grad.data[0])
            #if model.posterior_mu.grad.data[0] == 0:
            #    pass
            #else:
            #    print('mu != 0')
            #model.posterior_mu[0].grad = None

            '''

            for param_grad in optimizer.param_groups[0]["params"]:
                if "posterior_mu" in param_grad:
                    print(param_grad)

            '''

            





        scheduler.step()
        accuracy_result = accuracy(output,y)
        #print('mu_grad',model.posterior_mu.grad)
        #print('mu_grad',model.posterior_mu.grad)
        
        #print ("Epoch[{}/{}], Step [{}/{}], train Loss: {:.4f}, acc: {:.4f}".format(epoch+1, num_epochs, i+1, len(data_loader), loss.item(), accuracy_result.item()))

        acc, pruned_number = test(epoch)
        if epoch > 150:
            if acc > test_acc2:
                test_acc2 = acc
                pruned_number2 = pruned_number
                saved_model2 = model.state_dict()
                #torch.save({'model': model.state_dict()}, './model_vfe/PSNR:{}_hid{}_penalty_{}_epoch:{}.pth'.format(args.channel_noise,args.hid_dim,args.penalty,args.epoch))

        if epoch > 60:
            if (acc > test_acc and pruned_number == prune_dim) or pruned_number > prune_dim:
                test_acc = acc
                prune_dim = pruned_number
                saved_model1 = model.state_dict()
                #torch.save({'model': model.state_dict()}, './model_vfe/PSNR:{}_hid{}_penalty_{}_epoch:{}.pth'.format(args.channel_noise,args.hid_dim,args.penalty,args.epoch))
    

    print()
    print('best acc:',test_acc,'number:',args.hid_dim,'pruned_number',prune_dim,'beta:',args.penalty,'threshold:',args.thr_ratio)
    open('./result_vfe/PSNR:{}_hid:{}_remain:{}_penalty:{}_acc:{}_epoch:{}.txt'.format(args.channel_noise,args.hid_dim,args.hid_dim - prune_dim,args.penalty,test_acc,args.epoch),'w+')
    
    #torch.save({'model': saved_model2}, './model_vfe/PSNR:{}_hid{}_penalty:{}_epoch:{}_model2.pth'.format(args.channel_noise,args.hid_dim,args.penalty,args.epoch))

    torch.save({'model': saved_model2}, './model_vfe/PSNR:{}_hid:{}_remain:{}_penalty:{}_acc:{}_epoch:{}_model2.pth'.format(args.channel_noise,args.hid_dim,args.hid_dim - pruned_number2,args.penalty,test_acc2,args.epoch))
    torch.save({'model': saved_model1}, './model_vfe/PSNR:{}_hid:{}_remain:{}_penalty:{}_acc:{}_epoch:{}_model1.pth'.format(args.channel_noise,args.hid_dim,args.hid_dim - prune_dim,args.penalty,test_acc,args.epoch))

def test(epoch):
    with torch.no_grad():
        
        model.eval()

        correct = 0
        correct_top5 = 0#top5
        total = 0

        for i, (images, labels) in enumerate(test_loader_this): 
            images = images.to(device)
            labels = labels.to(device)
            outputs,_= model(images,epoch)
            maxk = max((1,5))
            labels_relize = labels.view(-1,1)
            _, top5_pred = outputs.topk(maxk, 1, True, True)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_top5 +=torch.eq(top5_pred, labels_relize).sum().float().item()
            correct += (predicted == labels).sum().item()
                            
        index = torch.nonzero(torch.lt(model.get_mask(args.thr_ratio),0.5)).squeeze(1)
        #index = torch.nonzero(torch.lt(model.posterior_mu,threshold)).squeeze(1)
        pruned_number = index.size()[0]
        print(model.posterior_mu)
        print('pruned alpha number:',pruned_number)
        #print('mu_mask:',model.get_mask(1.5))
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

        return 100 * correct / total, pruned_number




      
if __name__=='__main__':
    train()
