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
import copy
#import resnet
#import compression_resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--hid_dim', type=int, default=64, help='dimension of encoded vector')
parser.add_argument('--epoch', type=int, default=320, help='epoch')
parser.add_argument('--batch', type=int, default=128, help='batch size')
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
#parser.add_argument('--channel_noise', type=float, default=0.1)
#parser.add_argument('--hid_dim', type=int, default=64)
parser.add_argument('--penalty', type=float, default=1e-2)
parser.add_argument('--thr_ratio', type=float, default=1e-2)
parser.add_argument('--init_gamma', type=float, default=0)
parser.add_argument('--decay_step', type=int, default=60)
#parser.add_argument('--prune_epoch', type=int, default=100)
args = parser.parse_args()

# Device configuration†
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = args.epoch
print('-------',num_epochs)
batch_size = args.batch
learning_rate = args.lr

#nan_flag = 0

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
test_loader_this  = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

class gamma_layer(nn.Module):
    '''
    save params
    '''
    def __init__(self, input_channel, output_channel):
        super(gamma_layer, self).__init__()
        self.H = nn.Parameter(torch.ones(output_channel, input_channel))
        self.b = nn.Parameter(torch.ones(output_channel))
        #self.a = nn.Parameter(torch.ones(output_channel))

        self.H.data.normal_(0, 0.1)
        self.b.data.normal_(0, 0.001)
        #self.a.data.normal_(0, 0.01)

    def forward(self, x):
        H = torch.abs(self.H) # F.softplus()
        #H = torch.clamp(H,min = 5e-3)
        #b = torch.abs(self.b)
        #a = torch.tanh(self.a)
        #x = F.linear(x,H,b) #+ x
        x = F.linear(x,H)

        return torch.tanh(x)
        #return x + torch.tanh(x) * a

class gamma_function(nn.Module):

    def __init__(self):
        super(gamma_function, self).__init__()
        self.f1 = gamma_layer(1,16)
        self.f2 = gamma_layer(16,16)
        self.f3 = gamma_layer(16,16)
        self.f4 = gamma_layer(16,args.hid_dim)

        
    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x

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
        self.nan_flag = 0

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
                        nn.Linear(64,64),
                        nn.Sigmoid()
                        )
        #self.encoder2_2 = nn.Linear(64,self.hidden_channel)

        self.noise_encoding = nn.Sequential(
                        nn.Linear(1,16),
                        nn.ReLU(),
                        nn.Linear(16,16),
                        nn.ReLU(),
                        nn.Linear(16,16),
                        nn.ReLU()
                        )

        self.noise_decoding = nn.Sequential(
                        nn.Linear(1,16),
                        nn.ReLU(),
                        nn.Linear(16,16),
                        nn.ReLU(),
                        nn.Linear(16,16),
                        nn.ReLU()
                        )


        self.encoded_weight = nn.Parameter(torch.Tensor(self.hidden_channel, 64))
        self.encoded_bias = nn.Parameter(torch.Tensor(self.hidden_channel))
        self.encoded_weight.data.normal_(0, 0.5)
        self.encoded_bias.data.normal_(0, 0.1)

        self.decoder1 = nn.Linear(self.hidden_channel,64)
        self.decoder1_2 = nn.Sequential(
                        nn.Linear(64,64),
                        nn.ReLU()
                        )
        self.decoder1_3 = nn.Sequential(
                        nn.Linear(64+16,64),
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

        #self.posterior_mu = nn.Parameter(torch.ones(args.hid_dim))
        init_var = 0.01
        #self.posterior_mu.data.normal_(args.init_gamma, init_var)
        self.gamma_mu = gamma_function().to(device)
        self.upper_tri_matrix = torch.triu(torch.ones((args.hid_dim,args.hid_dim))).to(device)


        

    def forward(self, x, epoch, noise = 0.1):


        

        
        x = self.prep(x)
        if self.training and torch.max(torch.isnan(x)) == 1 and self.nan_flag ==0:
            print(x)
            print(torch.mean(x))
            print('prep x nan')
            self.nan_flag = 1
            print(self.prep[0].weight)
            print(self.prep[0].bias)
            print(error)

        iden = self.layer1(x)
        if self.training and torch.max(torch.isnan(iden)) == 1 and self.nan_flag ==0:
            print(iden)
            print(torch.mean(iden))
            print('layer1 iden nan')
            self.nan_flag = 1
            print(error)

        x = self.layer1_res(iden)
        if self.training and torch.max(torch.isnan(x)) == 1 and self.nan_flag ==0:
            print(x)
            print(torch.mean(x))
            print('layer1_res x nan')
            self.nan_flag = 1
            print(error)

        x = x + iden
        x = self.layer2(x)
        if self.training and torch.max(torch.isnan(x)) == 1 and self.nan_flag ==0:
            print(x)
            print(torch.mean(x))
            print('layer2 x nan')
            self.nan_flag = 1
            print(error)

        x = self.layer3(x)
        if self.training and torch.max(torch.isnan(x)) == 1 and self.nan_flag ==0:
            print(x)
            print(torch.mean(x))
            print('layer3 x nan')
            self.nan_flag = 1
            print(error)

        B = x.size()[0]
        x = self.encoder1(x)
        if self.training and torch.max(torch.isnan(x)) == 1 and self.nan_flag ==0:
            print(x)
            print(torch.mean(x))
            print('encoder1 x nan')
            self.nan_flag = 1
            print(error)

        x = torch.reshape(x,(B,4*4*4))

        if self.training:
            channel_noise = torch.rand(1)*0.27 + 0.05 #[0.04,0.46]
            #noise_list = [ 0.3162, 0.22387, 0.1778, 0.1259, 0.1, 0.0709, 0.056 ]
            #m = torch.distributions.categorical.Categorical(torch.tensor([ 1/7.0,1/7.0,1/7.0,1/7.0,1/7.0,1/7.0,1/7.0 ]))
            #index = m.sample()
            #channel_noise = noise_list[index] * torch.FloatTensor([1])

        else:
            channel_noise = torch.FloatTensor([1]) * noise
        channel_noise = channel_noise.to(device)

        #channel_info_encoding = self.noise_encoding(channel_noise)
        channel_info_decoding = self.noise_decoding(channel_noise)

        B = x.size()[0]

        #expand_channela_info_enc = channel_info_encoding.expand(B,16)
        expand_channela_info_dec = channel_info_decoding.expand(B,16)

        #print(x.size(),expand_channela_info_enc.size())

        #x = torch.cat((x,expand_channela_info_enc),dim = 1)


        x = self.encoder2(x)
        if self.training and torch.max(torch.isnan(x)) == 1 and self.nan_flag ==0:
            print(x)
            print(torch.mean(x))
            print('encoder2 x nan')
            self.nan_flag = 1
            print(error)

        # activation normalize
        x_norm2 = torch.norm(x,dim=1)
        x = 64 * (x.permute(1,0)/(x_norm2+1e-6)).permute(1,0) 
        if self.training and torch.max(torch.isnan(x)) == 1 and self.nan_flag ==0:
            print(x)
            print(torch.mean(x))
            #print(x_norm2)
            print('normalize x nan')
            self.nan_flag = 1
            print(error)
            
        #weight normalize
        weight2_2 = F.tanh(self.encoded_weight)
        bias2_2 = F.tanh(self.encoded_bias)
        weight2_2 = torch.clamp(torch.abs(weight2_2),min = 1e-3) * torch.sign(weight2_2.detach())
        bias2_2 = torch.clamp(torch.abs(bias2_2),min = 1e-3) * torch.sign(bias2_2.detach())

        #weight2_2 = (weight2_2_clamp - weight2_2).detach() + weight2_2

        l2_norm_squared = torch.sum(weight2_2.pow(2),dim = 1) + bias2_2.pow(2)
        l2_norm = l2_norm_squared.pow(0.5)
        if torch.min(l2_norm) < 1e-6:
            print(l2_norm)
        encoder_weight = (weight2_2.permute(1,0) / (l2_norm+1e-6)).permute(1,0)
        encoder_bias = bias2_2 / (l2_norm+1e-6)
        x = F.linear(x, encoder_weight, encoder_bias)
        if self.training and torch.max(torch.isnan(x)) == 1 and self.nan_flag ==0:
            print(x)
            print('encoded feature x nan')
            print(l2_norm)
            self.nan_flag = 1
            print(error)

        #channel noise and mu
        
        mu = self.gamma_mu(channel_noise)
        if self.training and torch.max(torch.isnan(mu)) == 1 and self.nan_flag ==0:
            print(mu)
            print('mu is nan')
            self.nan_flag = 1
            print(error)

        mu = F.linear(mu, self.upper_tri_matrix)
        mu = torch.clamp(mu,min = 1e-4)
        encoded_feature = torch.tanh(x * mu)
        clamp_feature = torch.clamp(torch.abs(encoded_feature),min = 1e-2) * torch.sign(encoded_feature.detach())
        #raw_x_mask = self.get_mask(torch.abs(encoded_feature),threshold = 1e-2)
        

        # KL divergence
        KL = self.KL_log_uniform(channel_noise,torch.abs(clamp_feature))
        if self.training and torch.max(torch.isnan(KL)) == 1 and self.nan_flag == 0:
            print(KL)
            print('KL is nan')
            self.nan_flag = 1
            print(error)

        x = clamp_feature + torch.randn_like(clamp_feature) * channel_noise

        if self.training:
            if epoch > 60:
                x = x * self.get_mask(mu,threshold = args.thr_ratio)
            pass
            #x = x * raw_x_mask
        else:
            x = x * self.get_mask(mu,threshold = args.thr_ratio)

        x = F.relu(self.decoder1(x))

        

        x = self.decoder1_2(x)
        x = torch.cat((x,expand_channela_info_dec),dim=1)
        x = self.decoder1_3(x)
        x = torch.reshape(x,(B,4,4,4))
        decoded_feature = self.decoder2(x)
        x = self.layer3_res(decoded_feature)
        x = x + decoded_feature
        x = self.classifier1(x)
        output = self.classifier2(x)

        
        return output, KL * 0.1 / channel_noise

    def KL_log_uniform(self,channel_noise,encoded_feature):
        #alpha_squared = (sigma/mu)^(2)
        alpha = (channel_noise/encoded_feature)
        #print(torch.min(alpha),torch.min(torch.log(alpha)))
        k1 = 0.63576
        k2 = 1.8732
        k3 = 1.48695
        batch_size = alpha.size(0)
        KL_term = k1 * F.sigmoid(k2 + k3 * 2 * torch.log(alpha)) - 0.5 * F.softplus(-2 * torch.log(alpha)) - k1
        return - torch.sum(KL_term) / batch_size

    def get_mask(self, mu, threshold=args.thr_ratio):

        #alpha = F.linear(torch.abs(mu), self.upper_tri_matrix)
        alpha = mu.detach()
        hard_mask = (alpha > threshold).float()
        return hard_mask

    def get_mask_test(self, channel_noise, threshold = args.thr_ratio):
        mu = self.gamma_mu(channel_noise)
        
        #mu = F.softplus(self.gamma_mu(channel_noise)-3)
        #mu = F.relu(self.gamma_mu(channel_noise)+0.1)
        alpha = F.linear(mu, self.upper_tri_matrix)
        mu = torch.clamp(mu,min = 1e-4)
        hard_mask = (alpha > threshold).float()
        return hard_mask, alpha

#model = compression_resnet.split_resnet4(args).to(device)
model = split_resnet4(args).to(device)

#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)

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
    last_param = []
    last_param_grad = []

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

            #if KL > 300:
                #print(KL)

            if i == 0:
                parameters = model.parameters()
                if isinstance(parameters, torch.Tensor):
                    print('true')
                    parameters = [parameters]
                parameters = [p for p in parameters if p.grad is not None]
                max_mag = max(p.grad.detach().abs().max().to(device) for p in parameters)
                total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2)
                nan_value = torch.max(torch.stack([torch.any(torch.isnan(p.grad.detach())).to(device) for p in parameters]))
                nan_weight_value = torch.max(torch.stack([torch.max(torch.isnan(p.detach())).to(device) for p in parameters]))
                max_weight = max(p.detach().abs().max().to(device) for p in parameters)
                min_weight = min(p.detach().abs().max().to(device) for p in parameters)
                #print('max magnitude:{:.4f}'.format(max_mag.item()),'l2 norm:{:.4f}'.format(total_norm.item()),'grad_nan',nan_value.item(),'weight_nan',nan_weight_value.item(),'weight_max:{:.4f}_min:{:.10f}'.format(max_weight,min_weight))
                print('KL:{:.4f}'.format(KL.item()),'max magnitude:{:.4f}'.format(max_mag.item()),'l2 norm:{:.4f}'.format(total_norm.item()),'grad_nan',nan_value.item(),'weight_nan',nan_weight_value.item(),'weight_max:{:.4f}_min:{:.10f}'.format(max_weight,min_weight))
                #print('encoded weight:{:.4f}_{:.4f}'.format(torch.max(torch.abs(model.encoded_weight)).item(),torch.max(torch.abs(model.encoded_bias)).item()),'grad max_{:.4f}_{:.4f}'.format(torch.max(torch.abs(model.encoded_weight.grad)).item(),torch.max(torch.abs(model.encoded_bias.grad)).item()))
                #print('prep weight',torch.norm(model.prep[0].weight))

            #torch.nn.utils.clip_grad_norm_(model.parameters(),0.4)
            
            #torch.nn.utils.clip_grad_value_(model.encoded_weight,0.1)
            #torch.nn.utils.clip_grad_value_(model.encoded_bias,0.1)

            optimizer.step()
            
            parameters = model.parameters()
            parameters = [p for p in parameters if p.grad is not None]
            nan_weight_value = torch.max(torch.stack([torch.max(torch.isnan(p.detach())).to(device) for p in parameters]))
            #print(nan_weight_value)
            
            #last_param_grad = 
            #for i, p in enumerate(parameters):
            #    print(i,p.size(),torch.max(torch.isnan(p)).item(),torch.max(torch.abs(p)).item(),torch.min(torch.abs(p)).item())

            if nan_weight_value == True:
                for i, p in enumerate(parameters):
                    print(i,p.size(),torch.max(torch.isnan(p)).item(),torch.max(torch.abs(p)).item(),torch.min(torch.abs(p)).item())
                    print(i,torch.max(torch.isnan(p.grad)),torch.max(torch.abs(p.grad)))
                print('------last batch--------')
                for i, p in enumerate(last_param):
                    print(i,p.size(),torch.max(torch.isnan(p)).item(),torch.max(torch.abs(p)).item(),torch.min(torch.abs(p)).item())

            last_param = copy.deepcopy(parameters)
        scheduler.step()
        accuracy_result = accuracy(output,y)


        acc, pruned_number = test(epoch,noise = 0.1)
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
    torch.save({'model': saved_model2}, './model_dynamic_vfe/changing_beta_dynamic_hid:{}_remain:{}_penalty:{}_acc:{}_epoch:{}_model2.pth'.format(args.hid_dim,args.hid_dim - pruned_number2,args.penalty,test_acc2,args.epoch))
    torch.save({'model': saved_model1}, './model_dynamic_vfe/changing_beta_dynamic_hid:{}_remain:{}_penalty:{}_acc:{}_epoch:{}_model1.pth'.format(args.hid_dim,args.hid_dim - prune_dim,args.penalty,test_acc,args.epoch))

def test(epoch,noise=0.1):
    with torch.no_grad():

        #print(model.parameters())
        
        model.eval()

        correct = 0
        correct_top5 = 0#top5
        total = 0

        for i, (images, labels) in enumerate(test_loader_this): 
            images = images.to(device)
            labels = labels.to(device)
            outputs,_= model(images,epoch,noise)
            maxk = max((1,5))
            labels_relize = labels.view(-1,1)
            _, top5_pred = outputs.topk(maxk, 1, True, True)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_top5 +=torch.eq(top5_pred, labels_relize).sum().float().item()
            correct += (predicted == labels).sum().item()
                            
        #index = torch.nonzero(torch.lt(model.get_mask(args.thr_ratio),0.5)).squeeze(1)
        #index = torch.nonzero(torch.lt(model.posterior_mu,threshold)).squeeze(1)
        #pruned_number = index.size()[0]
        #print(model.posterior_mu)
        #print('pruned alpha number:',pruned_number)
        #print('mu_mask:',model.get_mask(1.5))

        hard_mask, mu = model.get_mask_test(torch.FloatTensor([noise]).to(device))

        index = torch.nonzero(torch.lt(hard_mask,0.5)).squeeze(1)

        pruned_number = index.size()[0]

        #print(list(model.gamma_mu.named_parameters())[0])

        print(mu)

        print('pruned alpha number:',pruned_number)

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

        return 100 * correct / total, pruned_number




      
if __name__=='__main__':
    train()
