from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init
import math


#import matplotlib.pyplot as plt
import numpy as np



parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--channel_noise', type=float, default=0.1)
parser.add_argument('--intermediate_dim', type=int, default=64)
parser.add_argument('--penalty', type=float, default=1e-3)
parser.add_argument('--thr_ratio', type=float, default=1e-2)
parser.add_argument('--prune_epoch', type=int, default=100)
parser.add_argument('--model_path', type=str)

args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.f4 = gamma_layer(16,args.intermediate_dim)

        
    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x
        


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.args = args
        self.fc1 = nn.Linear(784, args.intermediate_dim)
        self.fc2 = nn.Linear(args.intermediate_dim, 1024)
        self.fc3 = nn.Linear(1024 + 16, 256)
        self.fc4 = nn.Linear(256, 10)

        self.noise_decoding = nn.Sequential(
                        nn.Linear(1,16),
                        nn.ReLU(),
                        nn.Linear(16,16),
                        nn.ReLU(),
                        nn.Linear(16,16),
                        nn.ReLU()
                        )

        self.gamma_mu = gamma_function().to(device)

        self.mask = torch.ones((args.intermediate_dim)).to(device)
        self.mask_dim = 0
        self.pruned_valued_vector = torch.zeros((args.intermediate_dim)).to(device)

        self.upper_tri_matrix = torch.triu(torch.ones((args.intermediate_dim,args.intermediate_dim))).to(device)



    def get_mask(self, mu, threshold=args.thr_ratio):

        #alpha = F.linear(torch.abs(mu), self.upper_tri_matrix)
        alpha = mu
        hard_mask = (alpha > threshold).float()
        return hard_mask

    def get_mask_test(self, channel_noise, threshold = args.thr_ratio):
        mu = self.gamma_mu(channel_noise)
        alpha = F.linear(mu, self.upper_tri_matrix)
        mu = torch.clamp(mu,min = 1e-4)
        hard_mask = (alpha > threshold).float()
        return hard_mask, alpha

    def forward(self, x, args, noise = 0.2):
        x = x.view(-1, int(x.nelement() / x.shape[0]))

        weight = self.fc1.weight
        bias = self.fc1.bias

        #bias = torch.clamp(torch.abs(bias),min = 1e-3) * torch.sign(bias.detach())
        #weight = torch.clamp(torch.abs(weight),min = 1e-3) * torch.sign(weight.detach())

        l2_norm_squared = torch.sum(weight.pow(2),dim = 1) + bias.pow(2)
        l2_norm = l2_norm_squared.pow(0.5)

        fc1_weight = (weight.permute(1,0) / l2_norm).permute(1,0)
        fc1_bias = bias / l2_norm
        x = F.linear(x, fc1_weight, fc1_bias)

        if self.training:
            channel_noise = torch.rand(1)*0.27 + 0.05 #[0.05,0.32]
        else:
            channel_noise = torch.FloatTensor([1]) * noise

        channel_noise = channel_noise.to(device)

        channel_info_decoding = self.noise_decoding(channel_noise)
        B = x.size()[0]
        expand_channela_info_dec = channel_info_decoding.expand(B,16)



        mu = self.gamma_mu(channel_noise)
        mu = F.linear(mu, self.upper_tri_matrix)
        mu = torch.clamp(mu,min = 1e-4)

        x = torch.tanh(mu * x)

        KL = self.KL_log_uniform(channel_noise**2/(x.pow(2)+1e-4))

        #add AWGN channel noise
        x = x + torch.randn_like(x) * channel_noise#args.channel_noise

        if self.training:
            pass
            #x = x * self.get_mask(mu, channel_noise)
        else:
            x = x * self.get_mask(mu)

        x = F.relu(self.fc2(x))
        x = torch.cat((x,expand_channela_info_dec),dim=1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1), KL * (0.1 / channel_noise)

    def KL_log_uniform(self,alpha_squared):
        #alpha_squared = (sigma/mu)^(2)
        k1 = 0.63576
        k2 = 1.8732
        k3 = 1.48695

        batch_size = alpha_squared.size(0)

        KL_term = k1 * F.sigmoid(k2 + k3 * torch.log(alpha_squared)) - 0.5 * F.softplus(-1 * torch.log(alpha_squared)) - k1

        return - torch.sum(KL_term) / batch_size



def test(args, model, device, test_loader,noise = 0.2):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, KL = model(data, args,noise = noise)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    #print(torch.exp(model.prior_logsigma))

    #print('---sum---')

    hard_mask, mu = model.get_mask_test(torch.FloatTensor([noise]).to(device))

    index = torch.nonzero(torch.lt(hard_mask,0.5)).squeeze(1)

    pruned_number = index.size()[0]

    print(mu)

    #print(model.gamma_mu.f1.a)

    #print(model.gamma_mu.f1.a.grad)


    print('pruned alpha number:',pruned_number)

    #mu = model.noise_to_mu(torch.FloatTensor([noise]).to(device))

    #mu = F.linear(torch.abs(mu), model.upper_tri_matrix)

    

    #print('mask_value:',model.mask_dim)



    return 100. * correct / len(test_loader.dataset), pruned_number


def main():
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net(args).to(device)
    #model.load_state_dict(torch.load('./test_20dB_IB_model.pth')['model'])
    model.load_state_dict(torch.load(args.model_path)['model'])

    avg = 0
    t = 20
    for i in range (t):
        #acc, _ = test(0,args.noise_test)
        acc, _ = test(args, model, device, test_loader, args.channel_noise)
        avg += acc
    print('noise:',args.channel_noise,'dynamic avg:',avg/t)



if __name__ == '__main__':
    main()
