from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


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
parser.add_argument('--thr_ratio', type=float, default=5e-2)
parser.add_argument('--prune_epoch', type=int, default=60)

args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.args = args
        self.fc1 = nn.Linear(784, args.intermediate_dim)
        self.fc2 = nn.Linear(args.intermediate_dim, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 10)

        self.posterior_mu = nn.Parameter(torch.ones(args.intermediate_dim))
        #self.posterior_logsigma = nn.Parameter(torch.ones(args.intermediate_dim))
        #self.prior_logsigma = nn.Parameter(torch.ones(args.intermediate_dim) * (np.log(args.channel_noise))) # variational gaussian std

        #initialize
        init_var = 0.01
        #init_mag = 9
        self.posterior_mu.data.normal_(1, init_var)
        #self.posterior_logsigma.data.normal_(-init_mag, init_var)
        #self.prior_logsigma.data.normal_(-init_mag, init_var)

        #mask
        self.mask = torch.ones((args.intermediate_dim)).to(device)
        self.mask_dim = 0
        self.pruned_valued_vector = torch.zeros((args.intermediate_dim)).to(device)

        #self.upper_tri_matrix = torch.triu(torch.ones((args.intermediate_dim,args.intermediate_dim))).to(device)

        #self.

    def get_mask(self, threshold=args.thr_ratio):

        #alpha =  torch.exp(self.posterior_logsigma).pow(2) / self.posterior_mu.data.pow(2)
        alpha = torch.abs(self.posterior_mu.data)
        #alpha = self.posterior_mu.data.pow(2)  / (torch.exp(self.posterior_logsigma).pow(2) + 1e-8)
        #alpha = 1/alphas
        #alpha = F.sigmoid(self.posterior_mu.data).pow(2)
        hard_mask = (alpha > threshold).float()
        return hard_mask

    def forward(self, x, args):
        x = x.view(-1, int(x.nelement() / x.shape[0]))

        #print(self.fc1.weight.size())

        l2_norm_squared = torch.sum(self.fc1.weight.pow(2),dim = 1) + self.fc1.bias.pow(2)
        l2_norm = l2_norm_squared.pow(0.5)
        fc1_weight = (self.fc1.weight.permute(1,0) / l2_norm).permute(1,0)
        fc1_bias = self.fc1.bias / l2_norm

        #print(torch.sum(fc1_weight.pow(2),dim = 1) + fc1_bias.pow(2))

        x = F.linear(x, fc1_weight, fc1_bias)

        x = torch.tanh(self.posterior_mu * x)

        #x = torch.tanh(self.posterior_mu * self.fc1(x))

        #x = x * F.sigmoid(self.posterior_mu) # global parameter mu

        KL = self.KL_log_uniform(args.channel_noise**2/(x.pow(2)+1e-8))

        #add AWGN channel noise
        x = x + torch.randn_like(x) * args.channel_noise

        if self.training:
            pass
        else:
            x = x * self.get_mask()

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1), KL

    def KL_log_uniform(self,alpha_squared):
        #alpha_squared = (sigma/mu)^(2)
        k1 = 0.63576
        k2 = 1.8732
        k3 = 1.48695

        batch_size = alpha_squared.size(0)

        KL_term = k1 * F.sigmoid(k2 + k3 * torch.log(alpha_squared)) - 0.5 * F.softplus(-1 * torch.log(alpha_squared)) - k1

        return - torch.sum(KL_term) / batch_size

    def information_pruning_threshold(self, threshold):

        mu = F.sigmoid(self.posterior_mu)
        index = torch.nonzero(torch.lt(mu,threshold)).squeeze(1)
        print('threshold:',threshold)
        print(index)

        self.mask[index] = 0
        self.mask_dim = index.size()[0]
        #self.pruned_valued_vector[index] = 1#self.mu.data[index]
        #self.prior_logsigma.data[index] = np.log(args.channel_noise) # dangerous

        #print(self.pruned_valued_vector)
        #print(F.sigmoid(self.posterior_mu))
        print(mu)
        print(self.mask)

        return 


def train(args, model, device, train_loader, optimizer, epoch):

    model.train()


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, KL = model(data, args)
        if epoch > args.prune_epoch or epoch <=2:
          loss = F.nll_loss(output, target)
        else:
          anneal_ratio = min(1,(epoch - 2)/10)
          loss = F.nll_loss(output, target) + args.penalty * KL * anneal_ratio
        loss.backward()
        optimizer.step()

    



def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    print(model.get_mask())

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, KL = model(data, args)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    #print(torch.exp(model.prior_logsigma))

    #print('---sum---')

    index = torch.nonzero(torch.lt(model.get_mask(),0.5)).squeeze(1)

    pruned_number = index.size()[0]

    print('pruned alpha number:',pruned_number)

    #print(model.posterior_mu.pow(2)/(torch.exp(model.posterior_logsigma).pow(2)+1e-8))

    #print(torch.sum(model.fc1.weight.pow(2),dim = 1) + model.fc1.bias.pow(2))

    print(model.posterior_mu)

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
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = 5e-5)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=15, gamma=args.gamma)
    
    test_acc = 0
    #save_mask_dim = 0
    prune_dim = 0

    for epoch in range(1, args.epochs + 1):
        print()
        print('epoch:',epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        
        scheduler.step()
        #var_print = 0
        
        if epoch == args.prune_epoch:
            #prune the redundant dimensions
            #model.information_pruning_threshold(args.thr_ratio)
            print('prune finished')
            #print('mask_value:',model.mask_dim)
        
        acc, pruned_number = test(args, model, device, test_loader)
        
        #if (not (acc <= test_acc and model.mask_dim == save_mask_dim)) :
        if (acc > test_acc and pruned_number == prune_dim) or pruned_number > prune_dim:
            test_acc = acc
            prune_dim = pruned_number

            #save_mask_dim = model.mask_dim
    #open('DVIB_result/DVIB_MnistJSCC_noise_{:}_thr_ratio{:}_penalty{:}_dim_{:}_ori_dim_{:}_mask_dim_{:}_acc_{:.2f}.txt'.format(args.channel_noise, args.thr_ratio, args.penalty, args.intermediate_dim - model.mask_dim, args.intermediate_dim, save_mask_dim, test_acc),'w').close()
    print('init_number:',args.intermediate_dim, 'penalty:', args.penalty)
    print('best acc:',test_acc,'pruned_number',prune_dim)
    open('./result_vfe/noise{}_ori:{}_prune:{}_remain:{}_acc:{}_thr:{}.txt'.format(args.channel_noise,args.intermediate_dim,prune_dim,args.intermediate_dim - prune_dim,test_acc,args.thr_ratio),'w+')


if __name__ == '__main__':
    main()
