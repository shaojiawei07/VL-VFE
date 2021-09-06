import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
import argparse
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--intermediate_dim', type=int, default=64)
parser.add_argument('--epochs', type=int, default=320, help='epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=1e-2)
parser.add_argument('--threshold', type=float, default=1e-2)
parser.add_argument('--decay_step', type=int, default=60)
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--weights', type=str)
parser.add_argument('--channel_noise', type=float, default = 0.1)
args = parser.parse_args()

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
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


def seed_torch(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class gamma_layer(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(gamma_layer, self).__init__()
        self.H = nn.Parameter(torch.ones(output_channel, input_channel))
        self.b = nn.Parameter(torch.ones(output_channel))
        self.H.data.normal_(0, 0.1)
        self.b.data.normal_(0, 0.001)

    def forward(self, x):
        H = torch.abs(self.H)
        x = F.linear(x,H)
        return torch.tanh(x)

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

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x*self.weight

class Net(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.args = args
        self.hidden_channel = args.intermediate_dim

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
        
        self.encoder1 = nn.Sequential(
                        nn.Conv2d(512,4,kernel_size = 3,stride = 1, padding = 1, bias = False),
                        nn.BatchNorm2d(4),
                        nn.ReLU()
                        )
        self.encoder2 = nn.Sequential(
                        nn.Linear(64,64),
                        nn.Sigmoid()
                        )

        self.encoder3_weight = nn.Parameter(torch.Tensor(self.hidden_channel, 64))
        self.encoder3_bias = nn.Parameter(torch.Tensor(self.hidden_channel))
        self.encoder3_weight.data.normal_(0, 0.5)
        self.encoder3_bias.data.normal_(0, 0.1)

        self.decoder1 = nn.Linear(self.hidden_channel,64)
        self.decoder1_2 = nn.Sequential(
                        nn.Linear(64,64),
                        nn.ReLU()
                        )
        self.decoder1_2_2 = nn.Sequential(
                        nn.Linear(1,16),
                        nn.ReLU(),
                        nn.Linear(16,16),
                        nn.ReLU(),
                        nn.Linear(16,16),
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
        self.gamma_mu = gamma_function().to(device)
        self.upper_tri_matrix = torch.triu(torch.ones((args.intermediate_dim,args.intermediate_dim))).to(device)

    def forward(self, x, epoch, noise = 0.1):

        x = self.prep(x)
        x = self.layer1(x)
        res = self.layer1_res(x)
        x = res + x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.encoder1(x)
        x = torch.reshape(x,(x.size()[0],4*4*4))

        # Dynamic Channel Conditions
        if self.training:
            channel_noise = torch.rand(1)*0.27 + 0.05
        else:
            channel_noise = torch.FloatTensor([1]) * noise
        channel_noise = channel_noise.to(device)

        x = self.encoder2(x)
        x_norm2 = torch.norm(x,dim=1)
        x = 64 * (x.permute(1,0)/(x_norm2+1e-6)).permute(1,0)
        weight3 = F.tanh(self.encoder3_weight)
        bias3 = F.tanh(self.encoder3_bias)
        weight3 = torch.clamp(torch.abs(weight3),min = 1e-3) * torch.sign(weight3.detach())
        bias3 = torch.clamp(torch.abs(bias3),min = 1e-3) * torch.sign(bias3.detach())
        l2_norm_squared = torch.sum(weight3.pow(2),dim = 1) + bias3.pow(2)
        l2_norm = l2_norm_squared.pow(0.5)
        weight3 = (weight3.permute(1,0) / (l2_norm+1e-6)).permute(1,0)
        bias3 = bias3 / (l2_norm+1e-6)
        x = F.linear(x, weight3, bias3)

        mu = self.gamma_mu(channel_noise)
        mu = F.linear(mu, self.upper_tri_matrix)
        mu = torch.clamp(mu,min = 1e-4)
        encoded_feature = torch.tanh(x * mu)
        encoded_feature = torch.clamp(torch.abs(encoded_feature),min = 1e-2) * torch.sign(encoded_feature.detach())
        
        # KL divergence
        KL = self.KL_log_uniform(channel_noise,torch.abs(encoded_feature))

        # Gaussian channel noise
        x = encoded_feature + torch.randn_like(encoded_feature) * channel_noise

        if self.training:
            if epoch > 60:
                x = x * self.get_mask(mu,threshold = args.threshold)
        else:
            x = x * self.get_mask(mu,threshold = args.threshold)

        x = F.relu(self.decoder1(x))
        x = self.decoder1_2(x)
        noise_feature = self.decoder1_2_2(channel_noise)
        noise_feature = noise_feature.expand(x.size()[0],16)
        x = torch.cat((x,noise_feature),dim=1)
        x = self.decoder1_3(x)
        x = torch.reshape(x,(-1,4,4,4))
        decoded_feature = self.decoder2(x)
        x = self.layer3_res(decoded_feature)
        x = x + decoded_feature
        x = self.classifier1(x)
        output = self.classifier2(x)
        
        return output, KL * 0.1 / channel_noise

    def KL_log_uniform(self,channel_noise,encoded_feature):

        alpha = (channel_noise/encoded_feature)
        k1 = 0.63576
        k2 = 1.8732
        k3 = 1.48695
        batch_size = alpha.size(0)
        KL_term = k1 * F.sigmoid(k2 + k3 * 2 * torch.log(alpha)) - 0.5 * F.softplus(-2 * torch.log(alpha)) - k1
        return - torch.sum(KL_term) / batch_size

    def get_mask(self, mu, threshold=args.threshold):
        alpha = mu.detach()
        hard_mask = (alpha > threshold).float()
        return hard_mask

    def get_mask_inference(self, channel_noise, threshold = args.threshold):
        mu = self.gamma_mu(channel_noise)
        alpha = F.linear(mu, self.upper_tri_matrix)
        hard_mask = (alpha > threshold).float()
        return hard_mask, alpha

model = Net(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,  weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)

def train(model=model):
    
    test_acc = 0
    pruned_dim = 0
    saved_model = {}

    for epoch in range(args.epochs):

        print('\nepoch:{}'.format(epoch))
        if (epoch)%10 == 0:
            data_loader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size, shuffle=True,
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
                loss = loss1 + args.beta * KL * anneal_ratio

            if torch.isnan(loss):
                raise Exception("NaN value")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        acc, pruned_number = test(epoch,noise = 0.1)

        print('Test Accuracy:',acc,'Pruned dim:',pruned_number,'Activated dim:', args.intermediate_dim - pruned_number)

        if epoch > 180:
            if (acc > test_acc and pruned_number == pruned_dim) or pruned_number > pruned_dim:
                test_acc = acc
                pruned_dim = pruned_number
                saved_model = copy.deepcopy(model.state_dict())
                print('Best ckpt:',test_acc,'pruned_number:',pruned_dim,'beta:',args.beta)
                torch.save({'model': saved_model}, './CIFAR_best_ckpt_dim:{}_beta:{}_model.pth'.format(args.intermediate_dim,args.beta))
    print('Best Accuracy:',test_acc,'Intermediate Dim:',args.intermediate_dim,'Beta:',args.beta)
    torch.save({'model': saved_model}, './CIFAR_model_dim:{}_beta:{}_accuracy:{}_model.pth'.format(args.intermediate_dim - pruned_dim,args.beta,test_acc))

def test(epoch,noise=0.1):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader_this): 
            images = images.to(device)
            labels = labels.to(device)
            outputs,_= model(images,epoch,noise)
            labels_relize = labels.view(-1,1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        hard_mask, mu = model.get_mask_inference(torch.FloatTensor([noise]).to(device))
        index = torch.nonzero(torch.lt(hard_mask,0.5)).squeeze(1)
        pruned_number = index.size()[0]
        return 100 * correct / total, pruned_number
      
if __name__=='__main__':
    seed_torch(0)
    if args.test == 0:
        train(model)
    else:
        model.load_state_dict(torch.load(args.weights)['model'])
        accuracy = 0
        t = 20
        for i in range (t):
            acc, pruned_number = test(0,args.channel_noise)
            accuracy += acc
        #print('Noise level:',args.channel_noise,'Test Accuracy:',accuracy/t,'Activated dim:', args.intermediate_dim - pruned_number)
        print('Noise level:',args.channel_noise, 'Test Accuracy:', accuracy/t, 'Pruned dim:', pruned_number, 'Activated dim:', args.intermediate_dim - pruned_number)

