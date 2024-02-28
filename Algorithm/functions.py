import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm


## ------------------- label conversion tools ------------------ ##
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)


def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()


def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()


def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


## ---------------------- Dataloaders ---------------------- ##
## for single
class Dataset_SingleCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_path, folders, labels, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        img_list = os.listdir(path + os.sep + selected_folder)

        image = Image.open(path + os.sep + selected_folder + os.sep + img_list[0])

        if use_transform is not None:
            image = use_transform(image)

        X = image

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        # X = X.permute(1, 0, 2, 3)
        return X, y  ## (3，224，224） +  （1，）


class Dataset_SingleCNN_Detect(data.Dataset):
    def __init__(self, data_path, images, transform=None):
        "Initialization"
        self.data_path = data_path
        self.images = images
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.images)

    def read_images(self, path, image, use_transform):
        image = Image.open(path + os.sep + image)

        if use_transform is not None:
            image = use_transform(image)

        X = image

        return X

    def __getitem__(self, index):
        # Select sample
        image = self.images[index]
        # Load data
        X = self.read_images(self.data_path, image, self.transform)  # (input) spatial images

        return X  ## (3，224，224）


## ---------------------- end of Dataloaders ---------------------- ##
# for 3DCNN
class Dataset_3DCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i))).convert('L')

            if use_transform is not None:
                image = use_transform(image)

            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform).unsqueeze_(0)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y


# for CRNN
class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, selected_folder + '_{:02d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        X = X.permute(1, 0, 2, 3)
        return X, y  ## (28，3，224，224） +  （1，）


## ---------------------- end of Dataloaders ---------------------- ##


## -------------------- (reload) model prediction ---------------------- ##
def Conv3d_final_prediction(model, device, loader):
    model.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = model(X)
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred


def CRNN_final_prediction(model, device, loader):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = rnn_decoder(cnn_encoder(X))
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred


def Single_final_prediction(model, device, loader):
    model.eval()
    all_y_pred = []
    with torch.no_grad():
        for X in tqdm(loader):
            # distribute data to device
            X = X.to(device)
            output = model(X)
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            y_pred = y_pred.item() if y_pred.numel() == 1 else y_pred.squeeze().cpu().numpy().tolist()
            all_y_pred.append(y_pred)

    return all_y_pred


## -------------------- end of model prediction ---------------------- ##


## ------------------------ 3D CNN module ---------------------- ##
def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape


class CNN3D(nn.Module):
    def __init__(self, t_dim=120, img_x=90, img_y=120, drop_p=0.2, fc_hidden1=256, fc_hidden2=128, num_classes=50):
        super(CNN3D, self).__init__()

        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2],
                             self.fc_hidden1)  # fully connected hidden layer
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # fully connected layer, output = multi-classes

    def forward(self, x_3d):
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        return x


## --------------------- end of 3D CNN module ---------------- ##


## ------------------------ CRNN module ---------------------- ##

def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape


class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


# 2D CNN encoder train from scratch (no transfer learning)
class EncoderCNN(nn.Module):
    def __init__(self, img_x=90, img_y=120, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        super(EncoderCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y), self.pd1, self.k1,
                                                 self.s1)  # Conv1 output shape
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)

        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                      padding=self.pd2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3,
                      padding=self.pd3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4,
                      padding=self.pd4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.drop = nn.Dropout2d(self.drop_p)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1],
                             self.fc_hidden1)  # fully connected layer, output k classes
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # CNNs
            x = self.conv1(x_3d[:, t, :, :, :])
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)  # flatten the output of conv

            # FC layers
            x = F.relu(self.fc1(x))
            # x = F.dropout(x, p=self.drop_p, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class SPPLayer(nn.Module):
    def __init__(self, levels):
        super(SPPLayer, self).__init__()
        self.levels = levels

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        output = []

        for level in self.levels:
            pooling_size = (height // level, width // level)
            spp = F.adaptive_max_pool2d(x, pooling_size)
            spp = spp.view(batch_size, -1)
            output.append(spp)

        output = torch.cat(output, dim=1)
        return output


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):  ##(40,28,512)
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  ##(40,28,512)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step ##(40,512) -->(40,256)
        pre_x = self.fc1(RNN_out[:, 15, :])
        x = F.relu(x)
        pre_x = F.relu(pre_x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        pre_x = F.dropout(pre_x, p=self.drop_p, training=self.training)
        x = self.fc2(x)  # (40,101)
        pre_x = self.fc2(pre_x)  # (40,101)

        return x, pre_x


## ---------------------- end of CRNN module ---------------------- ##


# 2D CNN encoder using ResNet-152 pretrained
class ResNet152Encoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResNet152Encoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):  ##（40,28,3,224,224）
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet    (40,3,224,224) --> (40,2048,1,1)
                x = x.view(x.size(0), -1)  # flatten output of conv  (40,2048)

            # FC layers
            x = self.bn1(self.fc1(x))  ## (40,1024)
            x = F.relu(x)
            x = self.bn2(self.fc2(x))  ## (40,768)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)  ## (40,512)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)  ## (40,28,512)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


## ---------------------- Adapt + SPP module ---------------------- ##
class ResNet152EncoderAdapt(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResNet152EncoderAdapt, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]  # delete the last fc + pool
        self.resnet = nn.Sequential(*modules)

        # Replace AdaptiveAvgPool2d with standard AvgPool2d
        self.avgpool = nn.AvgPool2d((16, 16))

        # Convert the original fc layer to a convolutional layer.
        self.last_conv = torch.nn.Conv2d(in_channels=resnet.fc.in_features, out_channels=1000, kernel_size=1)
        self.last_conv.weight.data.copy_(resnet.fc.weight.data.view(*resnet.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_(resnet.fc.bias.data)

        self.fc1 = nn.Linear(1000, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(1000, momentum=0.01)
        self.fc2 = nn.Linear(1000, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):  ##（40,28,3,224,224）
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet    (40,3,224,224) --> (40,2048,7,7)
                x = self.avgpool(x)  # (40,2048,1,1)

            x = self.last_conv(x)  # (40,1000,1,1)
            x = x.view(x.size(0), -1)  # (40,1000)
            # FC layers
            x = self.bn1(x)  ## (40,1000)
            x = F.relu(x)
            x = self.bn2(self.fc2(x))  ## (40,768)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)  ## (40,512)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)  ## (40,28,512)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class ResNet152EncoderSPP(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResNet152EncoderSPP, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]  # delete the last fc + pool
        self.resnet = nn.Sequential(*modules)

        self.spp = SPPLayer([2, 4])

        self.fc1 = nn.Linear(163840, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):  ##（40,28,3,224,224）
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet    (40,3,500,500) --> (40,2048,16,16)

            x = self.spp(x)  # (40,163840)

            # FC layers
            x = self.bn1(self.fc1(x))  ## (40,1024)
            x = F.relu(x)
            x = self.bn2(self.fc2(x))  ## (40,768)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)  ## (40,512)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)  ## (40,28,512)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


# 2D CNN encoder using ResNet-18 pretrained
class ResNet18Encoder(nn.Module):
    def __init__(self, fc_hidden1=0, fc_hidden2=0, num_classes=9, drop_p=0.3, pretrained=True):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResNet18Encoder, self).__init__()

        self.drop_p = drop_p
        self.num_classes = num_classes

        resnet = models.resnet18(pretrained=pretrained)
        modules = list(resnet.children())[:-1]  # del the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):  ##（b,3,224,224）
        x = self.resnet(x)  # ResNet    (b,3,224,224) --> (b,512,1,1)
        x = x.view(x.size(0), -1)  # flatten output of conv  (b,512)

        # FC layers
        x = self.fc(x)  ## (b,9)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)

        return x


class ResNet18EncoderAdapt(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResNet18EncoderAdapt, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]  # delete the last fc + pool
        self.resnet = nn.Sequential(*modules)

        # Replace AdaptiveAvgPool2d with standard AvgPool2d
        self.avgpool = nn.AvgPool2d((7, 7))

        # Convert the original fc layer to a convolutional layer.
        self.last_conv = torch.nn.Conv2d(in_channels=resnet.fc.in_features, out_channels=1000, kernel_size=1)
        self.last_conv.weight.data.copy_(resnet.fc.weight.data.view(*resnet.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_(resnet.fc.bias.data)

        self.fc1 = nn.Linear(1000, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(1000, momentum=0.01)
        self.fc2 = nn.Linear(1000, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):  ##（40,28,3,224,224）
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet    (40,3,224,224) --> (40,512,7,7)
                x = self.avgpool(x)  # (40,512,1,1)

            x = self.last_conv(x)  # (40,1000,1,1)
            x = x.view(x.size(0), -1)  # (40,1000)
            # FC layers
            x = self.bn1(x)  ## (40,1000)
            x = F.relu(x)
            x = self.bn2(self.fc2(x))  ## (40,768)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)  ## (40,512)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)  ## (40,28,512)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class ResNet18EncoderSPP(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResNet18EncoderSPP, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]  # delete the last fc + pool
        self.resnet = nn.Sequential(*modules)

        self.spp = SPPLayer([2, 4])

        self.fc1 = nn.Linear(5120, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):  ##（40,28,3,224,224）
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet    (40,3,224,224) --> (40,512,7,7)

            x = self.spp(x)  # (40,5120)

            # FC layers
            x = self.bn1(self.fc1(x))  ## (40,1024)
            x = F.relu(x)
            x = self.bn2(self.fc2(x))  ## (40,768)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)  ## (40,512)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)  ## (40,28,512)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


# 2D CNN encoder using ResNet-50 pretrained
class ResNet50Encoder(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, num_classes=9, pretrained=True):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(ResNet50Encoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet50(pretrained=pretrained)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, num_classes)

    def forward(self, x):  ##（40,28,3,224,224）

        x = self.resnet(x)  # ResNet    (b,3,224,224) --> (b,2048,1,1)
        x = x.view(x.size(0), -1)  # flatten output of conv  (b,2048)

        # FC layers
        x = self.bn1(self.fc1(x))  ## (b,1024)
        x = F.relu(x)
        x = self.bn2(self.fc2(x))  ## (b,768)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)  ## (40,9)

        return x


class ResNet50EncoderAttention(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, num_classes=9, pretrained=True):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(ResNet50EncoderAttention, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet50(pretrained=pretrained)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.nonlocal1 = NonLocalBlock(channel=256)
        self.nonlocal2 = NonLocalBlock(channel=512)
        self.nonlocal3 = NonLocalBlock(channel=1024)
        modules.insert(5, self.nonlocal1)
        modules.insert(7, self.nonlocal2)
        modules.insert(9, self.nonlocal3)

        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, num_classes)

    def forward(self, x):  ##（40,28,3,224,224）

        x = self.resnet(x)  # ResNet    (b,3,224,224) --> (b,2048,1,1)
        x = x.view(x.size(0), -1)  # flatten output of conv  (b,2048)

        # FC layers
        x = self.bn1(self.fc1(x))  ## (b,1024)
        x = F.relu(x)
        x = self.bn2(self.fc2(x))  ## (b,768)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)  ## (40,9)

        return x


class ResNet50EncoderAdapt(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResNet50EncoderAdapt, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # delete the last fc + pool
        self.resnet = nn.Sequential(*modules)

        # Replace AdaptiveAvgPool2d with standard AvgPool2d
        self.avgpool = nn.AvgPool2d((16, 16))

        # Convert the original fc layer to a convolutional layer.
        self.last_conv = torch.nn.Conv2d(in_channels=resnet.fc.in_features, out_channels=1000, kernel_size=1)
        self.last_conv.weight.data.copy_(resnet.fc.weight.data.view(*resnet.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_(resnet.fc.bias.data)

        self.fc1 = nn.Linear(1000, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(1000, momentum=0.01)
        self.fc2 = nn.Linear(1000, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):  ##（40,28,3,224,224）
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet    (40,3,224,224) --> (40,2048,7,7)
                x = self.avgpool(x)  # (40,2048,1,1)

            x = self.last_conv(x)  # (40,1000,1,1)
            x = x.view(x.size(0), -1)  # (40,1000)
            # FC layers
            x = self.bn1(x)  ## (40,1000)
            x = F.relu(x)
            x = self.bn2(self.fc2(x))  ## (40,768)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)  ## (40,512)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)  ## (40,28,512)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class ResNet50EncoderSPP(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResNet50EncoderSPP, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # delete the last fc + pool
        self.resnet = nn.Sequential(*modules)

        self.spp = SPPLayer([2, 4])

        self.fc1 = nn.Linear(163840, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):  ##（40,28,3,224,224）
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet    (40,3,500,500) --> (40,2048,16,16)

            x = self.spp(x)  # (40,163840)

            # FC layers
            x = self.bn1(self.fc1(x))  ## (40,1024)
            x = F.relu(x)
            x = self.bn2(self.fc2(x))  ## (40,768)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)  ## (40,512)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)  ## (40,28,512)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


# 2D CNN encoder using ResNet-50 pretrained
class ResNet50EncoderLittle(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(ResNet50EncoderLittle, self).__init__()

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)

    def forward(self, x_3d):  ##（40,28,3,224,224）
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet    (40,3,224,224) --> (40,2048,1,1)
                x = x.view(x.size(0), -1)  # flatten output of conv  (40,2048)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)  ## (b,f,2048)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNNLittle(nn.Module):
    def __init__(self):
        super(DecoderRNNLittle, self).__init__()

        self.LSTM = nn.LSTM(
            input_size=2048,
            hidden_size=100,
            num_layers=2,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(100, 9)

    def forward(self, x_RNN):  ##(b,f,2048)
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  ##(40,28,100)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step ##(40,100) -->(40,256)

        return x

## ---------------------- end of CRNN module ---------------------- ##
