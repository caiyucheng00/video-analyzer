import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


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


def init_model(save_model_path, res_size, k):
    save_model_path = save_model_path
    res_size = res_size
    k = k
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
    # data loading parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # Create model
    resnet = ResNet50Encoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, num_classes=k,
                             pretrained=False).to(device)
    resnet = nn.DataParallel(resnet)
    resnet.load_state_dict(torch.load(os.path.join(save_model_path, '1s_epoch50.pth')))
    print('resnet model reloaded!')
    labels = ['出苗期', '分蘖期', '拔节期', '孕穗期', '抽穗期', '开花期', '灌浆期', '成熟期']

    return resnet, device, transform, labels


def prediction(model, device, image):
    model.eval()
    with torch.no_grad():
        # distribute data to device
        image = image.to(device)   # (1, 3, 512, 512)  tensor
        output = model(image)    # (1,8)  tensor
        y_pred = output.max(1, keepdim=True)[1]  #(1,1) tensor  location of max log-probability as prediction
        y_pred = y_pred.item() if y_pred.numel() == 1 else y_pred.squeeze().cpu().numpy().tolist()   #int

    return y_pred
