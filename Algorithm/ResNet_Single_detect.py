from functions import *


def init_model(save_model_path, res_size, k):
    save_model_path = save_model_path
    res_size = res_size
    k = k
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
    # data loading parameters
    use_cuda = True
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
        image = image.to(device)
        output = model(image)
        y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
        y_pred = y_pred.item() if y_pred.numel() == 1 else y_pred.squeeze().cpu().numpy().tolist()

    return y_pred
