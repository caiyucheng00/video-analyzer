import os
from functions import *
import time
import cv2
from PIL import Image



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

    return resnet, device, transform

def prediction(model, device, image):
    model.eval()
    all_y_pred = []
    with torch.no_grad():
            # distribute data to device
            image = image.to(device)
            output = model(image)
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            y_pred = y_pred.item() if y_pred.numel() == 1 else y_pred.squeeze().cpu().numpy().tolist()
            all_y_pred.append(y_pred)

    return all_y_pred
def detect_forward( resnet, device, image):
    # image = Image.open("show_single/classify.png")
    # image = cv2.imread("show_single/classify.png")
    # 图片转化tensor
    # PIL_image = Image.fromarray(image)  # 这里ndarray_image为原来的numpy数组类型的输入
    # image = transform(image)
    # image = image.unsqueeze(0)
    # fnames = sorted(os.listdir(data_path))
    # all_X_list = []
    # for f in fnames:
    #     all_X_list.append(f)
    #
    # # reset data loader
    # all_data_params = {'shuffle': False, 'num_workers': 2, 'pin_memory': True}
    # all_data_loader = data.DataLoader(Dataset_SingleCNN_Detect(data_path, all_X_list, transform=transform),
    #                                   **all_data_params)
    #
    # # make all video predictions by reloaded model
    # print('Predicting all {} images:'.format(len(all_data_loader.dataset)))
    labels = ['1Emergence', '2Tillering', '3Jointing', '4Booting', '5Heading',
              '6Anthesis', '7Filling', '8Maturity']
    all_y_pred = prediction(resnet, device, image)  # list[0,1]

    return labels[all_y_pred[0]]


if __name__ == '__main__':
    start_time = time.time()
    data_path = './show_single/'
    save_model_path = 'weights'
    res_size = 512
    k = 8

    resnet, device, transform = init_model(save_model_path, res_size, k)
    label = detect_forward(resnet, device, transform)
    print(label)
    end_time = time.time()
    exe_time = end_time - start_time
    print("Time:", exe_time)
