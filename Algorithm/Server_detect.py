import os
from functions import *
import time


def detect_forward(data_path, save_model_path, res_size, k):
    data_path = data_path
    save_model_path = save_model_path
    res_size = res_size
    k = k
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768

    # 图片转化tensor
    fnames = sorted(os.listdir(data_path))
    all_X_list = []
    for f in fnames:
        all_X_list.append(f)

    # data loading parameters
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # reset data loader
    all_data_params = {'shuffle': False, 'num_workers': 2, 'pin_memory': True}
    all_data_loader = data.DataLoader(Dataset_SingleCNN_Detect(data_path, all_X_list, transform=transform),
                                      **all_data_params)

    # Create model
    resnet = ResNet50Encoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, num_classes=k,
                             pretrained=False).to(device)
    resnet = nn.DataParallel(resnet)
    resnet.load_state_dict(torch.load(os.path.join(save_model_path, '1s_epoch50.pth')))
    print('resnet model reloaded!')

    # make all video predictions by reloaded model
    print('Predicting all {} images:'.format(len(all_data_loader.dataset)))
    labels = ['1Emergence', '2Tillering', '3Jointing', '4Booting', '5Heading',
            '6Anthesis', '7Filling', '8Maturity']
    all_y_pred = Single_final_prediction(resnet, device, all_data_loader)  # list[0,1]
    for y in all_y_pred:
        print(labels[y])


if __name__ == '__main__':
    start_time = time.time()
    data_path = './show_single/'
    save_model_path = 'weights/'
    res_size = 512
    k = 8

    detect_forward(data_path, save_model_path, res_size, k)
    end_time = time.time()
    exe_time = end_time - start_time
    print("Time:", exe_time)
