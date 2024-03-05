import cv2
import numpy as np
import base64
from ResNet_Single_detect import init_model, prediction


class Algorithm():
    def __init__(self):
        save_model_path = './weights/'
        res_size = 512
        k = 8
        self.resnet, self.device, self.transform, self.labels = init_model(save_model_path, res_size, k)

    def __del__(self):
        print("__del__.%s" % (self.__class__.__name__))

    def imageClassify(self, image_type, image):
        if 1 == image_type:
            # 1 == image_type 则 image是str类型的 image_base64
            encoded_image_byte = base64.b64decode(image)
            image_array = np.frombuffer(encoded_image_byte, np.uint8)
            image = cv2.imdecode(image_array, cv2.COLOR_RGB2BGR)  # opencv 解码

        image = self.transform(image)
        image = image.unsqueeze(0)

        index = prediction(self.resnet, self.device, image)
        label = self.labels[index]
        return label
