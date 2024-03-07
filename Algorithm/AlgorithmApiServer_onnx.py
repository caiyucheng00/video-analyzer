import base64
import json
from PIL import Image
import io
import onnxruntime
from ResNet_Single_detect import init_model
from flask import Flask, request
import torch

app = Flask(__name__)
save_model_path = './weights/'
res_size = 512
k = 8
resnet, device, transform, labels = init_model(save_model_path, res_size, k)
ort_session = onnxruntime.InferenceSession('./weights/resnet50.onnx')


@app.route("/image/imageClassify", methods=['POST'])
def imageClassify():
    data = {
        "code": 0,
        "msg": "unknown error",
    }
    try:
        params = request.get_json()
    except:
        params = request.form

    # 请求参数
    algorithm_str = params.get("algorithm")
    appKey = params.get("appKey")
    image_base64 = params.get("image_base64", None)  # 接收base64编码的图片并转换成cv2的图片格式

    if image_base64:
        if algorithm_str in ["ResNet50"]:
            encoded_image_byte = base64.b64decode(image_base64)
            image_stream = io.BytesIO(encoded_image_byte)  # 创建一个内存流对象
            image_pil = Image.open(image_stream)  # 使用PIL库打开图像
            input_img = transform(image_pil)
            input_tensor = input_img.unsqueeze(0).numpy()  # (1, 3, 512, 512) ndarray

            ort_inputs = {'input': input_tensor}
            pred_logits = ort_session.run(['output'], ort_inputs)[0]  # (1, 8)  ndarray
            y_pred = torch.from_numpy(pred_logits).max(1, keepdim=True)[1]  # tensor (1)
            index = y_pred.item() if y_pred.numel() == 1 else y_pred.squeeze().cpu().numpy().tolist()  # int

            data["code"] = 1000
            data["msg"] = "success"
            data["result"] = labels[index]

    return json.dumps(data, ensure_ascii=False)
