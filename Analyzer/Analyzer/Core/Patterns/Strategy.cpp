#include "Strategy.h"
#include "Model.h"
#include "../Analyzer.h"


void ImageClassificationStrategy::preprocess(cv::Mat image, float* inputData)
{
	float mean[] = { 0.406, 0.456, 0.485 };
	float std[] = { 0.225, 0.224, 0.229 };
	cv::resize(image, image, cv::Size(512, 512));

	int image_area = image.cols * image.rows;
	unsigned char* pimage = image.data;
	float* phost_b = inputData + image_area * 0;
	float* phost_g = inputData + image_area * 1;
	float* phost_r = inputData + image_area * 2;
	for (int i = 0; i < image_area; ++i, pimage += 3) {
		// 注意这里的顺序rgb调换了  
		*phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
		*phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
		*phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
	}
}

std::map<std::string, float> ImageClassificationStrategy::postprocess(const float* outputData, int length)
{
	std::map<std::string, float> resultMap;
	// 假设找出最大概率的类别作为预测结果
	int maxIndex = 0;
	float maxProbability = outputData[0];
	for (int i = 1; i < length; ++i) {
		if (outputData[i] > maxProbability) {
			maxIndex = i;
			maxProbability = outputData[i];
		}
	}
	//std::cout << "Predicted class: " << maxIndex << ", Probability: " << maxProbability << std::endl;
	std::string maxCategory = _labels[maxIndex];
	resultMap[maxCategory] = maxProbability;
	return resultMap;
}

WheatImageClassificationStrategy::WheatImageClassificationStrategy()
{
	_inputSize = 512 * 512 * 3;
	_outputSize = 8;
	_inputData = new float[_inputSize];
	_outputData = new float[_outputSize];
	cudaMalloc(&_deviceInput, sizeof(float) * _inputSize); // 输入是 512x512x3
	cudaMalloc(&_deviceOutput, sizeof(float) * _outputSize); // 输出是 8 类
	_labels = { "出苗期", "分蘖期", "拔节期", "孕穗期", "抽穗期", "开花期", "灌浆期", "成熟期" };
}

WheatImageClassificationStrategy::~WheatImageClassificationStrategy()
{
	delete[] _inputData;
	delete[] _outputData;
	cudaFree(_deviceInput);
	cudaFree(_deviceOutput);
}

void WheatImageClassificationStrategy::doStrategy(cv::Mat image, Model* model, std::vector<AlgorithmResult>& results)
{
	preprocess(image, _inputData);

	// 将预处理后的图像拷贝到设备内存中
	cudaMemcpy(_deviceInput, _inputData, sizeof(float) * _inputSize, cudaMemcpyHostToDevice);

	// 执行推理
	_bindings[0] = _deviceInput;
	_bindings[1] = _deviceOutput;
	model->getContext()->executeV2(_bindings);

	// 拷贝输出数据回主机
	cudaMemcpy(_outputData, _deviceOutput, sizeof(float) * _outputSize, cudaMemcpyDeviceToHost);

	// 后处理
	std::map<std::string, float> resultMap = postprocess(_outputData, _outputSize);
	for (const auto& pair : resultMap) {
		AlgorithmResult result;
		result.strategy_name = "ImageClassification";
		result.class_name = pair.first;
		result.score = pair.second;
	}
}
