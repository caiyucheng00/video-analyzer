#include "AlgorithmWithTensorRT.h"


AlgorithmWithTensorRT::AlgorithmWithTensorRT(Config* config) :
	_config(config)
{
	// 从文件中读取已保存的引擎
	std::ifstream engineFile("D:\\Developer\\Visual Studio\\Project\\TensorRT\\resnet50.engine", std::ios::binary);
	std::vector<char> engineData((std::istreambuf_iterator<char>(engineFile)), std::istreambuf_iterator<char>());
	engineFile.close();

	// 使用读取的数据创建运行时和引擎
	_runtime = nvinfer1::createInferRuntime(_logger);
	_engine = _runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr);

	// 创建执行上下文
	_context = _engine->createExecutionContext();
	cudaMalloc(&_deviceInput, sizeof(float) * 512 * 512 * 3); // 假设输入是 224x224x3
	cudaMalloc(&_deviceOutput, sizeof(float) * 8); // 假设输出是 1000 类
}

AlgorithmWithTensorRT::~AlgorithmWithTensorRT()
{
	// 释放资源
	delete[] _inputData;
	delete[] _outputData;
	_context->destroy();
	_engine->destroy();
	_runtime->destroy();
	cudaFree(_deviceInput);
	cudaFree(_deviceOutput);
}

void AlgorithmWithTensorRT::imageClassify(int height, int width, unsigned char* bgr, std::string& classify_result)
{
	cv::Mat image(height, width, CV_8UC3, bgr);
	preprocess(image, _inputData);

	// 将预处理后的图像拷贝到设备内存中
	cudaMemcpy(_deviceInput, _inputData, sizeof(float) * 512 * 512 * 3, cudaMemcpyHostToDevice);

	// 执行推理
	_bindings[0] = _deviceInput;
	_bindings[1] = _deviceOutput;
	_context->executeV2(_bindings);

	// 拷贝输出数据回主机
	cudaMemcpy(_outputData, _deviceOutput, sizeof(float) * 8, cudaMemcpyDeviceToHost);

	// 后处理（这里假设你有一个名为 postprocess 的函数来处理输出数据）
	postprocess(_outputData);
}



void AlgorithmWithTensorRT::preprocess(cv::Mat image, float* inputData)
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

void AlgorithmWithTensorRT::postprocess(const float* outputData)
{
	// 假设找出最大概率的类别作为预测结果
	int maxIndex = 0;
	float maxProbability = outputData[0];
	for (int i = 1; i < 8; ++i) {
		if (outputData[i] > maxProbability) {
			maxIndex = i;
			maxProbability = outputData[i];
		}
	}
	std::cout << "Predicted class: " << maxIndex << ", Probability: " << maxProbability << std::endl;

}
