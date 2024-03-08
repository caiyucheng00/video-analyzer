#ifndef __ALGORITHMWITHTENSORRT__H_
#define __ALGORITHMWITHTENSORRT__H_

#include "Algorithm.h"
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <../samples/common/logger.h> 

class Config;

class AlgorithmWithTensorRT : public Algorithm
{
public:
	AlgorithmWithTensorRT() = delete;
	AlgorithmWithTensorRT(Config* config);
	virtual ~AlgorithmWithTensorRT();

	virtual void imageClassify(int height, int width, unsigned char* bgr, std::string& classify_result);

private:
	void preprocess(cv::Mat image, float* inputData);
	void postprocess(const float* outputData);

	Config* _config;
	sample::Logger _logger;
	nvinfer1::IRuntime* _runtime;
	nvinfer1::ICudaEngine* _engine;
	nvinfer1::IExecutionContext* _context;

	float* _inputData = new float[512 * 512 * 3];
	float* _outputData = new float[8];
	void* _deviceInput;
	void* _deviceOutput;
	void* _bindings[2];
};

#endif