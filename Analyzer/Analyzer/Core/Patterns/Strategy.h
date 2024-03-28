#ifndef __STRATEGY__H_
#define __STRATEGY__H_

#include <string>
#include <opencv2/opencv.hpp>
class Model;
struct AlgorithmResult;

class Strategy
{
public:
	virtual void doStrategy(cv::Mat image, Model* model, std::vector<AlgorithmResult>& results) = 0;
};


// 图像分类
class ImageClassificationStrategy : public Strategy
{
public:
	virtual void doStrategy(cv::Mat image, Model* model, std::vector<AlgorithmResult>& results) = 0;

protected:
	void preprocess(cv::Mat image, float* inputData);
	std::map<std::string, float> postprocess(const float* outputData, int length);

	float* _inputData;
	float* _outputData;
	void* _deviceInput;
	void* _deviceOutput;
	void* _bindings[2];

	int _inputSize;
	int _outputSize;

	std::vector<std::string> _labels;
};

class WheatImageClassificationStrategy : public ImageClassificationStrategy
{
public:
	WheatImageClassificationStrategy();
	virtual ~WheatImageClassificationStrategy();
	void doStrategy(cv::Mat image, Model* model, std::vector<AlgorithmResult>& results) override;
};


#endif