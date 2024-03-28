#ifndef __ALGORITHMWITHTENSORRT__H_
#define __ALGORITHMWITHTENSORRT__H_

#include "Algorithm.h"
#include <string>

class Model;
class Strategy;

class AlgorithmWithTensorRT : public Algorithm
{
public:
	AlgorithmWithTensorRT() = delete;
	AlgorithmWithTensorRT(Model* model, std::string behaviorCode);
	virtual ~AlgorithmWithTensorRT();

	virtual void doAlgorithm(cv::Mat image, std::vector<AlgorithmResult>& results);

private:
	Model* _model;
	Strategy* _strategy;
};

#endif