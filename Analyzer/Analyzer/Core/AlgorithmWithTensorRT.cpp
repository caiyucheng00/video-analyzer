#include "AlgorithmWithTensorRT.h"
#include "Patterns/Strategy.h"
#include "Patterns/Model.h"
#include "Utils/Log.h"
#include <opencv2/opencv.hpp>


AlgorithmWithTensorRT::AlgorithmWithTensorRT(Model* model, std::string behaviorCode) :
	_model(model)
{
	if ("PHE_WHEAT" == behaviorCode) {
		_strategy = new WheatImageClassificationStrategy();
	}
	else if ("PHE_RICE" == behaviorCode) {
		_strategy = new WheatImageClassificationStrategy();
	}
	else if ("SPIKE_WHEAT" == behaviorCode) {
		_strategy = new WheatImageClassificationStrategy();
	}
	else if ("SPIKE_RICE" == behaviorCode) {
		_strategy = new WheatImageClassificationStrategy();
	}
	else if ("SEEDLING_WHEAT" == behaviorCode) {
		_strategy = new WheatImageClassificationStrategy();
	}
	else {
		LOGE("Strategy Type Error");
	}
}

AlgorithmWithTensorRT::~AlgorithmWithTensorRT()
{
	if (_strategy) {
		delete _strategy;
		_strategy = nullptr;
	}
}

void AlgorithmWithTensorRT::doAlgorithm(cv::Mat image, std::vector<AlgorithmResult>& results)
{
	_strategy->doStrategy(image, _model, results);
}

