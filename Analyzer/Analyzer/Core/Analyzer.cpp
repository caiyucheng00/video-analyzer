#include "Analyzer.h"
#include <opencv2/opencv.hpp>
#include "Control.h"
#include "Scheduler.h"
#include "AlgorithmWithAPI.h"

Analyzer::Analyzer(Scheduler* scheduler, Control* control) :
	_scheduler(scheduler),
	_control(control)
{
	_algorithm = new AlgorithmWithAPI(_scheduler->getConfig());   // 及时delete
}

Analyzer::~Analyzer()
{
	if (_algorithm) {
		delete _algorithm;
		_algorithm = nullptr;
	}
}

void Analyzer::checkVideoFrame(bool check, int64_t frameCount, unsigned char* data)
{
	std::string class_name = "PHE";
	cv::Mat image(_control->videoHeight, _control->videoWidth, CV_8UC3, data);
	
	if (check) {
		_algorithm->imageClassify(_control->videoHeight, _control->videoWidth, data, _classify);
		if (_classify.size() > 0) {
			class_name = _classify[0];
		}
	}

	cv::putText(image, class_name, cv::Point(100, 150), cv::FONT_HERSHEY_SIMPLEX, _control->videoWidth / 1000, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
	std::string info = "checkFps:" + std::to_string(_control->checkFps);  //ControlExecutor::DecodeAndAnalyzeVideoThread 计算得到
	cv::putText(image, info, cv::Point(20, 40), cv::FONT_HERSHEY_COMPLEX, _control->videoWidth / 1000, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
}

void Analyzer::checkAudioFrame(bool check, int64_t frameCount, unsigned char* data, int size)
{

}
