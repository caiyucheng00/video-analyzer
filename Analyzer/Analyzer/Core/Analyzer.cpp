#include "Analyzer.h"
#include <opencv2/opencv.hpp>
#include <json/json.h>
#include <json/value.h>
#include <Python.h>
#include "Control.h"
#include "Scheduler.h"
#include "AlgorithmWithAPI.h"
#include "AlgorithmWithPy.h"
#include "Utils/PutText.h"
#include "Utils/Log.h"

Analyzer::Analyzer(Scheduler* scheduler, Control* control) :
	_scheduler(scheduler),
	_control(control)
{
	Py_SetPythonHome(L"D:\\Developer\\Python\\Anaconda3\\envs\\video-analyzer");
	_algorithm = new AlgorithmWithPy(_scheduler->getConfig());   // 及时delete
}

Analyzer::~Analyzer()
{
	if (_algorithm) {
		delete _algorithm;
		_algorithm = nullptr;
	}
}

void Analyzer::checkVideoFrame(bool check, unsigned char* data)
{
	cv::Mat image(_control->videoHeight, _control->videoWidth, CV_8UC3, data);
	
	if (check) {
		_algorithm->imageClassify(_control->videoHeight, _control->videoWidth, data, classify_result);
		if (classify_result.length() > 0) {
			result_str = classify_result;
		}
	}
	
	//LOGI("%s", result_str);   //utf8
	std::string show_gbk = UTF8ToGBK(show_str + result_str);   //gbk
	const char* show_gbk_char = show_gbk.c_str();
	putTextHusky(image, show_gbk_char, cv::Point(100, 150), cv::Scalar(0, 0, 255), 50, "Arial", false, false);
	//cv::putText(image, class_name, cv::Point(100, 150), cv::FONT_HERSHEY,SIMPLEX, _control->videoWidth / 1000, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
	std::string info = "checkFps:" + std::to_string(_control->checkFps);  //ControlExecutor::DecodeAndAnalyzeVideoThread 计算得到
	cv::putText(image, info, cv::Point(20, 40), cv::FONT_HERSHEY_COMPLEX, _control->videoWidth / 1000, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
}

