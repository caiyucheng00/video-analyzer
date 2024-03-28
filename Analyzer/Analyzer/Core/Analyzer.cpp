#include "Analyzer.h"
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include <json/json.h>
#include <json/value.h>
#include <Python.h>
#include "Control.h"
#include "Config.h"
#include "AlgorithmWithAPI.h"
#include "AlgorithmWithPy.h"
#include "AlgorithmWithTensorRT.h"
#include "Utils/PutText.h"
#include "Utils/Log.h"
#include "Patterns/Model.h"

Analyzer::Analyzer(Config* config, Control* control, Model* model) :
	_config(config),
	_control(control)
{
	Py_SetPythonHome(L"D:\\Developer\\Python\\Anaconda3\\envs\\video-analyzer");
	std::string type = _config->algorithmType;

	if ("api" == type) {
		_algorithm = new AlgorithmWithAPI(_config);   // 及时delete
	}
	else if("py" == type) {
		_algorithm = new AlgorithmWithPy(_config);   // 及时delete
	}
	else if ("tensorrt" == type) {
		_algorithm = new AlgorithmWithTensorRT(model, _control->behaviorCode);   // 及时delete
	}
	else {
		LOGE("Algorithm Type Error");
	}
	
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
		_algorithm->doAlgorithm(image, _results);
		//result_str = _results[0].class_name;
	}
	
	//LOGI("%s", result_str);   //utf8
	std::string show_gbk = UTF8ToGBK(show_str + result_str);   //gbk
	const char* show_gbk_char = show_gbk.c_str();
	putTextHusky(image, show_gbk_char, cv::Point(100, 150), cv::Scalar(0, 0, 255), 50, "Arial", false, false);
	//cv::putText(image, class_name, cv::Point(100, 150), cv::FONT_HERSHEY,SIMPLEX, _control->videoWidth / 1000, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
	std::string info = "checkFps:" + std::to_string(_control->checkFps);  //ControlExecutor::DecodeAndAnalyzeVideoThread 计算得到
	cv::putText(image, info, cv::Point(20, 40), cv::FONT_HERSHEY_COMPLEX, _control->videoWidth / 1000, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
}

