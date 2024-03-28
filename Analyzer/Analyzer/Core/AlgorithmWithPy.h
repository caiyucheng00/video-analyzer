#ifndef __ALGORITHMWITHPY__H_
#define __ALGORITHMWITHPY__H_

#include "Algorithm.h"
#include <Python.h>

class Config;

class AlgorithmWithPy : public Algorithm
{
public:
	AlgorithmWithPy() = delete;
	AlgorithmWithPy(Config* config);
	virtual ~AlgorithmWithPy();

	virtual void doAlgorithm(cv::Mat image, std::vector<AlgorithmResult>& results);

private:
	Config* _config;

	PyObject* _module = NULL;
	PyObject* _class = NULL;
	PyObject* _object = NULL;

	PyObject* _func_imageClassify = NULL;
	PyObject* _func_imageClassifyArgs = NULL;
};

#endif