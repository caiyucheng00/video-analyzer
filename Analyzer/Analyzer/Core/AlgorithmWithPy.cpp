#include "AlgorithmWithPy.h"
#include <opencv2/opencv.hpp>
#include "Utils/Log.h"
#include "Utils/Common.h"

#ifndef _DEBUG
#include "numpy/arrayobject.h"
#endif

#ifndef _DEBUG
size_t init_numpy() {
	import_array();
}
#endif

AlgorithmWithPy::AlgorithmWithPy(Config* config) :
	_config(config)
{
	Py_Initialize();
#ifndef _DEBUG
	init_numpy();
#endif
	_module = PyImport_ImportModule("Algorithm");
	_class = PyObject_GetAttrString(_module, "Algorithm");

	PyObject* pyArgs = PyTuple_New(2);
	PyTuple_SetItem(pyArgs, 0, Py_BuildValue("s", "sun"));
	PyTuple_SetItem(pyArgs, 1, Py_BuildValue("i", "30"));

	_object = PyEval_CallObject(_class, pyArgs);   // 实例化

	_func_imageClassify = PyObject_GetAttrString(_object, "objectDetect");
	_func_imageClassifyArgs = PyTuple_New(2);
}

AlgorithmWithPy::~AlgorithmWithPy()
{
	Py_Finalize();
}

void AlgorithmWithPy::imageClassify(int height, int width, unsigned char* bgr, std::string& classify_result)
{
	cv::Mat image(height, width, CV_8UC3, bgr);
	//cv::Mat image = cv::imread("D:\\Developer\\Visual Studio\\Project\\PythonAPI\\test.jpg");

#ifndef _DEBUG
	// cv::Mat->numpy
	int r = image.rows;
	int c = image.cols;
	int chnl = image.channels();
	int nElem = r * c * chnl;
	uchar* imageData = new uchar[nElem];
	std::memcpy(imageData, image.data, nElem * sizeof(uchar));
	npy_intp mdim[] = { r, c, chnl };
	PyObject* pyImage = PyArray_SimpleNewFromData(chnl, mdim, NPY_UINT8, (void*)imageData);

	PyTuple_SetItem(_func_imageClassifyArgs, 0, Py_BuildValue("i", 0));
	PyTuple_SetItem(_func_imageClassifyArgs, 1, pyImage);

#else
	//cv::Mat->imageBase64
	std::string imageBase64;
	Common_CompressAndEncodeBase64(image.rows, image.cols, 3, image.data, imageBase64);

	PyTuple_SetItem(mFunc_objectDetectArgs, 0, Py_BuildValue("i", 1));
	PyTuple_SetItem(mFunc_objectDetectArgs, 1, Py_BuildValue("s", imageBase64.data()));
#endif
	
	PyObject* pyResponse = PyEval_CallObject(_func_imageClassify, _func_imageClassifyArgs);

	char* response_c = NULL;
	PyArg_Parse(pyResponse, "s", &response_c);

	if (NULL != response_c) {
		std::string response(response_c);
		LOGI("response:%s \n", response.data());
		classify_result = response;
	}

#ifndef _DEBUG
	delete[]imageData;
	imageData = NULL;
	Py_CLEAR(pyResponse);
#endif

	pyResponse = NULL;
}
