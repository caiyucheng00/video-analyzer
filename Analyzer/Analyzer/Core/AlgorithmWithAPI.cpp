#include "AlgorithmWithAPI.h"
#include <opencv2/opencv.hpp>
#include <json/json.h>
#include <turbojpeg.h>
#include "Config.h"
#include "Utils/Base64.h"
#include "Utils/Common.h"
#include "Utils/Request.h"
#include "Utils/Log.h"
//#include <windows.h>

#ifndef WIN32
#include <opencv2/opencv.hpp>
#else
#ifndef _DEBUG
#include <turbojpeg.h>
#else
#include <opencv2/opencv.hpp>
#endif
#endif


AlgorithmWithAPI::AlgorithmWithAPI(Config* config) :
	_config(config)
{

}

AlgorithmWithAPI::~AlgorithmWithAPI()
{

}

void AlgorithmWithAPI::imageClassify(int height, int width, unsigned char* bgr, std::string& classify_result)
{
	cv::Mat image(height, width, CV_8UC3, bgr);
	std::string imageBase64;
	analy_compressBgrAndEncodeBase64(image.rows, image.cols, 3, image.data, imageBase64);

	int randIndex = rand() % _config->algorithmApiHosts.size();
	std::string host = _config->algorithmApiHosts[randIndex];
	std::string url = host + "/image/imageClassify";

	Json::Value param;
	param["appKey"] = "s84dsd#7hf34r3jsk@fs$d#$dd";
	param["algorithm"] = "ResNet50";
	param["image_base64"] = imageBase64;
	std::string data = param.toStyledString();
	param = NULL;

	int64_t start_time = getCurTime();
	Request request;
	std::string response;
	request.post(url.data(), data.data(), response);

	Json::CharReaderBuilder builder;  //json.h
	const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
	Json::Value root;
	JSONCPP_STRING errs;
	if (reader->parse(response.c_str(), response.c_str() + std::strlen(response.c_str()), &root, &errs)) {
		classify_result = root["result"].asString();
	}
	else
	{
		classify_result = "";
	}

	int64_t end_time = getCurTime();

	LOGI("cost time=%i", end_time - start_time);
}

bool AlgorithmWithAPI::analy_turboJpeg_compress(int height, int width, int channels, unsigned char* bgr, unsigned char*& out_data, unsigned long* out_size)
{
#ifdef WIN32
#ifndef _DEBUG

	tjhandle handle = tjInitCompress();
	if (nullptr == handle) {
		return false;
	}

	//pixel_format : TJPF::TJPF_BGR or other
	const int JPEG_QUALITY = 75;
	int pixel_format = TJPF::TJPF_BGR;
	int pitch = tjPixelSize[pixel_format] * width;
	int ret = tjCompress2(handle, bgr, width, pitch, height, pixel_format,
		&out_data, out_size, TJSAMP_444, JPEG_QUALITY, TJFLAG_FASTDCT);

	tjDestroy(handle);

	if (ret != 0) {
		return false;
	}
	return true;
#endif // !_DEBUG
#endif

	return false;
}

bool AlgorithmWithAPI::analy_compressBgrAndEncodeBase64(int height, int width, int channels, unsigned char* bgr, std::string& out_base64)
{
#ifdef WIN32
#ifndef _DEBUG
	unsigned char* jpeg_data = nullptr;
	unsigned long  jpeg_size = 0;

	analy_turboJpeg_compress(height, width, channels, bgr, jpeg_data, &jpeg_size);

	if (jpeg_size > 0 && jpeg_data != nullptr) {

		Base64Encode(jpeg_data, jpeg_size, out_base64);

		free(jpeg_data);
		jpeg_data = nullptr;

		return true;
	}
	else {
		return false;
	}

#endif // !_DEBUG

#else
	cv::Mat bgr_image(height, width, CV_8UC3, bgr);

	std::vector<int> quality = { 100 };
	std::vector<uchar> jpeg_data;
	cv::imencode(".jpg", bgr_image, jpeg_data, quality);

	Base64Encode(jpeg_data.data(), jpeg_data.size(), out_base64);

	return true;
#endif //WIN32

}
