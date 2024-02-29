#ifndef __ALGORITHMWITHAPI__H_
#define __ALGORITHMWITHAPI__H_

#include "Algorithm.h"

class Config;

class AlgorithmWithAPI : public Algorithm
{
public:
	AlgorithmWithAPI() = delete;
	AlgorithmWithAPI(Config* config);
	~AlgorithmWithAPI();

	virtual void imageClassify(int height, int width, unsigned char* bgr, std::string& classify_result);

	static bool analy_turboJpeg_compress(int height, int width, int channels, unsigned char* bgr, unsigned char*& out_data, unsigned long* out_size);
	static bool analy_compressBgrAndEncodeBase64(int height, int width, int channels, unsigned char* bgr, std::string& out_base64);

private:
	Config* _config;
};

#endif