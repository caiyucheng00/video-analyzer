#ifndef __CONTROL__H_
#define __CONTROL__H_

#include <string>

class Control
{
public:
	bool validateAdd(std::string& result_msg);
	bool validateCancel(std::string& result_msg);

	// 布控请求必需参数
	std::string code;
	std::string streamUrl;
	bool        pushStream = false;
	std::string pushStreamUrl;
	std::string behaviorCode;

	// 通过计算获得的参数
	int64_t executorStartTimestamp = 0;// 执行器启动时毫秒级时间戳（13位）
	float   checkFps = 0;// 算法检测的帧率（每秒检测的次数）
	int     videoWidth = 0;  // 布控视频流的像素宽
	int     videoHeight = 0; // 布控视频流的像素高
	int     videoChannel = 0;
	int     videoIndex = -1;
	int     videoFps = 0;
};

#endif