#ifndef __CONFIG__H_
#define __CONFIG__H_

#include <string>
#include <vector>

class Config
{
public:
	Config(const char* file, const char* ip, short port);
	~Config();

	void show();

	// 配置文件（本项目ip，port + json）
	const char* file = nullptr;
	const char* serverIp = nullptr;
	short serverPort = 0;

	std::string adminHost;
	std::string rootVideoDir;
	std::string subVideoDirFormat;
	int controlExecutorMaxNum = 0;
	bool supportHardwareVideoDecode = false;
	bool supportHardwareVideoEncode = false;
	std::vector<std::string> algorithmApiHosts;

	// 状态
	bool _state = false;
};

#endif