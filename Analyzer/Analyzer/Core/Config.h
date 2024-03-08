#ifndef __CONFIG__H_
#define __CONFIG__H_

#include <string>
#include <vector>

class Config
{
public:
	//************************************
	// Method:    Config
	// FullName:  Config::Config
	// Access:    public 
	// Returns:   
	// Qualifier: 读取json文件，填充成员数据，设置_state=true
	// Parameter: const char * file 配置文件
	// Parameter: const char * ip	本项目ip
	// Parameter: short port		本项目port
	//************************************
	Config(const char* file, const char* ip, short port);
	~Config();

	//************************************
	// Method:    show
	// FullName:  Config::show
	// Access:    public 
	// Returns:   void
	// Qualifier: 展示配置文件属性
	//************************************
	void show();

	// 配置文件属性（本项目ip，port + json）
	const char* file = nullptr;
	const char* serverIp = nullptr;
	short serverPort = 0;

	std::string adminHost;
	int controlExecutorMaxNum = 0;
	bool supportHardwareVideoDecode = false;
	bool supportHardwareVideoEncode = false;
	std::string algorithmType;
	std::string engine;
	std::vector<std::string> algorithmApiHosts;

	// 状态
	bool _state = false;
};

#endif