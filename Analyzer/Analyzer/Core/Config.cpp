#include "Config.h"
#include <fstream>
#include <json/json.h>
#include "Utils/Log.h"
#include "Utils/Version.h"

Config::Config(const char* file, const char* ip, short port) :
	file(file),
	serverIp(ip),
	serverPort(port)
{
	std::ifstream ifs(file, std::ios::binary);

	if (!ifs.is_open()) {  // 读取json文件失败
		LOGE("open %s error", file);
		return;
	}
	else {				   // 读取json文件成功
		Json::CharReaderBuilder builder;    //json.h
		builder["collectComments"] = true;
		JSONCPP_STRING errs;
		Json::Value root;

		if (parseFromStream(builder, ifs, &root, &errs)) { //json文件写入root成功
			this->adminHost = root["adminHost"].asString();
			this->controlExecutorMaxNum = root["controlExecutorMaxNum"].asInt();
			this->supportHardwareVideoDecode = root["supportHardwareVideoDecode"].asBool();
			this->supportHardwareVideoEncode = root["supportHardwareVideoEncode"].asBool();
			this->algorithmType = root["algorithmType"].asString();
			
			Json::Value modelPath = root["modelPath"];
			this->modelPath["PHE_WHEAT"] = modelPath["PHE_WHEAT"].asString();
			this->modelPath["PHE_RICE"] = modelPath["PHE_RICE"].asString();
			this->modelPath["SPIKE_WHEAT"] = modelPath["SPIKE_WHEAT"].asString();
			this->modelPath["SPIKE_RICE"] = modelPath["SPIKE_RICE"].asString();
			this->modelPath["SEEDLING_WHEAT"] = modelPath["SEEDLING_WHEAT"].asString();

			Json::Value algorithmApiHosts = root["algorithmApiHosts"];
			for (auto& item : algorithmApiHosts) {
				this->algorithmApiHosts.push_back(item.asString());
			}

			_state = true;
		}
		else {                                             // 失败
			LOGE("parse %s error", file);
		}

		ifs.close();  // 关闭流
	}
}

Config::~Config()
{

}

void Config::show()
{
	printf("--------%s-------- \n", PROJECT_VERSION);

	printf("config.file=%s\n", file);
	printf("config.serverIp=%s\n", serverIp);
	printf("config.serverPort=%d\n", serverPort);
	printf("config.adminHost=%s\n", adminHost.data());   // c_str()
	printf("config.controlExecutorMaxNum=%d\n", controlExecutorMaxNum);
	printf("config.supportHardwareVideoDecode=%d\n", supportHardwareVideoDecode);
	printf("config.supportHardwareVideoEncode=%d\n", supportHardwareVideoEncode);
	printf("config.algorithmType=%s\n", algorithmType.data());

	if ("api" == this->algorithmType) {
		for (int i = 0; i < algorithmApiHosts.size(); i++)
		{
			printf("config.algorithmApiHosts[%d]=%s\n", i, algorithmApiHosts[i].data());
		}
	}
	else if ("tensorrt" == this->algorithmType) {
		for (auto& item : modelPath) {
			printf("config.modelPath[%s]=%s\n", item.first.data(), item.second.data());
		}
	}
	
	printf("--------end-------- \n");
}
