#ifndef __MODEL__H_
#define  __MODEL__H_

#include <string>
#include <vector>
#include <NvInfer.h>
#include <../samples/common/logger.h> 

class Config;
class Model
{
public:
	Model(const std::string path);
	~Model();

	nvinfer1::IExecutionContext* getContext();

private:
	void setContext(std::vector<char> engineData);

	std::string _path;
	sample::Logger _logger;
	nvinfer1::IRuntime* _runtime;
	nvinfer1::ICudaEngine* _engine;
	nvinfer1::IExecutionContext* _context;
};


class ModelFactory 
{
public:
	ModelFactory(Config* config);
	~ModelFactory();

	Model* getModel(const std::string& key);

private:
	Model* createModel(const std::string& key);

	Config* _config;
	std::unordered_map<std::string, Model*> models;
};

#endif