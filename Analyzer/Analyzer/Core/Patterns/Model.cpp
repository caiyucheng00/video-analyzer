#include "Model.h"
#include "../Config.h"
#include <opencv2/opencv.hpp>

Model::Model(const std::string path) :
	_path(path)
{
	std::cout << "init model" << std::endl;
	// 从文件中读取已保存的引擎
	std::ifstream engineFile(_path, std::ios::binary);
	std::vector<char> engineData((std::istreambuf_iterator<char>(engineFile)), std::istreambuf_iterator<char>());
	engineFile.close();

	setContext(engineData);
}

Model::~Model()
{
	_context->destroy();
	_engine->destroy();
	_runtime->destroy();
}

nvinfer1::IExecutionContext* Model::getContext()
{
	return _context;
}

void Model::setContext(std::vector<char> engineData)
{
	// 使用读取的数据创建运行时和引擎
	_runtime = nvinfer1::createInferRuntime(_logger);
	_engine = _runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr);

	// 创建执行上下文
	_context = _engine->createExecutionContext();
}

ModelFactory::ModelFactory(Config* config) :
	_config(config)
{

}

ModelFactory::~ModelFactory()
{

}

Model* ModelFactory::getModel(const std::string& behaviorCode)
{
	std::string path = _config->modelPath[behaviorCode];
	if (models.find(behaviorCode) == models.end()) {
		models[behaviorCode] = createModel(path);
	}
	return models[behaviorCode];
}

Model* ModelFactory::createModel(const std::string& path)
{
	return new Model(path);
}
