#include "Scheduler.h"
#include "Config.h"
#include "Control.h"
#include "ControlExecutor.h"
#include "Utils/Log.h"
#include "Patterns/Model.h"

Scheduler::Scheduler(Config* config) :
	_config(config),
	_state(false)
{
	_factory = new ModelFactory(_config);
}

Scheduler::~Scheduler()
{
	if (_factory) {
		delete _factory;
		_factory = nullptr;
	}
}

Config* Scheduler::getConfig()
{
	return _config;
}

void Scheduler::setState(bool state)
{
	_state = state;
}

bool Scheduler::getState()
{
	return _state;
}

void Scheduler::loop()
{
	LOGI("Loop Start");

	int64_t l = 0;
	while (_state)
	{
		++l;
		handleDeleteExecutor();
	}

	LOGI("Loop End");
}

int Scheduler::apiControls(std::vector<Control*>& controls)
{
	int len = 0;

	_executorMapMtx.lock();   // ==============
	for (auto iter = _executorMap.begin(); iter != _executorMap.end(); ++iter) {
		++len;
		controls.push_back(iter->second->_control);      //controls数组填充
	}
	_executorMapMtx.unlock(); // ==============

	return len;
}

Control* Scheduler::apiControl(std::string& code)
{
	Control* control = nullptr;

	_executorMapMtx.lock();   // ==============
	for (auto iter = _executorMap.begin(); iter != _executorMap.end(); ++iter) {
		if (iter->first == code) {                 // 根据code获取control
			control = iter->second->_control;   
		}
	}
	_executorMapMtx.unlock(); // ==============

	return control;
}

void Scheduler::apiControlAdd(Control* control, int& result_code, std::string& result_msg)
{
	if (isAdd(control)) {  // 已经添加布控了
		result_msg = "the control is running";
		result_code = 1000;
		return;
	}

	if (getExecutorMapSize() >= _config->controlExecutorMaxNum) {   // 大于config限制数量
		result_msg = "the number of control exceeds the limit";
		result_code = 0;
	}
	else                                                            // 满足config限制数量
	{
		ControlExecutor* executor = new ControlExecutor(this, control, _factory->getModel(control->behaviorCode));  // 及时delete

		if (executor->start(result_msg)) {                       //executor成功
			if (addExecutor(control, executor)) {  //添加executor到map成功
				result_msg = "add success";
				result_code = 1000;
			}
			else                                   //添加executor到map失败
			{
				delete executor;
				executor = nullptr;
				result_msg = "add error";
				result_code = 0;
			}
		}
		else                                                     //executor失败
		{
			delete executor;
			executor = nullptr;
			result_code = 0;
		}
	}
}

void Scheduler::apiControlCancel(Control* control, int& result_code, std::string& result_msg)
{
	ControlExecutor* executor = getExecutor(control);   // 根据control查找executor
	if (executor) {                                      // 查找成功
		if (executor->getState()) {
			result_msg = "control is running, ";
		}
		else
		{
			result_msg = "control is not running, ";
		}

		removeExecutor(control);

		result_msg += "remove success";
		result_code = 1000;
		return;
	}
	else {                                                // 查找失败
		result_msg = "there is no such control";
		result_code = 0;
		return;
	}
}

int Scheduler::getExecutorMapSize()
{
	_executorMapMtx.lock();   // ==============
	int size = _executorMap.size();
	_executorMapMtx.unlock(); // ==============

	return size;
}

bool Scheduler::isAdd(Control* control)
{
	_executorMapMtx.lock();   // ==============
	bool added = _executorMap.end() != _executorMap.find(control->code);
	_executorMapMtx.unlock(); // ==============

	return added;
}

bool Scheduler::addExecutor(Control* control, ControlExecutor* controlExecutor)
{
	bool res = false;

	_executorMapMtx.lock();   // ==============
	if (_executorMap.size() < _config->controlExecutorMaxNum) { // executorMap数量小于config规定数量
		bool added = _executorMap.end() != _executorMap.find(control->code);   //true:已经添加  false：还未添加
		if (!added) {  //还未添加
			_executorMap.insert(std::pair<std::string, ControlExecutor*>(control->code, controlExecutor));
			res = true;
		}
	}
	_executorMapMtx.unlock(); // ==============

	return res;
}

bool Scheduler::removeExecutor(Control* control)
{
	bool res = false;

	_executorMapMtx.lock();   // ==============
	auto iter = _executorMap.find(control->code);  
	if (_executorMap.end() != iter) {  //  找到需要删除的
		ControlExecutor* executor = iter->second;
		std::unique_lock<std::mutex> lck(_deletedExecutorQueueMtx);
		_deletedExecutorQueue.push(executor);         // 加入删除队列 消费生产
		_deletedExecutorQueueCv.notify_one();

		res = _executorMap.erase(control->code) != 0;   // 删除map里面的
	}
	_executorMapMtx.unlock(); // ==============

	return res;
}

ControlExecutor* Scheduler::getExecutor(Control* control)
{
	ControlExecutor* executor = nullptr;

	_executorMapMtx.lock();   // ==============
	auto iter = _executorMap.find(control->code);
	if (_executorMap.end() != iter) {  //  找到需要的
		executor = iter->second;
	}
	_executorMapMtx.unlock(); // ==============

	return executor;
}

void Scheduler::handleDeleteExecutor()
{
	std::unique_lock<std::mutex> lck(_deletedExecutorQueueMtx);
	_deletedExecutorQueueCv.wait(lck);

	while (!_deletedExecutorQueue.empty())
	{
		ControlExecutor* executor = _deletedExecutorQueue.front();
		_deletedExecutorQueue.pop();
		
		delete executor;
		executor = nullptr;
	}
}
