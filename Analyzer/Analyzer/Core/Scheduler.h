#ifndef __SCHEDULER__H_
#define __SCHEDULER__H_

#include <map>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include "Config.h"
#include "Control.h"
#include "ControlExecutor.h"

class Scheduler
{
public:
	Scheduler(Config* config);
	~Scheduler();

	Config* getConfig();
	void setState(bool state);
	bool getState();

	void loop();

	// ApiServer 对应的函数 start
	int  apiControls(std::vector<Control*>& controls);
	Control* apiControl(std::string& code);
	void apiControlAdd(Control* control, int& result_code, std::string& result_msg);
	void apiControlCancel(Control* control, int& result_code, std::string& result_msg);
	// ApiServer 对应的函数 end

private:
	int  getExecutorMapSize();
	bool isAdd(Control* control);
	bool addExecutor(Control* control, ControlExecutor* controlExecutor);
	bool removeExecutor(Control* control);//加入到待实际删除队列
	ControlExecutor* getExecutor(Control* control);

	void handleDeleteExecutor();

	Config* _config;
	bool _state;

	std::map<std::string, ControlExecutor*> _executorMap; // <control.code,ControlExecutor*>
	std::mutex                              _executorMapMtx;
	
	std::queue<ControlExecutor*> _deletedExecutorQueue;
	std::mutex                   _deletedExecutorQueueMtx;
	std::condition_variable      _deletedExecutorQueueCv;
};

#endif