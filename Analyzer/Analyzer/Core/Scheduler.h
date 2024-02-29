#ifndef __SCHEDULER__H_
#define __SCHEDULER__H_

#include <map>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>

class Config;
class ControlExecutor;
class Control;

class Scheduler
{
public:
	//************************************
	// Method:    Scheduler
	// FullName:  Scheduler::Scheduler
	// Access:    public 
	// Returns:   
	// Qualifier: Config配置文件由Scheduler获取，初始化时_state = false
	// Parameter: Config * config
	//************************************
	Scheduler(Config* config);
	~Scheduler();

	Config* getConfig();
	void setState(bool state);
	bool getState();

	//************************************
	// Method:    loop
	// FullName:  Scheduler::loop
	// Access:    public 
	// Returns:   void
	// Qualifier: _state控制循环
	//			  消费者删除executor实例
	//************************************
	void loop();

	// ApiServer 对应的函数 start
	//************************************
	// Method:    apiControls
	// FullName:  Scheduler::apiControls
	// Access:    public 
	// Returns:   int 返回填充后数组的长度
	// Qualifier: map的所有executor的对应control填充空数组
	// Parameter: std::vector<Control * > & controls 传入空数组
	//************************************
	int  apiControls(std::vector<Control*>& controls);
	//************************************
	// Method:    apiControl
	// FullName:  Scheduler::apiControl
	// Access:    public 
	// Returns:   Control* 获取的结果
	// Qualifier: 获取map的code对应executor的对应control
	// Parameter: std::string & code req解析的
	//************************************
	Control* apiControl(std::string& code);
	//************************************
	// Method:    apiControlAdd
	// FullName:  Scheduler::apiControlAdd
	// Access:    public 
	// Returns:   void
	// Qualifier: 新建control对应的executor
	//            加入map
	// Parameter: Control * control req解析新建的
	// Parameter: int & result_code response
	// Parameter: std::string & result_msg response
	//************************************
	void apiControlAdd(Control* control, int& result_code, std::string& result_msg);
	//************************************
	// Method:    apiControlCancel
	// FullName:  Scheduler::apiControlCancel
	// Access:    public 
	// Returns:   void
	// Qualifier: 根据control查找executor，加入删除队列
	// Parameter: Control * control req解析新建的
	// Parameter: int & result_code response
	// Parameter: std::string & result_msg response
	//************************************
	void apiControlCancel(Control* control, int& result_code, std::string& result_msg);
	// ApiServer 对应的函数 end

	friend class ControlExecutor; // 可访问私有方法

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
	
	std::queue<ControlExecutor*> _deletedExecutorQueue;   // 生产者-消费者 删除executor队列
	std::mutex                   _deletedExecutorQueueMtx;
	std::condition_variable      _deletedExecutorQueueCv;
};

#endif