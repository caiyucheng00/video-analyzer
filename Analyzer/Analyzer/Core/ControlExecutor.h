#ifndef __CONTROLEXECUTOR__H_
#define __CONTROLEXECUTOR__H_

#include <thread>
#include <queue>
#include <mutex>

class Scheduler;
class Control;

class ControlExecutor
{
public:
	explicit ControlExecutor(Scheduler* scheduler, Control* control);
	~ControlExecutor();

	bool getState();
	void setState_remove();

	bool start(std::string& msg);
	static void DecodeAndAnalyzeVideoThread(void* arg);// 解码视频帧和实时分析视频帧
	Control* _control;

private:
	bool _state;
	std::vector<std::thread*> _threads;
};

#endif