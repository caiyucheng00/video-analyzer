#ifndef __CONTROLEXECUTOR__H_
#define __CONTROLEXECUTOR__H_

#include <thread>
#include <queue>
#include <mutex>

class Scheduler;
class Control;
class AVPullStream;
class AVPushStream;
class Analyzer;

class ControlExecutor
{
public:
	explicit ControlExecutor(Scheduler* scheduler, Control* control);
	~ControlExecutor();

	bool getState();
	void setState_remove();

	bool start(std::string& result_msg);

	static void DecodeAndAnalyzeVideoThread(void* arg);// 解码视频帧和实时分析视频帧

	Control* _control;
	Scheduler* _scheduler;
	AVPullStream* _pullStream;
	AVPushStream* _pushStream;
	Analyzer* _analyzer;

private:
	bool _state;
	std::vector<std::thread*> _threads;
};

#endif