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

struct VideoFrame
{
public:
	enum VideoFrameType
	{
		BGR = 0,
		YUV420P,

	};
	VideoFrame(VideoFrameType type, int size, int width, int height) {
		this->type = type;
		this->size = size;
		this->width = width;
		this->height = height;
		this->data = new uint8_t[this->size];

	}
	~VideoFrame() {
		delete[] this->data;
		this->data = nullptr;
	}

	VideoFrameType type;
	int size;
	int width;
	int height;
	uint8_t* data;
	bool happen = false;// 是否发生事件
	float happenScore = 0;// 发生事件的分数


};

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