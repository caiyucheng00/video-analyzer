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
class Model;

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

struct AudioFrame
{
public:
	AudioFrame(int size) {
		this->size = size;
		this->data = new uint8_t[this->size];
	}
	~AudioFrame() {
		delete[] this->data;
		this->data = NULL;
	}

	int size;
	uint8_t* data;
};

class ControlExecutor
{
public:
	//************************************
	// Method:    ControlExecutor
	// FullName:  ControlExecutor::ControlExecutor
	// Access:    public 
	// Returns:   
	// Qualifier: 初始state = false
	//			  设置control的执行器启动时毫秒级时间戳
	// Parameter: Scheduler * scheduler
	// Parameter: Control * control
	//************************************
	explicit ControlExecutor(Scheduler* scheduler, Control* control, Model* model);
	~ControlExecutor();   // 设置state = false

	bool getState();
	void setState_remove();  // 重连失败启用

	//************************************
	// Method:    start
	// FullName:  ControlExecutor::start
	// Access:    public 
	// Returns:   bool 是否成功
	// Qualifier: 新建类-AVPullStream/AVPushStream/Analyzer
	//			  state=true
	//            拉流媒体流+解码视频帧和实时分析视频帧+编码视频帧并推流
	// Parameter: std::string & result_msg
	//************************************
	bool start(std::string& result_msg);

	static void DecodeAndAnalyzeVideoThread(void* arg);// 解码视频帧和实时分析视频帧
	static void DecodeAndAnalyzeAudioThread(void* arg);// 解码音频帧

	Control* _control;
	Scheduler* _scheduler;
	AVPullStream* _pullStream;
	AVPushStream* _pushStream;
	Analyzer* _analyzer;
	Model* _model;

private:
	bool _state;
	std::vector<std::thread*> _threads;
};

#endif