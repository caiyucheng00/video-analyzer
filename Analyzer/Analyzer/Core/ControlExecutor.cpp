#include "ControlExecutor.h"
#include "Utils/Log.h"
#include "Utils/Common.h"
#include "Control.h"
#include "Scheduler.h"
#include "AvPullStream.h"
#include "AvPushStream.h"
#include "Analyzer.h"

extern "C" {
#include "libswscale/swscale.h"
#include <libavutil/imgutils.h>
#include <libswresample/swresample.h>
}

ControlExecutor::ControlExecutor(Scheduler* scheduler, Control* control) :
	_scheduler(scheduler),
	_control(control),
	_pullStream(nullptr),
	_pushStream(nullptr),
	_state(false)   // 将初始执行状态设置为false
{
	_control->executorStartTimestamp = getCurTimestamp();
}

ControlExecutor::~ControlExecutor()
{
	std::this_thread::sleep_for(std::chrono::milliseconds(1));

	_state = false;   // 将执行状态设置为false

	for (auto th : _threads) {
		th->join();
	}
	for (auto th : _threads) {
		delete th;
		th = nullptr;
	}
	_threads.clear();

	if (_pullStream) {
		delete _pullStream;
		_pullStream = nullptr;
	}
	if (_pushStream) {
		delete _pushStream;
		_pushStream = nullptr;
	}
	if (_analyzer) {
		delete _analyzer;
		_analyzer = nullptr;
	}
	if (_control) {
		delete _control;
		_control = nullptr;
	}
}

bool ControlExecutor::getState()
{
	return _state;
}

void ControlExecutor::setState_remove()
{
	_state = false;
	_scheduler->removeExecutor(_control);
}

bool ControlExecutor::start(std::string& result_msg)
{
	_pullStream = new AVPullStream(_scheduler->getConfig(), _control);    // 及时delete
	if (_pullStream->connect()) {                                   // 拉流连接成功
		if (_control->pushStream) {                   // 需要推流 
			_pushStream = new AVPushStream(_scheduler->getConfig(), _control);   // 及时delete
			if (!_pushStream->connect()) {     // 推流连接失败
				result_msg = "pull stream connect success, push stream connect error";
				return false;
			}
		}
	}
	else                                                            // 拉流连接失败
	{
		result_msg = "pull stream connect error";
		return false;
	}
	_analyzer = new Analyzer(_scheduler, _control);  // 及时delete

	_state = true;// 将执行状态设置为true 开始执行

	std::thread* th = new std::thread(AVPullStream::ReadThread, this);   // 1.拉流媒体流:pushVideoPacket TO queue
	_threads.push_back(th);

	th = new std::thread(ControlExecutor::DecodeAndAnalyzeVideoThread, this);  // 2.解码视频帧和实时分析视频帧
	_threads.push_back(th);

	if (_control->pushStream) {                                          // 如果推流 3.编码视频帧并推流
		if (_control->videoIndex > -1) {
			th = new std::thread(AVPushStream::EncodeVideoAndWriteStreamThread, this);
			_threads.push_back(th);
		}
	}

	for (auto th : _threads) {
		th->native_handle();
	}

	return true;
}

void ControlExecutor::DecodeAndAnalyzeVideoThread(void* arg)
{

}
