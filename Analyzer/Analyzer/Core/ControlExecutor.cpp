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
	_control(new Control(*control)),
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

	th = new std::thread(ControlExecutor::DecodeAndAnalyzeVideoThread, this);  // 2.解码视频帧和实时分析视频帧： 算法+pushVideoFrame TO queue
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
	ControlExecutor* executor = (ControlExecutor*)arg;
	int width = executor->_pullStream->_videoCodecCtx->width;
	int height = executor->_pullStream->_videoCodecCtx->height;

	AVPacket packet;     // 未解码的视频帧
	int packetQueueSize = 0; // packet队列当前长度

	AVFrame* frame_yuv420p = av_frame_alloc();// pkt->解码->frame
	AVFrame* frame_bgr = av_frame_alloc();

	int frame_bgr_buff_size = av_image_get_buffer_size(AV_PIX_FMT_BGR24, width, height, 1);
	uint8_t* frame_bgr_buff = (uint8_t*)av_malloc(frame_bgr_buff_size);
	av_image_fill_arrays(frame_bgr->data, frame_bgr->linesize, frame_bgr_buff, AV_PIX_FMT_BGR24, width, height, 1);

	SwsContext* swsCtx = sws_getContext(width, height,
		executor->_pullStream->_videoCodecCtx->pix_fmt,
		executor->_pullStream->_videoCodecCtx->width,
		executor->_pullStream->_videoCodecCtx->height,
		AV_PIX_FMT_BGR24,
		SWS_BICUBIC, nullptr, nullptr, nullptr);
	
	int fps = executor->_control->videoFps;

	//算法检测参数start
	bool cur_is_check = false;// 当前帧是否进行算法检测
	int  continuity_check_count = 0;// 当前连续进行算法检测的帧数
	int  continuity_check_max_time = 3000;//连续进行算法检测，允许最长的时间。单位毫秒
	int64_t continuity_check_start = getCurTime();//单位毫秒
	int64_t continuity_check_end = 0;
	//算法检测参数end

	int ret = -1;
	int64_t frameCount = 0;

	while (executor->getState()) {   // 新建线程条件
		if (executor->_pullStream->getVideoPacket(packet, packetQueueSize)) {   // 填充packet成功
			if (executor->_control->videoIndex > -1) {                 // 存在视频流
				ret = avcodec_send_packet(executor->_pullStream->_videoCodecCtx, &packet);
				if (ret == 0) {                           // 正常
					ret = avcodec_receive_frame(executor->_pullStream->_videoCodecCtx, frame_yuv420p);
					if (ret == 0) {                 // 正常
						frameCount++;

						sws_scale(swsCtx, frame_yuv420p->data, frame_yuv420p->linesize, 0, height,
							frame_bgr->data, frame_bgr->linesize);

						if (packetQueueSize == 0) {
							cur_is_check = true;
						}
						else {
							cur_is_check = false;
						}
						if (cur_is_check) {
							continuity_check_count += 1;
						}
						continuity_check_end = getCurTime();
						if (continuity_check_end - continuity_check_start > continuity_check_max_time) {
							executor->_control->checkFps = float(continuity_check_count) / (float(continuity_check_end - continuity_check_start) / 1000);
							continuity_check_count = 0;
							continuity_check_start = getCurTime();
						}

						executor->_analyzer->checkVideoFrame(cur_is_check, frameCount, frame_bgr->data[0]);  // 检测

						if (executor->_control->pushStream) {
							executor->_pushStream->pushVideoFrame(frame_bgr->data[0], frame_bgr_buff_size);  // 结果推流
						}
					}
					else {                          // 异常
						LOGE("avcodec_receive_frame error : ret=%d", ret);
					}
				}
				else {                                    // 异常
					LOGE("avcodec_send_packet error : ret=%d", ret);
				}
			}
			// 队列获取的pkt，必须释放!!!
			av_packet_unref(&packet);
		}
		else                                                                    // 填充packet失败
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}

	// 释放
	av_frame_free(&frame_yuv420p);
	frame_yuv420p = NULL;

	av_frame_free(&frame_bgr);
	frame_bgr = NULL;

	av_free(frame_bgr_buff);
	frame_bgr_buff = NULL;

	sws_freeContext(swsCtx);
	swsCtx = NULL;
}
