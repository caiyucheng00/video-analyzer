#include "ControlExecutor.h"
#include "Utils/Log.h"
#include "Utils/Common.h"
#include "Control.h"
#include "Scheduler.h"
#include "AvPullStream.h"
#include "AvPushStream.h"
#include "Analyzer.h"
#include "Patterns/Model.h"

extern "C" {
#include "libswscale/swscale.h"
#include <libavutil/imgutils.h>
#include <libswresample/swresample.h>
}

ControlExecutor::ControlExecutor(Scheduler* scheduler, Control* control, Model* model) :
	_scheduler(scheduler),
	_control(new Control(*control)),
	_pullStream(nullptr),
	_pushStream(nullptr),
	_state(false),   // 将初始执行状态设置为false
	_model(model)
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
	_analyzer = new Analyzer(_scheduler->getConfig(), _control, _model);  // 及时delete

	_state = true;// 将执行状态设置为true 开始执行

	std::thread* th = new std::thread(AVPullStream::ReadThread, this);   // 1.拉流媒体流:pushVideoPacket TO queue
	_threads.push_back(th);

	th = new std::thread(ControlExecutor::DecodeAndAnalyzeVideoThread, this);  // 2.解码视频帧和实时分析视频帧： getVideoPacket FROM queue + 算法 + pushVideoFrame TO queue
	_threads.push_back(th);

	if (_control->audioIndex > -1) {
		th = new std::thread(ControlExecutor::DecodeAndAnalyzeAudioThread, this); // 2.解码音频帧
		_threads.push_back(th);
	}

	if (_control->pushStream) {                                          // 如果推流 3.编码视频帧并推流
		if (_control->videoIndex > -1) {
			th = new std::thread(AVPushStream::EncodeVideoAndWriteStreamThread, this);
			_threads.push_back(th);
		}

		if (_control->audioIndex > -1) {
			th = new std::thread(AVPushStream::EncodeAudioAndWriteStreamThread, this);  // 3.编码音频帧
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

						executor->_analyzer->checkVideoFrame(cur_is_check, frame_bgr->data[0]);  // 检测

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

void ControlExecutor::DecodeAndAnalyzeAudioThread(void* arg)
{
	ControlExecutor* executor = (ControlExecutor*)arg;

	AVPacket packet; // 未解码的音频帧
	int      packetQueueSize = 0; // 未解码音频帧队列当前长度
	AVFrame* frame = av_frame_alloc();// pkt->解码->frame

	// 音频输入参数start
	int in_channels = executor->_pullStream->_audioCodecCtx->channels;// 输入声道数
	uint64_t in_channel_layout = av_get_default_channel_layout(in_channels);// 输入声道层
	AVSampleFormat in_sample_fmt = executor->_pullStream->_audioCodecCtx->sample_fmt;
	int in_sample_rate = executor->_pullStream->_audioCodecCtx->sample_rate;
	int in_nb_samples = executor->_pullStream->_audioCodecCtx->frame_size;
	// 音频输入参数end

	// 音频重采样输出参数start
	uint64_t out_channel_layout = AV_CH_LAYOUT_STEREO;
	int out_channels = av_get_channel_layout_nb_channels(out_channel_layout);
	AVSampleFormat out_sample_fmt = AV_SAMPLE_FMT_S16;//ffmpeg对于AAC编码的采样点格式默认只支持AV_SAMPLE_FMT_FLTP，通常PCM文件或者播放器播放的音频采样点格式是 AV_SAMPLE_FMT_S16
	int out_sample_rate = 48000;//采样率
	int out_nb_samples = 1024;//每帧单个通道的采样点数
	// 音频重采样输出参数end


	struct SwrContext* swsCtx = swr_alloc();
	swsCtx = swr_alloc_set_opts(swsCtx,
		out_channel_layout,
		out_sample_fmt,
		out_sample_rate,
		in_channel_layout,
		in_sample_fmt,
		in_sample_rate,
		0, NULL);
	swr_init(swsCtx);

	int out_buff_size = av_samples_get_buffer_size(NULL, out_channels, out_nb_samples, out_sample_fmt, 1);
	uint8_t* out_buff = (uint8_t*)av_malloc(out_buff_size);// 重采样得到的PCM

	int ret = -1;
	int64_t frameCount = 0;

	while (executor->getState()) {   // 新建线程条件
		if (executor->_pullStream->getAudioPacket(packet, packetQueueSize)) {      // 填充packet成功
			if (executor->_control->audioIndex > -1) {
				ret = avcodec_send_packet(executor->_pullStream->_audioCodecCtx, &packet);
				if (ret == 0) {
					while (avcodec_receive_frame(executor->_pullStream->_audioCodecCtx, frame) == 0)
					{
						frameCount++;

						swr_convert(swsCtx, &out_buff, out_buff_size, (const uint8_t**)frame->data, frame->nb_samples);

						if (executor->_control->pushStream) {
							// 重采样的参数决定着一帧音频的数据是out_buff_size=4096
							executor->_pushStream->pushAudioFrame(out_buff, out_buff_size);
						}
					}
				}
				else {
					LOGE("avcodec_send_packet : ret=%d", ret);
				}
			}
			av_packet_unref(&packet);
		}
		else
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(5));              // 填充packet失败
		}
	}

	av_frame_free(&frame);
	frame = NULL;

	av_free(out_buff);
	out_buff = NULL;

	swr_free(&swsCtx);
	swsCtx = NULL;
}
