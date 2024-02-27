#include "AVPullStream.h"
#include "Config.h"
#include "Utils/Log.h"
#include "Utils/Common.h"
#include "Control.h"
#include "ControlExecutor.h"

AVPullStream::AVPullStream(Config* config, Control* control) :
	_config(config),
	_control(control)
{

}

AVPullStream::~AVPullStream()
{
	closeConnect();
}

bool AVPullStream::connect()
{
	_fmtCtx = avformat_alloc_context();

	AVDictionary* fmt_opt = NULL;
	av_dict_set(&fmt_opt, "rtsp_transport", "tcp", 0); //设置rtsp底层网络协议 tcp or udp
	av_dict_set(&fmt_opt, "stimeout", "3000000", 0);   //设置rtsp连接超时（单位 us）
	av_dict_set(&fmt_opt, "rw_timeout", "3000000", 0); //设置rtmp/http-flv连接超时（单位 us）

	int ret = avformat_open_input(&_fmtCtx, _control->streamUrl.data(), NULL, &fmt_opt);
	if (ret != 0) {
		LOGE("avformat_open_input error: url=%s ", _control->streamUrl.data());
		return false;
	}

	if (avformat_find_stream_info(_fmtCtx, NULL) < 0) {
		LOGE("avformat_find_stream_info error");
		return false;
	}

	// video start
	_control->videoIndex = -1;
	for (int i = 0; i < _fmtCtx->nb_streams; i++) {
		if (_fmtCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
			_control->videoIndex = i;    // 确定视频流
			break;
		}
	}

	if (_control->videoIndex > -1) {                                       // 存在视频流
		AVCodecParameters* videoCodecPar = _fmtCtx->streams[_control->videoIndex]->codecpar;
		
		AVCodec* videoCodec = NULL;
		if (_config->supportHardwareVideoDecode) {       // 支持硬解
			if (AV_CODEC_ID_H264 == videoCodecPar->codec_id) {
				if (!videoCodec) {
					videoCodec = avcodec_find_decoder_by_name("h264_cuvid");// 英伟达独显
					if (videoCodec) {
						LOGI("avcodec_find_decoder_by_name = h264_cuvid");
					}
				}
			}
		}

		if (!videoCodec) {
			videoCodec = avcodec_find_decoder(videoCodecPar->codec_id);
			if (!videoCodec) {
				LOGE("avcodec_find_decoder error");
				return false;
			}
		}

		_videoCodecCtx = avcodec_alloc_context3(videoCodec);
		if (avcodec_parameters_to_context(_videoCodecCtx, videoCodecPar) != 0) {
			LOGE("avcodec_parameters_to_context error");
			return false;
		}
		if (avcodec_open2(_videoCodecCtx, videoCodec, NULL) < 0) {
			LOGE("avcodec_open2 error");
			return false;
		}

		_videoStream = _fmtCtx->streams[_control->videoIndex];
		if (0 == _videoStream->avg_frame_rate.den) {
			LOGE("videoIndex=%d,videoStream->avg_frame_rate.den = 0", _control->videoIndex);
			_control->videoFps = 25;
		}
		else {    // 算出fps
			_control->videoFps = _videoStream->avg_frame_rate.num / _videoStream->avg_frame_rate.den;
		}

		_control->videoWidth = _videoCodecCtx->width;   //计算值
		_control->videoHeight = _videoCodecCtx->height;
		_control->videoChannel = 3;
	}
	else                                                                    // 不存在视频流
	{
		LOGE("av_find_best_stream video error videoIndex=%d", _control->videoIndex);
		return false;
	}
	// video end;

	// audio start

	// audio end

	if (_control->videoIndex <= -1) {
		return false;
	}

	_connectCount++;

	return true;
}

bool AVPullStream::reConnect()
{
	if (_connectCount <= 100) {
		closeConnect();

		if (connect()) {
			return true;
		}
		else {
			return false;
		}
	}

	return false;
}

void AVPullStream::closeConnect()
{
	clearVideoPacketQueue();          // 清空队列
	std::this_thread::sleep_for(std::chrono::milliseconds(1));

	if (_videoCodecCtx) {
		avcodec_close(_videoCodecCtx);
		avcodec_free_context(&_videoCodecCtx);
		_videoCodecCtx = NULL;
		_control->videoIndex = -1;
	}

	if (_fmtCtx) {
		avformat_close_input(&_fmtCtx);
		avformat_free_context(_fmtCtx);
		_fmtCtx = NULL;
	}
}

bool AVPullStream::getVideoPacket(AVPacket& packet, int& packetQueueSize)
{
	_videoPacketQueueMtx.lock();

	if (!_videoPacketQueue.empty()) {
		packet = _videoPacketQueue.front();
		_videoPacketQueue.pop();
		packetQueueSize = _videoPacketQueue.size();
		_videoPacketQueueMtx.unlock();
		return true;
	}
	else {
		_videoPacketQueueMtx.unlock();
		return false;
	}
}

void AVPullStream::ReadThread(void* arg)
{
	ControlExecutor* executor = (ControlExecutor*)arg;    // 附属于
	int continuity_error_count = 0;

	AVPacket packet;
	while (executor->getState()) {     // 新建线程条件
		if (av_read_frame(executor->_pullStream->_fmtCtx, &packet) >= 0) {    // 成功
			continuity_error_count = 0;

			if (packet.stream_index == executor->_control->videoIndex) {  // 视频
				executor->_pullStream->pushVideoPacket(packet);
				std::this_thread::sleep_for(std::chrono::milliseconds(30));
			}
			else                                                         // 非视频
			{
				av_packet_unref(&packet);
			}
		}
		else                                                                   // 失败
		{
			av_packet_unref(&packet);
			continuity_error_count++;
			if (continuity_error_count > 5) {//大于5秒重启拉流连接
				LOGE("av_read_frame error, continuity_error_count = %d (s)", continuity_error_count);

				if (executor->_pullStream->reConnect()) {   // 重连接成功
					continuity_error_count = 0;
					LOGI("reConnect success : mConnectCount=%d", executor->_pullStream->_connectCount);
				}
				else                                        // 重连接失败
				{
					LOGI("reConnect error : mConnectCount=%d", executor->_pullStream->_connectCount);
					executor->setState_remove();        // state=flase 删除control/executor
				}
			}
			else {
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			}
		}
	}
}

bool AVPullStream::pushVideoPacket(const AVPacket& packet)
{
	if (av_packet_make_refcounted((AVPacket*)&packet) < 0) {
		return false;
	}

	_videoPacketQueueMtx.lock();   // ==================
	_videoPacketQueue.push(packet);
	_videoPacketQueueMtx.unlock(); // ==================

	return true;
}

void AVPullStream::clearVideoPacketQueue()
{
	_videoPacketQueueMtx.lock();
	while (!_videoPacketQueue.empty())
	{
		AVPacket packet = _videoPacketQueue.front();
		_videoPacketQueue.pop();
		av_packet_unref(&packet);
	}
	_videoPacketQueueMtx.unlock();
}
