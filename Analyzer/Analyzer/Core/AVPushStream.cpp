#include "AVPushStream.h"
#include "Config.h"
#include "Utils/Log.h"
#include "Utils/Common.h"
#include "Control.h"
#include "ControlExecutor.h"
#include "Analyzer.h"

extern "C" {
#include "libswscale/swscale.h"
#include <libavutil/imgutils.h>
#include <libswresample/swresample.h>
}

AVPushStream::AVPushStream(Config* config, Control* control) :
	_config(config),
	_control(control)
{

}

AVPushStream::~AVPushStream()
{
	closeConnect();
}

bool AVPushStream::connect()
{
	if (avformat_alloc_output_context2(&_fmtCtx, NULL, "rtsp", _control->pushStreamUrl.data()) < 0) {
		LOGI("avformat_alloc_output_context2 error: pushStreamUrl=%s", _control->pushStreamUrl.data());
		return false;
	}

	//video start
	AVCodec* videoCodec = avcodec_find_encoder(AV_CODEC_ID_H264);
	if (!videoCodec) {
		LOGI("avcodec_find_encoder error: pushStreamUrl=%s", _control->pushStreamUrl.data());
		return false;
	}
	_videoCodecCtx = avcodec_alloc_context3(videoCodec);
	if (!_videoCodecCtx) {
		LOGI("avcodec_alloc_context3 error: pushStreamUrl=%s", _control->pushStreamUrl.data());
		return false;
	}
	//int bit_rate = 300 * 1024 * 8;  //压缩后每秒视频的bit位大小 300kB
	int bit_rate = 4096000;
	// CBR：Constant BitRate - 固定比特率
	_videoCodecCtx->flags |= AV_CODEC_FLAG_QSCALE;
	_videoCodecCtx->bit_rate = bit_rate;
	_videoCodecCtx->rc_min_rate = bit_rate;
	_videoCodecCtx->rc_max_rate = bit_rate;
	_videoCodecCtx->bit_rate_tolerance = bit_rate;
	_videoCodecCtx->codec_id = videoCodec->id;
	_videoCodecCtx->pix_fmt = AV_PIX_FMT_YUVJ420P;// 不支持AV_PIX_FMT_BGR24直接进行编码
	_videoCodecCtx->codec_type = AVMEDIA_TYPE_VIDEO;
	_videoCodecCtx->width = _control->videoWidth;
	_videoCodecCtx->height = _control->videoHeight;
	_videoCodecCtx->time_base = { 1,_control->videoFps };
	_videoCodecCtx->gop_size = 5;
	_videoCodecCtx->max_b_frames = 0;
	_videoCodecCtx->thread_count = 1;
	_videoCodecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;   //添加PPS、SPS

	AVDictionary* video_codec_options = NULL;

	//H.264
	if (_videoCodecCtx->codec_id == AV_CODEC_ID_H264) {
		av_dict_set(&video_codec_options, "preset", "superfast", 0);
		av_dict_set(&video_codec_options, "tune", "zerolatency", 0);
	}
	//H.265
	if (_videoCodecCtx->codec_id == AV_CODEC_ID_H265) {
		av_dict_set(&video_codec_options, "preset", "ultrafast", 0);
		av_dict_set(&video_codec_options, "tune", "zero-latency", 0);
	}

	if (avcodec_open2(_videoCodecCtx, videoCodec, &video_codec_options) < 0) {
		LOGI("avcodec_open2 error: pushStreamUrl=%s", _control->pushStreamUrl.data());
		return false;
	}

	_videoStream = avformat_new_stream(_fmtCtx, videoCodec);
	if (!_videoStream) {
		LOGI("avformat_new_stream error: pushStreamUrl=%s", _control->pushStreamUrl.data());
		return false;
	}
	_videoStream->id = _fmtCtx->nb_streams - 1;

	avcodec_parameters_from_context(_videoStream->codecpar, _videoCodecCtx);
	_videoIndex = _videoStream->id;
	// init video end

	av_dump_format(_fmtCtx, 0, _control->pushStreamUrl.data(), 1);

	// open output url
	if (!(_fmtCtx->oformat->flags & AVFMT_NOFILE)) {
		if (avio_open(&_fmtCtx->pb, _control->pushStreamUrl.data(), AVIO_FLAG_WRITE) < 0) {
			LOGI("avio_open error: pushStreamUrl=%s", _control->pushStreamUrl.data());
			return false;
		}
	}

	AVDictionary* fmt_opt = NULL;
	av_dict_set(&fmt_opt, "rw_timeout", "30000000", 0); //设置rtmp/http-flv连接超时（单位 us）
	av_dict_set(&fmt_opt, "stimeout", "30000000", 0);   //设置rtsp连接超时（单位 us）
	av_dict_set(&fmt_opt, "rtsp_transport", "tcp", 0);

	_fmtCtx->video_codec_id = _fmtCtx->oformat->video_codec;

	if (avformat_write_header(_fmtCtx, &fmt_opt) < 0) { // 调用该函数会将所有stream的time_base，自动设置一个值，通常是1/90000或1/1000，这表示一秒钟表示的时间基长度
		LOGI("avformat_write_header error: pushStreamUrl=%s", _control->pushStreamUrl.data());
		return false;
	}

	_connectCount++;

	return true;
}

bool AVPushStream::reConnect()
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

void AVPushStream::closeConnect()
{
	clearVideoFrameQueue();
	std::this_thread::sleep_for(std::chrono::milliseconds(1));

	if (_fmtCtx) {
		// 推流需要释放start
		if (_fmtCtx && !(_fmtCtx->oformat->flags & AVFMT_NOFILE)) {
			avio_close(_fmtCtx->pb);
		}
		// 推流需要释放end

		avformat_free_context(_fmtCtx);
		_fmtCtx = NULL;
	}

	if (_videoCodecCtx) {
		if (_videoCodecCtx->extradata) {
			av_free(_videoCodecCtx->extradata);
			_videoCodecCtx->extradata = NULL;
		}

		avcodec_close(_videoCodecCtx);
		avcodec_free_context(&_videoCodecCtx);
		_videoCodecCtx = NULL;
		_videoIndex = -1;
	}
}

void AVPushStream::pushVideoFrame(unsigned char* data, int size)
{
	VideoFrame* frame = new VideoFrame(VideoFrame::BGR, size, _control->videoWidth, _control->videoHeight);
	frame->size = size;
	memcpy(frame->data, data, size);

	_videoFrameQueueMtx.lock();
	_videoFrameQueue.push(frame);
	_videoFrameQueueMtx.unlock();
}

void AVPushStream::EncodeVideoAndWriteStreamThread(void* arg)
{
	ControlExecutor* executor = (ControlExecutor*)arg;
	int width = executor->_control->videoWidth;
	int height = executor->_control->videoHeight;

	VideoFrame* videoFrame = NULL; // 未编码的视频帧（bgr格式）
	int frameQueueSize = 0; // frame队列当前长度
	AVFrame* frame_yuv420p = av_frame_alloc();
	frame_yuv420p->format = executor->_pushStream->_videoCodecCtx->pix_fmt;
	frame_yuv420p->width = width;
	frame_yuv420p->height = height;

	int frame_yuv420p_buff_size = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, width, height, 1);
	uint8_t* frame_yuv420p_buff = (uint8_t*)av_malloc(frame_yuv420p_buff_size);
	av_image_fill_arrays(frame_yuv420p->data, frame_yuv420p->linesize,frame_yuv420p_buff,AV_PIX_FMT_YUV420P,width, height, 1);

	AVPacket* packet = av_packet_alloc();// 编码后的视频帧

	int ret = -1;
	int64_t  encodeSuccessCount = 0;
	int64_t  frameCount = 0;
	int64_t t1 = 0;
	int64_t t2 = 0;

	while (executor->getState()) {   // 新建线程条件
		if (executor->_pushStream->getVideoFrame(videoFrame, frameQueueSize)) {           // 填充frame（bgr）成功
			executor->_pushStream->bgr24ToYuv420p(videoFrame->data, width, height, frame_yuv420p_buff);
			delete videoFrame;
			videoFrame = nullptr;

			frame_yuv420p->pts = frame_yuv420p->pkt_dts = av_rescale_q_rnd(frameCount,
				executor->_pushStream->_videoCodecCtx->time_base,
				executor->_pushStream->_videoStream->time_base,
				(AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));

			frame_yuv420p->pkt_duration = av_rescale_q_rnd(1,
				executor->_pushStream->_videoCodecCtx->time_base,
				executor->_pushStream->_videoStream->time_base,
				(AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));

			frame_yuv420p->pkt_pos = -1;

			t1 = getCurTime();
			ret = avcodec_send_frame(executor->_pushStream->_videoCodecCtx, frame_yuv420p);
			if (ret >= 0) {                              // 正常
				ret = avcodec_receive_packet(executor->_pushStream->_videoCodecCtx, packet);
				if (ret >= 0) {                // 正常
					t2 = getCurTime();
					encodeSuccessCount++;

					packet->stream_index = executor->_pushStream->_videoIndex;
					packet->pos = -1;
					packet->duration = frame_yuv420p->pkt_duration;

					ret = av_interleaved_write_frame(executor->_pushStream->_fmtCtx, packet);
					if (ret < 0) {
						LOGE("av_interleaved_write_frame error : ret=%d", ret);
					}
				}
				else {                          // 异常
					LOGE("avcodec_receive_packet error : ret=%d", ret);
				}
			}
			else {                                       // 异常
				LOGE("avcodec_send_frame error : ret=%d", ret);
			}

			frameCount++;
		}
		else {                                                                            // 填充frame（bgr）失败
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}

	// 释放
	av_packet_unref(packet);
	packet = NULL;

	av_free(frame_yuv420p_buff);
	frame_yuv420p_buff = NULL;

	av_frame_free(&frame_yuv420p);
	frame_yuv420p = NULL;
}

bool AVPushStream::getVideoFrame(VideoFrame*& frame, int& frameQueueSize)
{
	_videoFrameQueueMtx.lock();
	if (!_videoFrameQueue.empty()) {
		frame = _videoFrameQueue.front();
		_videoFrameQueue.pop();
		frameQueueSize = _videoFrameQueue.size();
		_videoFrameQueueMtx.unlock();
		return true;
	}
	else {
		frameQueueSize = 0;
		_videoFrameQueueMtx.unlock();
		return false;
	}
}

void AVPushStream::clearVideoFrameQueue()
{
	_videoFrameQueueMtx.lock();
	while (!_videoFrameQueue.empty()) {
		VideoFrame* frame = _videoFrameQueue.front();
		_videoFrameQueue.pop();
		delete frame;
		frame = NULL;
	}
	_videoFrameQueueMtx.unlock();
}

unsigned char AVPushStream::clipValue(unsigned char x, unsigned char min_val, unsigned char  max_val) {

	if (x > max_val) {
		return max_val;
	}
	else if (x < min_val) {
		return min_val;
	}
	else {
		return x;
	}
}

bool AVPushStream::bgr24ToYuv420p(unsigned char* bgrBuf, int w, int h, unsigned char* yuvBuf) {

	unsigned char* ptrY, * ptrU, * ptrV, * ptrRGB;
	memset(yuvBuf, 0, w * h * 3 / 2);
	ptrY = yuvBuf;
	ptrU = yuvBuf + w * h;
	ptrV = ptrU + (w * h * 1 / 4);
	unsigned char y, u, v, r, g, b;

	for (int j = 0; j < h; ++j) {

		ptrRGB = bgrBuf + w * j * 3;
		for (int i = 0; i < w; i++) {

			b = *(ptrRGB++);
			g = *(ptrRGB++);
			r = *(ptrRGB++);


			y = (unsigned char)((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
			u = (unsigned char)((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
			v = (unsigned char)((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
			*(ptrY++) = clipValue(y, 0, 255);
			if (j % 2 == 0 && i % 2 == 0) {
				*(ptrU++) = clipValue(u, 0, 255);
			}
			else {
				if (i % 2 == 0) {
					*(ptrV++) = clipValue(v, 0, 255);
				}
			}
		}
	}
	return true;

}