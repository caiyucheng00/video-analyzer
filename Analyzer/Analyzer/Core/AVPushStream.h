#ifndef __AVPUSHSTREAM__H_
#define __AVPUSHSTREAM__H_

#include <queue>
#include <mutex>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
}

class Config;
class Control;
struct VideoFrame;

class AVPushStream
{
public:
	AVPushStream(Config* config, Control* control);
	~AVPushStream();

	bool connect();     // 连接流媒体服务
	bool reConnect();   // 重连流媒体服务
	void closeConnect();// 关闭流媒体服务的连接

	void pushVideoFrame(unsigned char* data, int size);

	static void EncodeVideoAndWriteStreamThread(void* arg); // 编码视频帧并推流

	AVFormatContext* _fmtCtx = NULL;

	// 视频帧
	AVCodecContext* _videoCodecCtx = NULL;
	AVStream* _videoStream = NULL;
	int _videoIndex = -1;
	// 音频帧

	int _connectCount = 0;

private:
	bool getVideoFrame(VideoFrame*& frame, int& frameQueueSize);// 获取的frame，需要pushReusedVideoFrame
	void clearVideoFrameQueue();

	// bgr24转yuv420p
	unsigned char clipValue(unsigned char x, unsigned char min_val, unsigned char  max_val);
	bool bgr24ToYuv420p(unsigned char* bgrBuf, int w, int h, unsigned char* yuvBuf);

	Config* _config;
	Control* _control;

	std::queue<VideoFrame*> _videoFrameQueue;
	std::mutex           _videoFrameQueueMtx;
};

#endif