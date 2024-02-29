#ifndef __AVPULLSTREAM__H_
#define __AVPULLSTREAM__H_

#include <queue>
#include <mutex>
#include <condition_variable>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
}

class Config;
class Control;

class AVPullStream
{
public:
	AVPullStream(Config* config, Control* control);
	~AVPullStream();

	bool connect();     // 连接流媒体服务，填充control计算成员
	bool reConnect();   // 重连流媒体服务
	void closeConnect();// 关闭流媒体服务的连接

	bool getVideoPacket(AVPacket& packet, int& packetQueueSize);  // 从队列获取的pkt，一定要主动释放!!!

	static void ReadThread(void* arg); // 拉流媒体流

	AVFormatContext* _fmtCtx = NULL;

	// 视频帧
	AVCodecContext* _videoCodecCtx = NULL;
	AVStream* _videoStream = NULL;
	// 音频帧

	int _connectCount = 0;

private:
	bool pushVideoPacket(const AVPacket& packet);
	void clearVideoPacketQueue();

	Config* _config;
	Control* _control;

	std::queue<AVPacket> _videoPacketQueue;
	std::mutex           _videoPacketQueueMtx;
};

#endif