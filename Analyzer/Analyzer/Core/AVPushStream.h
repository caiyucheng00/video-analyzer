#ifndef __AVPUSHSTREAM__H_
#define __AVPUSHSTREAM__H_

class Config;
class Control;

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
};

#endif