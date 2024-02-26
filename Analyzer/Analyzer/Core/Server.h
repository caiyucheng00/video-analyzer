#ifndef __SERVER__H_
#define __SERVER__H_

class Server {
public:
	//************************************
	// Method:    Server
	// FullName:  Server::Server
	// Access:    public 
	// Returns:   
	// Qualifier: WSAStartup编程
	//************************************
	explicit Server();
	~Server();

	//************************************
	// Method:    start
	// FullName:  Server::start
	// Access:    public 
	// Returns:   void
	// Qualifier: 新建线程：设置路由
	//					   绑定socket 0.0.0.0:9002
	// Parameter: void * arg scheduler
	//************************************
	void start(void* arg);
};

//************************************
// Method:    api_index
// FullName:  api_index
// Access:    public 
// Returns:   void
// Qualifier: 展示urls
// Parameter: struct evhttp_request * req
// Parameter: void * arg
//************************************
void api_index(struct evhttp_request* req, void* arg);
//************************************
// Method:    api_controls
// FullName:  api_controls
// Access:    public 
// Returns:   void
// Qualifier: 获取数组，所有control
// Parameter: struct evhttp_request * req 
// Parameter: void * arg scheduler
//************************************
void api_controls(struct evhttp_request* req, void* arg);
//************************************
// Method:    api_control
// FullName:  api_control
// Access:    public 
// Returns:   void
// Qualifier: 根据req 获取一个control
// Parameter: struct evhttp_request * req
// Parameter: void * arg scheduler
//************************************
void api_control(struct evhttp_request* req, void* arg);
//************************************
// Method:    api_control_add
// FullName:  api_control_add
// Access:    public 
// Returns:   void
// Qualifier: 根据req 新建一个control
// Parameter: struct evhttp_request * req
// Parameter: void * arg scheduler
//************************************
void api_control_add(struct evhttp_request* req, void* arg);
//************************************
// Method:    api_control_cancel
// FullName:  api_control_cancel
// Access:    public 
// Returns:   void
// Qualifier: 根据req 取消一个control
// Parameter: struct evhttp_request * req
// Parameter: void * arg
//************************************
void api_control_cancel(struct evhttp_request* req, void* arg);

void parse_get(struct evhttp_request* req, struct evkeyvalq* params);
void parse_post(struct evhttp_request* req, char* buff);
#endif
