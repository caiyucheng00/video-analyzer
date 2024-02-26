#视频行为分析系统后台管理

#### 安装
| 程序         | 版本               |
| ---------- |------------------|
| python     | 3.7+             |
| 依赖库      | requirements.txt |

#### 启动配置

~~~
使用端口：9001

//启动服务
python manage.py runserver 0.0.0.0:9001

//管理员用户
admin/admin888

~~~

#### linux 创建python虚拟环境
~~~

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 更新虚拟环境的pip版本
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 在虚拟环境中安装依赖库
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

~~~

#### windows 创建python虚拟环境
~~~
# 创建虚拟环境
python -m venv venv

# 切换到虚拟环境
venv\Scripts\activate

# 更新虚拟环境的pip版本
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 在虚拟环境中安装依赖库
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

~~~

#### API
~~~
# 1.主页
    GET /
    GET /getIndex
    
# 2.1 视频流管理
    GET /stream
    GET /getStreams
# 2.2 预览
    GET /stream/play?app=live&name=test
    GET /stream/play?app=analyzer&name=c4570747232856
    
# 3 算法管理
    GET /behavior
    
# 4.1 布控管理
    GET /control
    GET /getControls
    
    加入布控
    POST /analyzerControlAdd
    {"code": "c4570747232856", "streamUrl": "rtsp://127.0.0.1:554/live/test", "pushStream": true, "pushStreamUrl": "rtsp://127.0.0.1:554/analyzer/c4570747232856", "behaviorCode": "ZHOUJIERUQIN"}
    
    取消布控
    POST /analyzerControlCancel
    
    保存
    POST /controlEdit
#4.2 添加布控
    GET /control/add
    
    保存
    POST /controlAdd
    
    加入布控
    POST /analyzerControlAdd
    
    取消布控
    POST /analyzerControlCancel
#4.3 编辑布控
    GET /control/edit?code=c4570747232856
~~~