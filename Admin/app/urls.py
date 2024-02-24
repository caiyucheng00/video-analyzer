
from django.urls import path
from .views.web import *
from .views.api import *


app_name = 'app'

urlpatterns = [
    path('', web_index),
    path('stream', web_stream),
    path('stream/play', web_stream_play),
    path('behavior', web_behavior),
    path('control', web_control),
    path('control/add', web_control_add),
    path('control/edit', web_control_edit),

    path('warning', web_warning),
    path('profile', web_profile),
    path('notification', web_notification),
    path('login', web_login),
    path('logout', web_logout),

    path('controlAdd', api_controlAdd),
    path('controlEdit', api_controlEdit),
    path('analyzerControlAdd', api_analyzerControlAdd),
    path('analyzerControlCancel', api_analyzerControlCancel),
    path('getControls', api_getControls),
    path('getIndex', api_getIndex),
    path('getStreams', api_getStreams),
    path('getVerifyCode', api_getVerifyCode)
]