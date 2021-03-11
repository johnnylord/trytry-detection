import sys
from .yolov3 import YOLOv3Agent


def get_agent_cls(name):
    return getattr(sys.modules[__name__], name)
