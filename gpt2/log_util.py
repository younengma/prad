# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:28:21 2021
为了效率，日志不写入文件了
@author: mayouneng-jk
"""
import logging

format_str = '%(asctime)s %(levelname)s %(filename)s-%(lineno)d %(message)s'

logging.basicConfig(format=format_str)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
