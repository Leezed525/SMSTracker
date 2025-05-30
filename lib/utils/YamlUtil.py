"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/18 13:33
"""

import os

import yaml


class YamlUtil():
    def __init__(self, yaml_path, yaml_name):
        """
        初始化,当前这个文件操作效率不高，懒得优化了，随便写的，哪天心情好了在优化
        :param yaml_path:
        :param yaml_name:
        """
        self.yaml_path = yaml_path
        self.yaml_name = yaml_name
        self.yaml_file = os.path.join(self.yaml_path, self.yaml_name)

        # 判断当前路径下文件是否存在，如果存在则读取，如果不存在则创建
        if not os.path.exists(self.yaml_file):
            with open(self.yaml_file, 'w') as f:
                f.write('')
            self.content = {}
            self.write_yaml()

        self.content = self.read_yaml()  # 读取到的yaml文件内容

    def __getattr__(self, key):
        return self.get_value(key)

    def get_value(self, key):
        """
        获取yaml中的值
        :param key: key路径,数组形式 ['a','b','c'] 或者字符串形式 'a.b.c'
        :return: key对应的值
        """
        really_key_path = self.is_exist_key(key)

        content = self.content
        for key in really_key_path:
            content = content[key]
        return content

    def add_root_key(self, key, value=None):
        """
        添加根key
        :param key: 添加的key
        :param value: 添加的值
        """
        if key not in self.content:
            self.content[key] = {} if value is None else value
        else:
            raise Exception('key已存在')

    def add_key(self, key_path, value=None):
        """
        添加key 当前只能添加最后一个key,比如a.b.c,只能添加c,如果b不存在就会报错
        :param key_path: key路径,数组形式 ['a','b','c'] 或者字符串形式 'a.b.c'
        :param value: 添加的值
        """
        # 判断key_path的类型，如果是字符串则转换为数组
        really_key_path = self.is_exist_key(key_path)

        content = self.content
        for key in really_key_path[:-1]:
            content = content[key]

        if really_key_path[-1] in content:
            raise Exception("当前key已存在")

        content[really_key_path[-1]] = {} if value is None else value

    def debug(self):
        print(self.content)

    # 修改某个key中的内容
    def modify_key(self, key_path, value=None):
        """
        修改内存中yaml中某个key的内容
        :param key_path: key路径,数组形式 ['a','b','c'] 或者字符串形式 'a.b.c'
        :param value: 修改后的值
        """
        # 判断key_path的类型，如果是字符串则转换为数组
        if value is None:
            value = {}
        really_key_path = self.is_exist_key(key_path, check_last=True)
        content = self.content
        for key in really_key_path[:-1]:
            content = content[key]
        content[really_key_path[-1]] = {} if value is None else value

    def is_exist_key(self, key_path, check_last=False):
        """
        判断yaml中是否存在某个key路径
        :param key_path: key_path,数组形式 ['a','b','c'] 或者字符串形式 'a.b.c'
        :param check_last: 是否检查最后一个key,因为对于最后一个key来说，可能是空的，所以默认不检查
        :return: 返回key_path,如果不存在key_path为[]
        """
        # 判断key_path的类型，如果是字符串则转换为数组
        really_key_path = key_path.split('.') if isinstance(key_path, str) else key_path

        assert isinstance(really_key_path, list), 'key_path必须是list或者str类型'

        content = self.content
        for index, key in enumerate(really_key_path):

            if not check_last and index == len(really_key_path) - 1:
                return really_key_path

            # if index == len(really_key_path) - 1 and content == {}:
            #     return True, really_key_path

            if key in content:
                content = content[key]
            else:
                raise Exception('key错误')
        return really_key_path

    def read_yaml(self):
        """
        读取整个yaml文件
        :return: 读取到的文件内容
        """
        with open(self.yaml_file, 'r') as f:
            content = yaml.safe_load(f)
        return content

    def write_yaml(self):
        """
        将content写入yaml文件
        """
        with open(self.yaml_file, 'w') as f:
            yaml.safe_dump(self.content, f)
