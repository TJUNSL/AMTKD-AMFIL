# -*- coding: utf-8 -*-
# coding=utf-8
import datetime
import json
import os
import time
import typing
from threading import Thread

import requests
import xlrd


# there must be a file named 'XXXX.xls' in the path
# The first column is the md5/sha256 value of the sample and the second column is the name of the sample
excel_file = 'D:/Android_Virustotal_Results/Virusshare/sample_file.xls'

api_keys = [
    'your api_keys',
]



threads_num = 8

url = "https://www.virustotal.com/vtapi/v2/file/report"
is_exit = False


class ApiManager:
    api_keys: typing.List[typing.Dict[str, typing.Union[str, datetime.datetime]]]
    """
    [{'key': 'key1', 'last': date}]
    """

    def __init__(self, _api_keys: typing.List[str]):
        self.api_keys = [{
            'key': x,
            'last': datetime.datetime(1970, 1, 1, 0, 0, 0)
        } for x in _api_keys]

    def get_api(self):
        self.api_keys.sort(key=lambda x: x['last'])
        while (datetime.datetime.now() - self.api_keys[0]['last']).seconds < 0.1:
            time.sleep(0.1)
        self.api_keys[0]['last'] = datetime.datetime.now()
        return self.api_keys[0]['key']


def run_crap(files_list: list):
    global is_exit
    api_manager = ApiManager(api_keys)
    for file in files_list:
        if is_exit:
            return
        sha256 = file[0]
        # md5 = file[1]
        # the path you want to save the json files from virustotal
        path = os.path.join('D:/Android_Virustotal_Results/AndroZoo/2023', f'{sha256}.json')
        if os.path.isfile(path) and os.stat(path).st_size > 0:
            continue
        while True:
            try:
                if is_exit:
                    return
                # param = {'resource': md5, 'apikey': api_manager.get_api(), 'allinfo': '1'}
                param = {'resource': sha256, 'apikey': api_manager.get_api(), 'allinfo': '1'}
                res = requests.get(url=url, params=param, proxies=None)
                if res.status_code != 200:
                    raise Exception(f"HTTP request failed with status code: {res.status_code}")
                    # raise Exception(f"HTTP请求失败，状态码: {res.status_code}")

                try:
                    res_text = json.dumps(res.json(), indent=4)
                    # the path you want to save the json files from virustotal
                    with open(os.path.join('D:/Android_Virustotal_Results/AndroZoo/2023', f'{sha256}.json'), "w") as out:
                        out.write(res_text)
                    print(f'---------- end: {sha256}----------')
                    break
                except Exception as e:
                    raise Exception(f'The returned content is not in JSON format: {e.__str__()}')
                    # raise Exception(f'返回的内容不是JSON格式: {e.__str__()}')
            except Exception as e:
                print(f'请求错误：{e.__str__()}')
                print(f'10秒后重试')
                time.sleep(10)


def main():
    # the path you want to save the json files from virustotal
    if not os.path.isdir('D:/Android_Virustotal_Results/AndroZoo/2023'):
        # the path you want to save the json files from virustotal
        os.mkdir('D:/Android_Virustotal_Results/AndroZoo/2023')
    bk = xlrd.open_workbook(excel_file)
    try:
        sh = bk.sheet_by_name("Sheet1")
    except Exception:
        print(f'no sheet in {excel_file} named requested')
        return
    # 获取行数
    nrows = sh.nrows
    # 获取列数
    ncols = sh.ncols
    print(f'nrows {nrows}, ncols {ncols}')

    input_list = [(sh.cell_value(i, 0), sh.cell_value(i, 1)) for i in range(0, nrows)]

    threads: typing.List[Thread] = []
    per = len(input_list) // threads_num
    for i in range(threads_num):
        t = Thread(target=run_crap, daemon=True, kwargs={
            'files_list': input_list[i * per: (i + 1) * per],
        })
        t.start()
        threads.append(t)

    for i in range(threads_num):
        while True:
            try:
                if threads[i].is_alive():
                    time.sleep(10)
                else:
                    break
            except KeyboardInterrupt:
                print('Exit in progress, donot force it to close')
                # print('退出中，不要强行关闭')
                global is_exit
                is_exit = True
                continue


if __name__ == '__main__':
    main()
