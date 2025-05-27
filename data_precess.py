from net_frame import *
import time
import os
from tqdm import tqdm 

# 加载原始数据
# 方法1
def get_raw_data(data_path = 'aclImdb',istrain = True):
    """获取原始的sentence及标签"""
    start = time.perf_counter()
    path_prefix = os.path.join(data_path,'train' if istrain else 'test')
    data = []
    labels = []
    for label in ['pos','neg']:
        folder = os.path.join(path_prefix,label)

        # 这样写比在最后加loop.set_description(f"read {label}")快50倍
        loop = tqdm(os.listdir(folder),total = len(os.listdir(folder)),desc = 'read ' + label)
        for filename in loop:
        # for filename in os.listdir(folder):
            filename = os.path.join(folder,filename)
            with open(filename,'r',encoding = 'utf-8') as file:
                for line in file:
                    data.append(line.strip())
                    labels.append(1 if label == 'pos' else 0)
    end = time.perf_counter()
    print(f"Load data use time:{end - start}s")
    return data,labels

# 方法2
def get_raw_data2(data_path = 'aclImdb',istrain = True):
    import glob
    """使用glob遍历文件"""
    start = time.perf_counter()
    path_prefix = os.path.join(data_path,'train' if istrain else 'test')
    data = []
    labels = []
    for label in ['pos','neg']:
        folder = os.path.join(path_prefix,label,"*.txt")
        loop = tqdm(glob.glob(folder),total = len(glob.glob(folder)),desc = 'read ' + label)
        for filename in loop:
            with open(filename,'r',encoding = 'utf-8') as file:
                for line in file:
                    data.append(line.strip())
                    labels.append(1 if label == 'pos' else 0)
            loop.set_description(f"read {label}",refresh = False)
    end = time.perf_counter()
    print(f"Load data use time:{end - start}s")
    return data,labels

def build_array(text,vocab,num_steps):
    """文本token to idx,将原始文本标量化"""
    lines = [vocab[l] for l in text]
    arrays = [truncate_pad(line,num_steps,padding_token = vocab['<pad>']) for line in lines]
    return torch.tensor(arrays,dtype = torch.int32)