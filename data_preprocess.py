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
    print(f"Load raw data use time:{end - start}s")
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

def get_iter(source,labels,vocab,batch_size = 64,num_steps = 500):
    source_arrays = build_array(source,vocab,num_steps)
    # target_arrays = torch.tensor(train_labels).reshape(-1,1)
    target_arrays = torch.tensor(labels).to(torch.long) # 类别索引格式，dtype = long
    dataset = data.TensorDataset(source_arrays,target_arrays)
    return data.DataLoader(dataset,batch_size,shuffle = True)

def get_data():
    """获得训练测试的迭代器及各自的vocab"""
    train_data,train_labels = get_raw_data()
    test_data,test_labels = get_raw_data(istrain = False)
    source_train = tokenize(train_data,token = 'word')
    source_test = tokenize(test_data,token = 'word')
    source = source_train + source_test # 训练与测试的corpus拼接
    print("Build vocab....")
    # train_vocab = Vocab(source_train,reserved_tokens = ['<pad>'])
    # test_vocab = Vocab(source_test,reserved_tokens = ['<pad>'])
    vocab = Vocab(source,min_freq = 5,reserved_tokens = ['<pad>']) # 设置min_freq = 5，以减少embeding层的大小
    print("Finish!")
    print("Build data-iter...")
    train_iter = get_iter(source_train,train_labels,vocab)
    test_iter = get_iter(source_test,test_labels,vocab)
    print("Finish!")
    return train_iter,test_iter,vocab