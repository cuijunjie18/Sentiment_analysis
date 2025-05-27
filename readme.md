## NLP应用——文本情感分析

### 日志

5.27 2025 数据预处理完成

### 意外之喜——tqdm的性能分析

- 问题发现
其实这个的性能跟循环次数有关，每次调用loop.set_description的时间几乎都是定的，所以对于循环次数多，循环主体
用时少的，这个就会成为影响性能的罪魁祸首!!!

- 隐藏原因
之前 没发现是因为在VGG中对图像识别处理数据集的时候，有batch_size,导致循环次数少，且循环主体又比较耗时，故几乎
不影响.

性能差的代码
```py
for x in loop:
    body
    loop.set_description(f"read {label}")
```

优化方法1
```py
loop = tqdm(xxxx,desc = f"read {label}") # 在循环前指定
for x in loop:
    body
```

优化方法2
```py
for x in loop:
    body
    loop.set_description(f"read {label}",refresh = False) # 设置仅绘制一次，关闭动态刷新
```