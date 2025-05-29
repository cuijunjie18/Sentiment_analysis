import matplotlib.pyplot as plt
import numpy as np
import joblib

# 加载训练数据
num_epochs = 20
plt_rnn = joblib.load("plots/rnn/exp2-20epoch.joblib")
plt_cnn = joblib.load("plots/cnn/exp2-20epoch.joblib")
l_rnn,train_rnn,test_rnn = plt_rnn
l_cnn,train_cnn,test_cnn = plt_cnn

# LOSS对比
plt.xticks(np.arange(0, num_epochs + 1, 1))
plt.ylim(0, 0.8)  # 强制设置y轴范围
plt.yticks(np.arange(0, 1.1, 0.2))
plt.grid(True, which='major', linestyle='-', alpha=0.7)
plt.title("Loss between rnn and cnn")
plt.plot(np.arange(num_epochs),l_rnn,label = 'RNN')
plt.scatter(np.arange(num_epochs),l_rnn,s = 20)
plt.plot(np.arange(num_epochs),l_cnn,label = 'CNN')
plt.scatter(np.arange(num_epochs),l_cnn,s = 20)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.legend()
plt.savefig("1.png")

# train_acc互相对比
plt.clf()
plt.xticks(np.arange(0, num_epochs + 1, 1))
plt.ylim(0, 1.1)  # 强制设置y轴范围
plt.yticks(np.arange(0, 1.1, 0.2))
plt.grid(True, which='major', linestyle='-', alpha=0.7)
plt.title("Train-acc between rnn and cnn")
plt.plot(np.arange(num_epochs),train_rnn,label = 'RNN')
plt.scatter(np.arange(num_epochs),train_rnn,s = 20)
plt.plot(np.arange(num_epochs),train_cnn,label = 'CNN')
plt.scatter(np.arange(num_epochs),train_cnn,s = 20)
plt.xlabel("Epoch")
plt.ylabel("Train acc")
plt.legend()
plt.savefig("2.png")

# test_acc互相对比
plt.clf()
plt.xticks(np.arange(0, num_epochs + 1, 1))
plt.ylim(0, 1.1)  # 强制设置y轴范围
plt.yticks(np.arange(0, 1.1, 0.2))
plt.grid(True, which='major', linestyle='-', alpha=0.7)
plt.title("Test-acc between rnn and cnn")
plt.plot(np.arange(num_epochs),test_rnn,label = 'RNN')
plt.scatter(np.arange(num_epochs),test_rnn,s = 20)
plt.plot(np.arange(num_epochs),test_cnn,label = 'CNN')
plt.scatter(np.arange(num_epochs),test_cnn,s = 20)
plt.xlabel("Epoch")
plt.ylabel("Test acc")
plt.legend()
plt.savefig("3.png")

# rnn拟合情况对比
plt.clf()
plt.xticks(np.arange(0, num_epochs + 1, 1))
plt.ylim(0, 1.1)  # 强制设置y轴范围
plt.yticks(np.arange(0, 1.1, 0.2))
plt.grid(True, which='major', linestyle='-', alpha=0.7)
plt.title("acc-rnn between train and test")
plt.plot(np.arange(num_epochs),train_rnn,label = 'train')
plt.scatter(np.arange(num_epochs),train_rnn,s = 20)
plt.plot(np.arange(num_epochs),test_rnn,label = 'test')
plt.scatter(np.arange(num_epochs),test_rnn,s = 20)
plt.xlabel("Epoch")
plt.ylabel("acc-rnn")
plt.legend()
plt.savefig("4.png")

# cnn拟合情况对比
plt.clf()
plt.xticks(np.arange(0, num_epochs + 1, 1))
plt.ylim(0, 1.1)  # 强制设置y轴范围
plt.yticks(np.arange(0, 1.1, 0.2))
plt.grid(True, which='major', linestyle='-', alpha=0.7)
plt.title("acc-cnn between train and test")
plt.plot(np.arange(num_epochs),train_cnn,label = 'train')
plt.scatter(np.arange(num_epochs),train_cnn,s = 20)
plt.plot(np.arange(num_epochs),test_cnn,label = 'test')
plt.scatter(np.arange(num_epochs),test_cnn,s = 20)
plt.xlabel("Epoch")
plt.ylabel("acc-cnn")
plt.legend()
plt.savefig("5.png")