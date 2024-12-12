import matplotlib.pyplot as plt
import torch
from torch import nn
import torchkeras
# import torchmetrics
import d2l.torch as d2l
import data
import model

def _get_batch_loss_bert(net, loss, vocab_size, tokens_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y):
    # 前向传播
    _, mlm_Y_hat = net(tokens_X, valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # 计算遮蔽语言模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    return mlm_l

def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['pre'])
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y in train_iter:
            print(tokens_X.shape)
            tokens_X = tokens_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y = mlm_Y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y)
            mlm_l.backward()
            trainer.step()
            metric.add(mlm_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1, (metric[0] / metric[2]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break
    animator.show()
    plt.figure(animator.fig)
    plt.savefig("pretrain.png")
    print(f'MLM loss {metric[0] / metric[2]:.3f}')
    print(f'{metric[1] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')

if __name__ == '__main__':
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss()
    batch_size, max_len = 512, 60
    print(1)
    train_iter, vocab = data.load_data_wiki(batch_size, max_len)
    print(2)
    print(len(vocab))
    h_num =128
    net = model.BERTModel(len(vocab), num_hiddens=h_num , norm_shape=h_num ,
                        ffn_num_input=h_num , ffn_num_hiddens=256, num_heads=4,
                        num_layers=2, dropout=0.1, key_size=h_num ,query_size=h_num ,
                        value_size=h_num , hid_in_features=h_num , mlm_in_features=h_num ,max_len=max_len
                        )
    # num_hiddens 为嵌入维度
    print(3)
    print(len(vocab))
    train_bert(train_iter, net, loss, len(vocab), devices, 100)
    torch.save(net.state_dict(), 'pretrain-1.pt')

