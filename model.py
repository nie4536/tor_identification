import linformer.linformer
import torch
from torch import nn
from d2l import torch as d2l
from linformer.linformer import LinformerLM, Linformer, LinformerSelfAttention
from linformer.reversible import ReversibleSequence, SequentialSequence

class Linformer1(nn.Module):
    def __init__(self, dim, seq_len, depth, k = 256, heads = 8, dim_head = None,
                 one_kv_head = False, share_kv = False, reversible = False, dropout = 0.):
        super().__init__()
        layers = nn.ModuleList([])
        #for _ in range(depth):
        attn = LinformerSelfAttention(dim, seq_len, k = k, heads = heads, dim_head = dim_head, one_kv_head = one_kv_head, share_kv = share_kv, dropout = dropout)
        ff = linformer.linformer.FeedForward(dim, dropout = dropout)

        layers.append(nn.ModuleList([
            linformer.linformer.PreNorm(dim, attn),
            linformer.linformer.PreNorm(dim, ff)
        ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        self.net = execute_type(layers)

    def forward(self, x, valid_lens):    #
        # print("net(x):",self.net(x))
        return self.net(x)

class BERTEncoder(nn.Module):
    #num_layers是层数，vocab_size是词汇量，num_hiddens是嵌入维度，
    #key_size、query_size、value_size分别是键、查询、值大小，分别用来产生Query、Key 和 Value矩阵的三个线性层的权重大小
    #ffn_num_input是FFN的输入，ffn_num_hiddens是FFN的嵌入维度
    #norm_shape=dim
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        # token_embedding是句子的embedding表示，他是按照正常词袋大小来计算的，segment_embedding是2嵌入，即只区分上下句
        self.blks = nn.Sequential()
        for i in range(num_layers):
            # self.blks.add_module(f"{i}", d2l.EncoderBlock(
            #     # num_hiddens是嵌入维度，num_heads是头数
            #     key_size, query_size, value_size, num_hiddens, norm_shape,
            #     ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
            self.blks.add_module(f"{i}", Linformer1(
                # num_hiddens是嵌入维度，num_heads是头数
                dim=num_hiddens, seq_len=vocab_size, depth=num_layers, k=key_size,
                dim_head=None, heads=num_heads, one_kv_head=False,
                share_kv=False, reversible=False, dropout=dropout))

        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs=128, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        # num_pred_positions是一个句子中mask的数量，这里是3
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        # 根据pred_positions(即mask的位置，找到被encoder编码过的mask对应的编码)
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens =128, norm_shape=128, ffn_num_input=128,
                 ffn_num_hiddens=256, num_heads=8, num_layers=2, dropout=0.2,
                 max_len=300, key_size=128, query_size=128, value_size=128,
                 hid_in_features=128, mlm_in_features=128):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)

    def forward(self, tokens, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, valid_lens) # tokens是数字id形式是一维序列
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
            return encoded_X, mlm_Y_hat
        else:
            return encoded_X