import os
import random
import torch
import torch.utils.data
from d2l import torch as d2l
def _read_bytes(data_dir):
    file_name = 'E:\code\pretraining\my_train_data.txt'
    #file_name = 'E:\code\pretraining\ISCX_data.txt'
    with open(file_name, 'r',errors='ignore') as f:
        lines = f.readlines()
    paragraphs = [[line.strip()[0:60]] for line in lines]
    random.shuffle(paragraphs)
    return paragraphs

def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
    # print(tokens)
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
    random.shuffle(candidate_pred_positions)
    # candidate_pred_positions其实是一个句子也就是tokens里的位置列表0,1,2...
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%的时间：将词替换为“<mask>”词元
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的时间：保持词不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间：用随机词替换该词
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
        # print(mlm_input_tokens)
    return mlm_input_tokens, pred_positions_and_labels # 真token(单词)

def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # tokens是一个字符串列表
    for i, token in enumerate(tokens):
        candidate_pred_positions.append(i)
    # 遮蔽语言模型任务中预测15%的随机词元
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    # print(num_mlm_preds)
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
    # 返回字节对应的 tokens_id(已经被masked的句子)(在词表中的序号), <mask>符号的位置(在句子中的序号), 被maked原单词的id

def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, valid_lens = [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []

    for (token_ids, pred_positions, mlm_pred_label_ids) in examples:
        # print(token_ids)
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
    return (all_token_ids, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels)

class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=1, reserved_tokens=[
            '<pad>', '<mask>'])
        # 获取遮蔽语言模型任务的数据
        examples = []
        for sentence in sentences:
            # print(len((_get_mlm_data_from_tokens(sentence, self.vocab))))
            examples.append((_get_mlm_data_from_tokens(sentence, self.vocab)))
        # 填充输入

        (self.all_token_ids, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels) = _pad_bert_inputs(examples, max_len, self.vocab)
    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)

def load_data_wiki(batch_size, max_len):
    num_workers = d2l.get_dataloader_workers()
    # data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    data_dir = ''
    # paragraphs = _read_wiki(data_dir)
    paragraphs = _read_bytes(data_dir)
    print(4)
    train_set = _WikiTextDataset(paragraphs, max_len)
    print(5)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=0)
    return train_iter, train_set.vocab
