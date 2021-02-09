import torch
import pandas as pd
from collections import Counter
from util import get_device

# 'train_data.txt'
class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name, max_len=128):
        self.file_name = file_name
        self.max_len = max_len+1 ## for GT in test set
        #self.max_len = max_len  ## for GT in test set

        self.dataset = self.load_dataset()
        self.dataset = self.dataset.reset_index()

        DEVICE = get_device()
        self.dataset = self.dataset
        self.dataset.set_index("idx", inplace=True)  ## to use index for comparing timestamp
        self.count = len(self.dataset)

        self.uniq_items = [i for i in range(self.count)]
        self.nuniq_items = len(self.uniq_items)

    def str_to_list(self, seq):
        return seq.split(',')

    def str_to_list_max_len(self, seq):
        _seq = seq.split(',')
        len_seq = len(_seq)
        if len_seq < self.max_len:
            ls_zero = [0 for i in range(self.max_len - len_seq)]
            ls_zero.extend(_seq)
            _seq = ls_zero
        else:
            _seq = _seq[-self.max_len:]

        return _seq

    def str_to_list_max_one(self, seq):
        _seq = seq.split(',')
        _seq = _seq[-1:]

        return _seq

    def load_dataset(self):
        names = ['user_id', 'sequence']
        train_df = pd.read_csv(self.file_name, delimiter=':', names=names)
        train_df['idx'] = range(0, len(train_df))
        train_df['sequence'] = train_df['sequence'] \
            .map(self.str_to_list_max_len).apply(lambda x: list(map(int, x)))
        #print(train_df.head())
        return train_df

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # return input_seq and gt
        #seq = self.dataset.loc[index]['sequence'][:-1]
        #gt = self.dataset.loc[index]['sequence'][-1]
        #print('index=', index, self.dataset.loc[index]['user_id'], len(seq), gt)
        return (torch.tensor(self.dataset.loc[index]['user_id']), \
                torch.LongTensor(self.dataset.loc[index]['sequence']))

if __name__ == '__main__' :
    dataset = Dataset('./data/train_data.txt', 3)
    print(dataset.uniq_items)
    print(dataset.count)
    for i in range(10):
        print(i, dataset[i])
