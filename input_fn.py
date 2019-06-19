# 1. read data from vw
# 2. make dictionary structure: counts, df; cache it
# 3. create batcher
# 4,


import torch
from torch.utils.data import Dataset


class WNTMDataSet(Dataset):
    def __init__(self,
                 documents,
                 context_size,
                 dictionary,
                 dtype,
                 device):
        self.documents = documents
        self.pad_inx = dictionary['<UNK>']
        self.context_size = context_size
        self.dtype = dtype
        self.device = device

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, i):
        context = self.documents[i]
        if len(context) <= self.context_size:
            context = context + [self.pad_inx] * \
                      (self.context_size - len(context))
        else:
            context = [c for c in context if c != self.pad_inx]
            if len(context) <= self.context_size:
                context = context + [self.pad_inx] * \
                          (self.context_size - len(context))
            else:
                context = context[:self.context_size]
            # raise Exception('oversized_context!')

        real_context_len = len([w for w in context if w != self.pad_inx])
        real_context_len = torch.tensor(real_context_len,
                                        dtype=self.dtype,
                                        device=self.device)
        n_dw = torch.ones(self.context_size)
        context = torch.tensor(context, dtype=self.dtype, device=self.device)

        return n_dw, i, context, real_context_len