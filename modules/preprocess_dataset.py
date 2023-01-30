import numpy as np
import torch
import torch.utils.data
import math

PAD = 1e20

class EventData(torch.utils.data.Dataset):

    def __init__(self, data, data_ratio, num_types, time_scale):
        """
        data: a list of lists of dictionaries
        data_ratio: the ratio of sequences
        num_types: the number of types of the observed events
        time_scale: a number to scale the time
        """

        self.length = int(math.ceil(len(data) * data_ratio))
        self.times_lst = []  # [num_sequences * [numypes * []]]
        self.interval_end = []
        self.max_length = 0
        for seq in data:
            self.times_lst.append([[] for _ in range(num_types)])
            t_max = 0
            if len(seq) > self.max_length:
                self.max_length = len(seq)
            for elem in seq:
                event_time = elem["time_since_start"] * time_scale
                event_type = elem["type_event"]
                self.times_lst[-1][event_type].append(event_time)
                if event_time > t_max:
                    t_max = event_time
            self.interval_end.append(t_max)
            if len(self.times_lst) == self.length:
                break
        self.pad = PAD


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ return a sequence with its index """
        return self.times_lst[idx], idx


def collate_fn(insts):
    # insts: a list of tuples [batch_size * (data, index)], data is a list of lists [num_types * [seq_len]]
    times_for_each_type_id = list(zip(*insts))  # a list [data, index], data is a tuple (batch_size * [num_types * [seq_len]])
    times_for_each_type = list(zip(*times_for_each_type_id[0]))  # a list of tuples (num_types * (batch_size * [seq_len]))
    ids = times_for_each_type_id[1]
    ret = []
    for insts_type in times_for_each_type:
        max_len = max(len(inst) for inst in insts_type)

        batch_seq = np.array(
            [inst + [PAD] * (max_len - len(inst)) for inst in insts_type]
        )

        ret.append(torch.tensor(batch_seq, dtype=torch.float32))
    return ret, ids


def get_dataloader(data, batch_size, data_ratio, num_types, time_scale, shuffle=True):

    ds = EventData(data, data_ratio, num_types, time_scale)
    dl = torch.utils.data.DataLoader(
        ds, num_workers=2, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle
    )
    return dl
