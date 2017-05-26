import numpy as np
from math import floor, ceil

_PAD_ID = 0
_EOS_ID = 1

class GetBatch(object):
    """GetBatch
    Args:
        data:   A tuple or list, which contains parallel data
    """
    def __init__(self, data, batch_size, big_batch=10):
        self.data_inp       = [item + [_EOS_ID] for item in data['input']]
        self.data_spc       = data['speaker_code']
        self.data_out       = data['output']
        self.samples        = len(self.data_inp)
        self.batch_size     = batch_size
        self.big_batch      = big_batch
        self.big_batch_step = 0
        if batch_size == 'all':
            self.batch_size = self.samples
            self.big_batch = 1
        if self.samples < self.big_batch * self.batch_size:
            self.max_batch_step = ceil(self.samples / (self.big_batch * self.batch_size))
        else:
            self.max_batch_step = floor(self.samples / (self.big_batch * self.batch_size)) - 1
        self.run_through    = 0
        self.perm           = None
        self.perm_index     = np.arange(min(self.big_batch * self.batch_size, self.samples))
        self.batch          = None

    def shuffle(self, real_shuffle=True):
        if real_shuffle:
            self.perm           = np.random.permutation(self.samples)
        else:
            self.perm           = np.arange(self.samples)
        self.run_through    += 1
        self.big_batch_step = 0
        self.local_step     = 0

    def get_batch(self):
        if self.big_batch_step > self.max_batch_step:
            print("Have run through data.")
            return None

        if self.local_step == 0:
            self.prepare_batch_index = list(self.perm[self.perm_index + self.big_batch_step * (self.big_batch * self.batch_size)])
            self.prepare_inp = [self.data_inp[idx] for idx in self.prepare_batch_index]
            self.prepare_spc = [self.data_spc[idx] for idx in self.prepare_batch_index]
            self.prepare_out = [self.data_out[idx] for idx in self.prepare_batch_index]
            ### sort index
            tmp = [(item, idx) for idx, item in enumerate(self.prepare_out)]
            sorted_tmp = sorted(tmp, key=lambda x: x[0].shape[0], reverse=True)
            sorted_idx = [item[-1] for item in sorted_tmp]
            ### sorted data
            self.prepare_inp = [self.prepare_inp[idx] for idx in sorted_idx]
            self.prepare_spc = [self.prepare_spc[idx] for idx in sorted_idx]
            self.prepare_out = [self.prepare_out[idx] for idx in sorted_idx]
            ### build buckets
            self.prepare_inp_bucket = []
            self.prepare_spc_bucket = []
            self.prepare_out_bucket = []
            for idx in range(self.big_batch):
                self.prepare_inp_bucket.append(self.prepare_inp[(idx*self.batch_size) : ((idx+1)*self.batch_size)])
                self.prepare_spc_bucket.append(self.prepare_spc[(idx*self.batch_size) : ((idx+1)*self.batch_size)])
                self.prepare_out_bucket.append(self.prepare_out[(idx*self.batch_size) : ((idx+1)*self.batch_size)])
            ### padding
            def padding(mini_batch):
                if mini_batch == []:
                    return None
                sorted_by_len = sorted(mini_batch, key=lambda x: len(x), reverse=True)
                max_len = len(sorted_by_len[0])
                tmp = []
                tmp_mask = []
                for item in mini_batch:
                    pad = [_PAD_ID] * (max_len - len(item))
                    padded = item + pad
                    tmp.append(padded)
                    tmp_mask.append(len(item))
                return np.asarray(tmp, dtype=np.int32), np.asarray(tmp_mask, dtype=np.int32)

            def arr_padding(mini_batch):
                if mini_batch == []:
                    return None, None
                sorted_by_len = sorted(mini_batch, key=lambda x: x.shape[0], reverse=True)
                max_len = len(sorted_by_len[0])
                tmp = []
                tmp_mask = []
                for item in mini_batch:
                    pad = np.zeros(shape=((max_len-item.shape[0]), item.shape[1]))
                    padded = np.concatenate([item, pad], axis=0)
                    mask = np.ones_like(item)
                    mask = np.concatenate([mask, pad], axis=0)
                    tmp.append(padded)
                    tmp_mask.append(mask)
                return np.asarray(tmp, dtype=np.float32), np.asarray(tmp_mask, dtype=np.float32)

            inp_lst = []
            spc_lst = []
            out_lst = []
            inp_mask_lst = []
            out_mask_lst = []
            for idx in range(self.big_batch):
                inp, inp_mask = padding(self.prepare_inp_bucket[idx])
                inp_lst.append(inp)
                inp_mask_lst.append(inp_mask)
                out, out_mask = arr_padding(self.prepare_out_bucket[idx])
                out_lst.append(out)
                out_mask_lst.append(out_mask)
                spc_lst.append(np.asarray(self.prepare_spc_bucket[idx], dtype=int))

            self.prepare_inp_bucket = inp_lst
            self.prepare_inp_mask_bucket = inp_mask_lst
            self.prepare_out_bucket = out_lst
            self.prepare_out_mask_bucket = out_mask_lst
            self.prepare_spc_bucket = spc_lst

            self.big_batch_step += 1


        this_batch = tuple([self.prepare_inp_bucket[self.local_step], self.prepare_spc_bucket[self.local_step], self.prepare_out_bucket[self.local_step], self.prepare_inp_mask_bucket[self.local_step], self.prepare_out_mask_bucket[self.local_step]])

        self.local_step += 1
        self.local_step %= self.big_batch
        return this_batch

