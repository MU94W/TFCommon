import threading, tensorflow as tf, random, collections
from six.moves import xrange

class BaseFeeder(threading.Thread):
    """
    Thread data feeder.
    You should never use this class directly,
    instead, write a child class that overwrites
        read_by_key,
        pre_process_batch,
        split_strategy (if split_nums is not None) methods.
    """
    def __init__(self, session, placeholders, meta, batch_size=32, split_nums=None, is_validation=False):
        """

        :param session:
        :param placeholders:
        :param meta:
        :param batch_size:
        :param split_nums:
        :param is_validation:
        """
        super(BaseFeeder, self).__init__()
        queue = tf.FIFOQueue(capacity=int(batch_size/4), dtypes=[item.dtype for item in placeholders])
        self.queue = queue  # for buf inspect
        self.enqueue_op = queue.enqueue(placeholders)
        self.fed_holders = [None] * len(placeholders)   # None placeholder for dequeue
        self.fed_holders = queue.dequeue()
        for idx in range(len(placeholders)):
            self.fed_holders[idx].set_shape(placeholders[idx].shape)
        self._placeholders = placeholders
        self.sess = session
        self.meta = meta
        key_lst = meta.get('key_lst')
        assert isinstance(key_lst, list) or isinstance(key_lst, tuple)
        self.key_lst = key_lst
        self.batch_size = batch_size
        self.split_bool = False if split_nums is None else True
        self.split_nums = split_nums
        assert isinstance(is_validation, bool)
        self.is_validation = is_validation
        self._total_samples = len(key_lst)
        self._iter = 0
        self._record_index = 0
        self._loss = 0.

    def read_by_key(self, key):
        pass

    def pre_process_batch(self, batch):
        pass

    def split_strategy(self, many_records) -> collections.Iterator:
        pass

    def collect_loss(self, loss):
        self._loss += loss

    def mean_loss(self):
        m_l = self._loss * self.batch_size / self._total_samples
        self._loss = 0.
        return m_l

    def prepare_batch(self):
        if not self.split_bool:
            self.feed_single_batch(self.fetch_one_batch())
        else:
            many_records = [self.fetch_one_record() for _ in xrange(self.batch_size * self.split_nums)]
            for batch in self.split_strategy(many_records):
                self.feed_single_batch(self.pre_process_batch(batch))

    def prepare_validation(self):
        if not self.split_bool:
            while self._record_index <= (self._total_samples - self.batch_size):
                self.feed_single_batch(self.fetch_one_batch())
            remain_batch = []
            while self._record_index != 0:
                remain_batch.append(self.fetch_one_record())
            self.feed_single_batch(self.pre_process_batch(remain_batch))
        else:
            many_records = [self.fetch_one_record() for _ in xrange(self._total_samples)]
            for batch in self.split_strategy(many_records):
                self.feed_single_batch(self.pre_process_batch(batch))

    def fetch_one_batch(self):
        records = [self.fetch_one_record() for _ in xrange(self.batch_size)]
        try:
            pre_processed = self.pre_process_batch(records)
        except Exception as e:
            print('[E] pre_processed failed')
            pre_processed = []
        return pre_processed

    def feed_single_batch(self, single_batch):
        """
        Will be blocked, if the queue is full.
        The item order in batch must match the placeholder list.
        :param single_batch:
        :return:
        """
        try:
            self.sess.run(self.enqueue_op, feed_dict=dict(zip(self._placeholders, single_batch)))
        except Exception as e:
            print('[E] feed_single_batch failed')

    def fetch_one_record(self):
        if self._record_index >= self._total_samples:
            random.shuffle(self.key_lst)
            self._record_index = 0
        self._record_index += 1
        try:
            record = self.read_by_key(self.key_lst[self._record_index-1])
        except Exception as e:
            print('[E] read_by_key failed')
            record = []
        return record

    def run(self):
        try:
            while True:
                if not self.is_validation:
                    self.prepare_batch()
                else:
                    self.prepare_validation()
        except Exception as e:
            print('Data feeder thread failed.')
        finally:
            print('Data feeder done.')
