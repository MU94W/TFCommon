import threading, tensorflow as tf

def build_feeder(prepare_batch_func, coordinator, placeholders, batch_size=32):
    """
    Helper function for building data feeder.
    :param prepare_batch_func:
    :param coordinator: tensorflow.train.Coordinator instance
    :param placeholders: list of tensorflow.placeholder instance
    :param batch_size:
    :return:
    """
    base_feeder = BaseFeeder(coordinator, placeholders, batch_size)
    base_feeder.prepare_batch = prepare_batch_func
    base_feeder.start()
    return base_feeder

class BaseFeeder(threading.Thread):
    def __init__(self, coordinator, placeholders, session, batch_size=32):
        """

        :param coordinator:
        :param placeholders:
        :param batch_size:
        """
        super(BaseFeeder, self).__init__()
        self.coord = coordinator
        self.queue = tf.FIFOQueue(capacity=int(batch_size/4), dtypes=[item.dtype for item in placeholders])
        self.enqueue_op = self.queue.enqueue(placeholders)
        fed_holders = [None] * len(placeholders)
        deq = self.queue.dequeue()
        for idx in range(len(fed_holders)):
            fed_holders[idx] = deq[idx]
        self.fed_holders = [item.set_shape(ph.get_shape()) for item, ph in zip(fed_holders, placeholders)]
        self.placeholders = placeholders
        self.sess = session

    def prepare_batch(self):
        """
        You should call self.feed_single_batch in the implementation.
        :return:
        """
        pass

    def feed_single_batch(self, single_batch):
        """
        Will be blocked, if the queue is full.
        :param single_batch:
        :return:
        """
        self.sess.run(self.enqueue_op, feed_dict=dict(zip(self.placeholders, single_batch)))

    def run(self):
        try:
            while not self.coord.should_stop():
                self.prepare_batch()
        except Exception as e:
            # Report exceptions to the coordinator.
            print('Data feeder thread failed.')
            self.coord.request_stop(e)
        finally:
            # Terminate as usual. It is safe to call `coord.request_stop()` twice.
            print('Data feeder done.')
            self.coord.request_stop()

