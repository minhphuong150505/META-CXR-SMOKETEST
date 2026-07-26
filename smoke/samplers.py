import math

from torch.utils.data import Sampler


class DistributedEvalSampler(Sampler):
    """Shard evaluation, padding every rank to the same number of samples.

    Equal length is a correctness requirement, not tidiness: with
    ``classification_only_eval: false`` the validation forward runs the full
    objective, whose ITC ``all_gather`` and ITM/teacher ``all_reduce`` are
    collectives. A rank that runs out of batches first leaves its peer blocked
    in a collective until the NCCL watchdog fires, so the ranks must agree on
    the batch count.

    Padding repeats at most ``num_replicas - 1`` samples in total (one study out
    of 2,963 on the 2-rank val split); those are counted twice in the loss and
    the confusion matrix. That is the same trade torch's ``DistributedSampler``
    makes, and the trade the pre-smoke setup ran under.
    """

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    @property
    def num_samples(self):
        """Samples this rank yields — identical on every rank by construction."""
        if self.num_replicas <= 0:
            return len(self.dataset)
        return math.ceil(len(self.dataset) / self.num_replicas)

    @property
    def total_size(self):
        return self.num_samples * self.num_replicas

    def __iter__(self):
        count = len(self.dataset)
        indices = list(range(count))
        padding = self.total_size - count
        if padding > 0 and count > 0:
            indices += indices[:padding]
        # Strided assignment, unchanged from the unpadded version: rank r keeps
        # every num_replicas-th index starting at r.
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self):
        return self.num_samples
