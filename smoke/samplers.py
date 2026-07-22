from torch.utils.data import Sampler


class DistributedEvalSampler(Sampler):
    """Shard evaluation without padding or duplicating any sample."""

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        n = len(self.dataset)
        return max(0, (n - self.rank + self.num_replicas - 1) // self.num_replicas)
