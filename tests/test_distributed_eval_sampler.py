import pytest

pytest.importorskip("torch")

from smoke.samplers import DistributedEvalSampler


def test_eval_shards_cover_each_sample_exactly_once():
    dataset = list(range(11))
    shards = [
        list(DistributedEvalSampler(dataset, num_replicas=2, rank=rank))
        for rank in range(2)
    ]
    combined = shards[0] + shards[1]
    assert sorted(combined) == list(range(len(dataset)))
    assert len(combined) == len(set(combined))
    assert [len(shard) for shard in shards] == [6, 5]
