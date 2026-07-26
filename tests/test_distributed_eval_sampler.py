import pytest

pytest.importorskip("torch")

from smoke.samplers import DistributedEvalSampler


def shards_for(dataset, num_replicas):
    return [
        list(DistributedEvalSampler(dataset, num_replicas=num_replicas, rank=rank))
        for rank in range(num_replicas)
    ]


def test_eval_shards_cover_every_sample():
    dataset = list(range(11))
    shards = shards_for(dataset, 2)
    assert set(shards[0] + shards[1]) == set(range(len(dataset)))


def test_every_rank_gets_the_same_number_of_batches():
    """Unequal shards deadlock the collectives inside the full eval forward.

    An odd-length split used to give rank 0 six samples and rank 1 five; rank 1
    then reached the end of its loader while rank 0 was still inside the ITC
    all_gather of its last batch.
    """
    shards = shards_for(list(range(11)), 2)
    assert [len(shard) for shard in shards] == [6, 6]

    shards = shards_for(list(range(2963)), 2)
    assert len(shards[0]) == len(shards[1])


def test_padding_never_exceeds_one_sample_per_extra_rank():
    """The duplicated tail is the only price paid for equal length."""
    for count, replicas in ((11, 2), (2963, 2), (10, 4), (7, 3)):
        shards = shards_for(list(range(count)), replicas)
        combined = [index for shard in shards for index in shard]
        assert len(combined) - len(set(combined)) <= replicas - 1


def test_single_rank_is_exactly_the_dataset():
    assert list(DistributedEvalSampler(list(range(5)), num_replicas=1, rank=0)) == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_empty_dataset_yields_nothing():
    sampler = DistributedEvalSampler([], num_replicas=2, rank=1)
    assert list(sampler) == []
    assert len(sampler) == 0
