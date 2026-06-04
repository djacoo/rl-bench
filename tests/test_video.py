from rl_bench.video import should_record_video


def test_should_record_video_every_100k():
    total = 500_000
    every = 100_000
    recorded = [s for s in range(1, total + 1) if should_record_video(s, total, every)]
    assert recorded == [100_000, 200_000, 300_000, 400_000, 500_000]


def test_should_record_video_final_only_when_not_divisible():
    total = 450_000
    every = 100_000
    recorded = [s for s in range(1, total + 1) if should_record_video(s, total, every)]
    assert recorded == [100_000, 200_000, 300_000, 400_000, 450_000]


def test_should_record_video_disabled():
    assert not should_record_video(100_000, 500_000, None)
    assert not should_record_video(0, 500_000, 100_000)
