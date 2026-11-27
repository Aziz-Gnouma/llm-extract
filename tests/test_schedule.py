from llmx.training.schedule import effective_eval_steps, total_optimizer_steps


def test_total_steps_matches_main_run():
    # 1500 examples, batch 4 x accum 4, 2 epochs -> 94 steps/epoch -> 188 steps
    assert total_optimizer_steps(1500, 4, 4, 2) == 188


def test_long_run_keeps_configured_interval():
    assert effective_eval_steps(188, configured=25) == 25


def test_short_run_is_still_evaluated_several_times():
    total = total_optimizer_steps(100, 4, 4, 2)  # 14 steps < 25
    interval = effective_eval_steps(total, configured=25)
    assert interval == 3
    assert total // interval >= 4


def test_never_zero():
    assert effective_eval_steps(1, configured=25) == 1
