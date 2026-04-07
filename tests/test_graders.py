"""
tests/test_graders.py — Pytest suite for per-task programmatic graders (H2)
============================================================================
Verifies that all three graders:
  - Return float in [0.0, 1.0]
  - Are deterministic (same input → same output)
  - Score perfect agents near 1.0 and failing agents near 0.0
  - Handle edge cases (empty trajectory, all-crash trajectory)
"""
import pytest
from graders import EasyGrader, MediumGrader, HardGrader, get_grader


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def make_step(
    reward_final=0.8,
    infra=0,
    crashed=False,
    p99=50.0,
    event_type="normal",
    risk_score=20.0,
    decision=0,
    crypto=0,
) -> dict:
    return {
        "reward_final":             reward_final,
        "action_infra_routing":     infra,
        "crashed":                  crashed,
        "obs_rolling_p99":          p99,
        "event_type":               event_type,
        "obs_risk_score":           risk_score,
        "action_risk_decision":     decision,
        "action_crypto_verify":     crypto,
    }


# ---------------------------------------------------------------------------
# get_grader factory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task,cls", [
    ("easy",   EasyGrader),
    ("medium", MediumGrader),
    ("hard",   HardGrader),
])
def test_get_grader_returns_correct_type(task, cls):
    assert isinstance(get_grader(task), cls)


def test_get_grader_invalid_task():
    with pytest.raises(ValueError, match="Unknown task"):
        get_grader("legendary")


# ---------------------------------------------------------------------------
# EasyGrader
# ---------------------------------------------------------------------------

class TestEasyGrader:
    def setup_method(self):
        self.grader = EasyGrader()

    def test_empty_trajectory_scores_zero(self):
        assert self.grader.grade([]) == 0.0

    def test_perfect_agent_scores_one(self):
        traj = [make_step(reward_final=0.8, infra=0)] * 20
        score = self.grader.grade(traj)
        assert score == pytest.approx(1.0)

    def test_throttle_heavy_scores_half(self):
        # reward=0.8 but unnecessary throttle → 0.5 credit each
        traj = [make_step(reward_final=0.8, infra=1)] * 10
        score = self.grader.grade(traj)
        assert score == pytest.approx(0.5)

    def test_low_reward_scores_zero(self):
        traj = [make_step(reward_final=0.3, infra=0)] * 10
        score = self.grader.grade(traj)
        assert score == pytest.approx(0.0)

    def test_score_in_range(self):
        traj = [make_step(reward_final=0.8, infra=0)] * 5 + \
               [make_step(reward_final=0.0, infra=2)] * 5
        score = self.grader.grade(traj)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        traj = [make_step(reward_final=0.8, infra=0)] * 10
        assert self.grader.grade(traj) == self.grader.grade(traj)


# ---------------------------------------------------------------------------
# MediumGrader
# ---------------------------------------------------------------------------

class TestMediumGrader:
    def setup_method(self):
        self.grader = MediumGrader()

    def test_empty_trajectory_scores_zero(self):
        assert self.grader.grade([]) == 0.0

    def test_all_crashes_scores_zero(self):
        traj = [make_step(crashed=True, p99=900.0)] * 10
        assert self.grader.grade(traj) == pytest.approx(0.0)

    def test_clean_steps_score_near_one(self):
        traj = [make_step(crashed=False, p99=400.0)] * 20
        score = self.grader.grade(traj)
        assert score >= 0.95

    def test_flash_sale_throttle_bonus(self):
        # Two identical no-crash episodes; one has throttle bonus during flash_sale
        base = [make_step(crashed=False, p99=400.0, event_type="normal")] * 10
        with_bonus = base + [make_step(crashed=False, p99=400.0, event_type="flash_sale", infra=1)] * 2
        no_bonus   = base + [make_step(crashed=False, p99=400.0, event_type="flash_sale", infra=0)] * 2
        assert self.grader.grade(with_bonus) >= self.grader.grade(no_bonus)

    def test_score_in_range(self):
        traj = [make_step(crashed=i % 3 == 0, p99=200.0) for i in range(30)]
        score = self.grader.grade(traj)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        traj = [make_step(crashed=False, p99=250.0)] * 10
        assert self.grader.grade(traj) == self.grader.grade(traj)


# ---------------------------------------------------------------------------
# HardGrader
# ---------------------------------------------------------------------------

class TestHardGrader:
    def setup_method(self):
        self.grader = HardGrader()

    def test_empty_trajectory_scores_zero(self):
        assert self.grader.grade([]) == 0.0

    def test_perfect_reject_no_crash_scores_one(self):
        traj = [make_step(risk_score=95.0, decision=1, crypto=0, crashed=False)] * 10
        score = self.grader.grade(traj)
        assert score == pytest.approx(1.0)

    def test_all_approve_scores_zero(self):
        traj = [make_step(risk_score=95.0, decision=0, crashed=False)] * 10
        score = self.grader.grade(traj)
        assert score == pytest.approx(0.0)

    def test_challenge_scores_less_than_reject(self):
        reject_traj    = [make_step(risk_score=95.0, decision=1, crashed=False)] * 10
        challenge_traj = [make_step(risk_score=95.0, decision=2, crashed=False)] * 10
        assert self.grader.grade(reject_traj) > self.grader.grade(challenge_traj)

    def test_crash_penalises_score(self):
        no_crash   = [make_step(risk_score=95.0, decision=1, crashed=False)] * 10
        with_crash = [make_step(risk_score=95.0, decision=1, crashed=True)]  * 10
        assert self.grader.grade(no_crash) > self.grader.grade(with_crash)

    def test_fullverify_bonus_applied(self):
        # Use a mix of correct (reject) and miss (approve) steps so the
        # score doesn't hit 1.0 for both — only then the FullVerify bonus
        # makes a visible difference.
        mixed = [make_step(risk_score=95.0, decision=1, crashed=False)] * 5 + \
                [make_step(risk_score=95.0, decision=0, crashed=False)] * 5   # 5 misses
        full_verify = [dict(s, **{"action_crypto_verify": 0}) for s in mixed]
        skip_verify = [dict(s, **{"action_crypto_verify": 1}) for s in mixed]
        assert self.grader.grade(full_verify) >= self.grader.grade(skip_verify)

    def test_score_in_range(self):
        traj = [make_step(risk_score=90.0, decision=i % 2, crashed=i % 4 == 0)
                for i in range(20)]
        score = self.grader.grade(traj)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        traj = [make_step(risk_score=92.0, decision=1, crashed=False)] * 10
        assert self.grader.grade(traj) == self.grader.grade(traj)
