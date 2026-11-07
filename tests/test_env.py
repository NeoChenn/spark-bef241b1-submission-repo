import numpy as np

from flange_robot.env import Action, FlangeEnv


def test_env_builds_and_observes():
    env = FlangeEnv(seed=1)
    obs = env.reset(seed=1)
    assert set(obs) == {"robot_xy", "gas_fraction_lel", "bolts", "camera"}
    assert obs["bolts"] == []  # flange is out of sensor range at the start


def test_robot_moves_toward_a_bolt_and_sees_it():
    env = FlangeEnv(seed=2)
    start = np.array(env.robot_xy)
    obs = None
    for _ in range(200):
        obs = env.step(Action("move_to", 0))
    assert np.linalg.norm(env.robot_xy - start) > 1.0
    assert any(b["bolt"] == 0 for b in obs["bolts"])


def test_loosened_bolt_shows_in_the_stripe_reading_and_retorque_fixes_it():
    env = FlangeEnv(seed=3)
    env.loosen_bolt(0, 0.4)
    for _ in range(300):
        obs = env.step(Action("move_to", 0))
    reading = next(b for b in obs["bolts"] if b["bolt"] == 0)
    assert reading["stripe_misalignment_rad"] > 0.2
    obs = env.step(Action("retorque", 0))
    assert obs["retorqued"]
    assert abs(env.true_bolt_angle(0)) < 1e-3


def test_retorque_out_of_reach_does_nothing():
    env = FlangeEnv(seed=4)
    env.loosen_bolt(0, 0.4)
    obs = env.step(Action("retorque", 0))
    assert not obs["retorqued"]
    assert env.true_bolt_angle(0) > 0.3


def test_gas_leak_raises_reading_near_the_flange():
    env = FlangeEnv(seed=5)
    before = env.gas_fraction_lel()
    env.trigger_gas_leak()
    for _ in range(300):
        env.step(Action("move_to", 0))
    assert env.gas_fraction_lel() > before + 0.2


def test_dust_lowers_sensor_confidence():
    env = FlangeEnv(seed=6)
    for _ in range(300):
        env.step(Action("move_to", 0))
    clear = max(b["confidence"] for b in env.observe()["bolts"])
    env.set_dust(1.0)
    dusty = max(b["confidence"] for b in env.observe()["bolts"])
    assert dusty < clear
