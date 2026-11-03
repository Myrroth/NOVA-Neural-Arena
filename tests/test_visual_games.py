"""Physics, genuine task outcomes and exact environment restoration, no learner."""
import copy
import json
import math

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from arena.universal.envs import FOOD, GAMES, circle, canvas, make_game


@pytest.mark.parametrize("name", GAMES)
def test_contract_and_seeded_random_stream(name):
    with make_game(name, max_steps=80) as a, make_game(name, max_steps=80) as b:
        check_env(a, skip_render_check=True)
        x, _ = a.reset(seed=312)
        y, _ = b.reset(seed=312)
        assert x.dtype == np.uint8 and x.shape == (32, 32, 3)
        assert np.array_equal(x, y)
        rng = np.random.default_rng(31)
        constant_mask = a.action_mask().copy()
        for _ in range(80):
            action = int(rng.choice(np.flatnonzero(constant_mask)))
            first, second = a.step(action), b.step(action)
            assert np.array_equal(first[0], second[0])
            assert first[1:4] == second[1:4]
            assert np.array_equal(a.action_mask(), constant_mask)
            assert all(np.isfinite(value) for key, value in first[4].items()
                       if isinstance(value, (int, float)))
            if first[2] or first[3]:
                with pytest.raises(RuntimeError):
                    a.step(action)
                break


@pytest.mark.parametrize("name", GAMES)
def test_restore_replays_future_rng_physics_and_metrics(name):
    with make_game(name, max_steps=100) as env:
        env.reset(seed=501)
        controls = list(np.flatnonzero(env.action_mask()))
        if name != "snake":  # A valid varied spawn may face a wall on its first move.
            env.step(int(controls[0]))
        saved = json.loads(json.dumps(env.snapshot()))
        actions = [int(controls[i % len(controls)]) for i in range(25)]
        expected = []
        for action in actions:
            result = env.step(action)
            expected.append((result, copy.deepcopy(env.snapshot())))
            if result[2] or result[3]:
                break
        env.restore(saved)
        for action, (result, state) in zip(actions, expected):
            actual = env.step(action)
            assert np.array_equal(actual[0], result[0])
            assert actual[1:4] == result[1:4]
            assert env.snapshot() == state


@pytest.mark.parametrize("name", GAMES)
def test_snapshot_cross_game_and_config_rejected(name):
    env = make_game(name, max_steps=50)
    env.reset(seed=11)
    state = env.snapshot()
    wrong = make_game(name, max_steps=51)
    wrong.reset(seed=11)
    with pytest.raises(ValueError, match="mismatch"):
        wrong.restore(state)
    state["environment_id"] = "unrelated-v999"
    with pytest.raises(ValueError, match="mismatch"):
        env.restore(state)


def test_snake_wall_body_and_tail_rules():
    env = make_game("snake")
    env.reset(seed=1)
    s = env.snapshot()
    s["state"].update(body=[[0, 3], [1, 3], [2, 3]], direction=3, food=[7, 7])
    env.restore(s)
    _, reward, done, _, info = env.step(3)
    assert done and reward == -1 and info["wall_collision"] == 1
    env.reset(seed=1)
    s = env.snapshot()
    s["state"].update(body=[[3, 3], [3, 4], [4, 4], [4, 3]], direction=1, food=[7, 7])
    env.restore(s)
    assert not env.step(4)[2]  # entering a tail cell vacated by this move is legal.


def test_pong_return_is_physical_not_free_points():
    env = make_game("pong")
    env.reset(seed=11)
    state = env.snapshot()
    state["state"].update(ball=[3.1, 16.], velocity=[-.75, .35], left=16.)
    env.restore(state)
    _, reward, done, _, info = env.step(0)
    assert reward == .15 and not done and info["hits"] == info["incoming"] == 1
    assert env.state["velocity"][0] > 0 and info["points"] == 0


def test_agar_size_determines_eating_not_color_or_label():
    env = make_game("agar")
    env.reset(seed=11)
    state = env.snapshot()
    state["state"].update(player=[16., 16., 10.], pellets=[],
                          bots=[[16., 16., 4., 0.]])
    env.restore(state)
    _, reward, done, _, info = env.step(0)
    assert not done and reward > 0 and info["prey_eaten"] == 1
    env.reset(seed=11)
    state = env.snapshot()
    state["state"].update(player=[16., 16., 10.], pellets=[],
                          bots=[[16., 16., 25., 0.]])
    env.restore(state)
    _, reward, done, _, info = env.step(0)
    assert done and reward == -1 and info["eaten"] == 1


def test_flap_changes_vertical_motion_and_pipe_collision_is_real():
    a, b = make_game("flappy"), make_game("flappy")
    a.reset(seed=31); b.reset(seed=31)
    a.step(9); b.step(0)
    assert a.state["y"] < b.state["y"] and a.state["vy"] < 0
    s = b.snapshot()
    s["state"].update(y=3., vy=0., pipes=[[8., 20., False, 10., False],
                                        [26., 16., False, 10., False]])
    b.restore(s)
    assert b.step(0)[2]


def test_racing_throttle_brake_grip_and_offroad():
    env = make_game("racing")
    env.reset(seed=11)
    initial = env.state["speed"]
    env.step(1)
    assert env.state["speed"] > initial
    env.step(2)
    assert env.state["speed"] < initial
    state = env.snapshot()
    state["state"]["x"] = 0.
    env.restore(state)
    _, reward, done, _, info = env.step(0)
    assert done and reward == -1 and info["collision"] == 1


def test_limits_are_truncations_not_wins():
    env = make_game("racing", max_steps=1)
    env.reset(seed=11)
    _, _, done, truncated, info = env.step(0)
    assert not done and truncated and info["terminal_reason"] == "step_limit"


def test_snake_seeded_spawns_vary_without_invalid_initial_bodies_or_safety_masks():
    env = make_game("snake")
    heads, directions, contexts = set(), set(), set()
    for seed in range(30, 70):
        _, info = env.reset(seed=seed)
        s = env.state
        head = tuple(s["body"][0])
        heads.add(head); directions.add(s["direction"]); contexts.add(info["spawn_context"])
        assert len({tuple(cell) for cell in s["body"]}) == 3
        assert all(0 <= v < 8 for cell in s["body"] for v in cell)
        assert all(math.dist(a, b) == 1 for a, b in zip(s["body"], s["body"][1:]))
        assert (info["initial_head_x"], info["initial_head_y"]) == head
        assert info["initial_direction"] == s["direction"]
    assert len(heads) > 12 and len(directions) == 4 and len(contexts) == 3
    state = env.snapshot()
    state["state"].update(body=[[0, 3], [1, 3], [2, 3]], direction=3)
    env.restore(state)
    assert env.action_mask()[3]  # A mechanically bound, fatal action stays available.


def test_snake_snapshot_replays_food_respawn_rng_after_capture():
    env = make_game("snake")
    env.reset(seed=75)
    saved = env.snapshot()
    saved["state"].update(body=[[3, 3], [2, 3], [1, 3]], direction=4, food=[4, 3])
    env.restore(saved)
    first = env.step(4)
    expected = env.snapshot()
    assert first[4]["foods"] == 1 and expected["state"]["food"] not in expected["state"]["body"]
    env.restore(saved)
    actual = env.step(4)
    assert np.array_equal(first[0], actual[0]) and first[1:4] == actual[1:4]
    assert env.snapshot() == expected


def test_pong_agent_controls_change_interception_with_identical_ball_physics():
    hit, miss = make_game("pong"), make_game("pong")
    hit.reset(seed=88); miss.reset(seed=88)
    state = hit.snapshot()
    state["state"].update(ball=[3.1, 19.5], velocity=[-.75, 0.], left=16.)
    hit.restore(state); miss.restore(state)
    a, b = hit.step(2), miss.step(1)
    assert a[4]["hits"] == 1 and a[4]["conceded"] == 0
    assert b[4]["hits"] == 0 and b[4]["conceded"] == 1
    assert a[4]["incoming"] == b[4]["incoming"] == 1


def test_subpixel_food_remains_visible_between_raster_sample_centres():
    frame = canvas()
    circle(frame, 10.5, 10.5, .65, FOOD)
    assert np.any(np.all(frame == FOOD, axis=-1))


@pytest.mark.parametrize("scenario", ("standard", "threat", "prey"))
def test_agar_probe_config_and_all_objectives_are_seeded_and_nonoverlapping(scenario):
    env = make_game("agar", scenario=scenario)
    starts, masses = set(), set()
    for seed in range(100, 112):
        _, info = env.reset(seed=seed)
        s, p = env.state, env.state["player"]
        starts.add(tuple(p[:2])); masses.add(p[2])
        assert info["scenario"] == scenario and len(s["pellets"]) == 18
        assert sum(b[2] > 1.2*p[2] for b in s["bots"]) == 1
        assert sum(p[2] > 1.2*b[2] for b in s["bots"]) == 2
        assert all(math.dist(food, p[:2]) > env._radius(p[2])+.65 for food in s["pellets"])
        actors = [p, *s["bots"]]
        assert all(math.dist(a[:2], b[:2]) > env._radius(a[2])+env._radius(b[2])
                   for i, a in enumerate(actors) for b in actors[i+1:])
        if scenario == "threat":
            assert math.dist(p[:2], s["bots"][2][:2]) == pytest.approx(6.)
        if scenario == "prey":
            assert math.dist(p[:2], s["bots"][0][:2]) == pytest.approx(4.)
    assert len(starts) == len(masses) == 12
    saved = json.loads(json.dumps(env.snapshot()))
    assert saved["config"]["scenario"] == scenario
    same = make_game("agar", scenario=scenario)
    same.reset(seed=444)
    assert np.array_equal(same.restore(saved), env.render())
    other = make_game("agar", scenario="prey" if scenario != "prey" else "threat")
    other.reset(seed=444)
    with pytest.raises(ValueError, match="mismatch"):
        other.restore(saved)


@pytest.mark.parametrize("name", ("snake", "pong", "flappy", "racing"))
def test_agar_probe_scenarios_cannot_silently_change_other_games(name):
    with pytest.raises(ValueError, match="specific to Agar"):
        make_game(name, scenario="threat")


def test_agar_stationary_capture_respawns_prey_away_and_pellets_require_contact():
    env = make_game("agar")
    env.reset(seed=71)
    state = env.snapshot()
    state["state"].update(player=[16., 16., 10.], pellets=[[16., 16.]],
                          bots=[[16., 16., 4., 0.]])
    env.restore(state)
    info = env.step(0)[4]
    assert info["foods"] == info["prey_eaten"] == 1
    assert math.dist(env.state["player"][:2], env.state["bots"][0][:2]) > 10
    assert math.dist(env.state["player"][:2], env.state["pellets"][0]) > 2
    assert env.step(0)[4]["prey_eaten"] == 1  # No reward from repeated overlapping respawns.


def test_agar_escape_is_post_opponent_separation_with_encounter_denominator():
    env = make_game("agar")
    env.reset(seed=72)
    state = env.snapshot()
    state["state"].update(player=[16., 16., 10.], pellets=[],
                          bots=[[8., 16., 25., 0.]], threat_active=[False])
    env.restore(state)
    info = env.step(4)[4]
    assert info["large_near_steps"] == info["large_encounters"] == 1
    assert info["large_distance_change"] == pytest.approx(.95)
    assert info["escape_events"] == 0  # Player-only radial gain is not yet an escape.
    for _ in range(4):
        info = env.step(4)[4]
    assert info["large_encounters"] == info["escape_events"] == 1
    assert math.dist(env.state["player"][:2], env.state["bots"][0][:2]) >= 10


def test_flappy_seeds_vary_heights_centres_and_all_pipes_stay_in_world():
    env = make_game("flappy")
    gaps, centres, strata = set(), set(), set()
    for seed in range(120, 140):
        env.reset(seed=seed)
        for _, centre, _, gap, _ in env.state["pipes"]:
            gaps.add(gap); centres.add(centre); strata.add(env._gap_stratum(gap))
            assert 2 <= centre-gap/2 < centre+gap/2 <= 30
        assert all(abs(a[1]-b[1]) <= 6 for a, b in zip(env.state["pipes"], env.state["pipes"][1:]))
    assert len(gaps) > 20 and len(centres) > 20 and strata == {"narrow", "wide"}


def test_flappy_scores_only_after_full_body_clearance_and_preserves_strata_counts():
    env = make_game("flappy")
    env.reset(seed=71)
    state = env.snapshot()
    state["state"].update(y=16., vy=-.15,
                          pipes=[[4.5, 16., False, 10., True], [22.5, 16., False, 9., False]],
                          gap_encounters_by_stratum={"narrow": 0, "wide": 1})
    env.restore(state)
    assert env.step(0)[4]["pipes_passed"] == 0
    info = env.step(0)[4]
    assert info["pipes_passed"] == 1 and info["gap_scored_by_stratum"] == {"narrow": 0, "wide": 1}
    state = env.snapshot()
    state["state"].update(y=16., vy=-.15,
                          pipes=[[3.5, 16., False, 9., True], [21.5, 16., False, 12., False]],
                          gap_encounters_by_stratum={"narrow": 1, "wide": 1})
    env.restore(state)
    info = env.step(0)[4]
    assert info["pipes_passed"] == 2 and info["gap_stratum"] == "wide"
    assert info["gap_scored_by_stratum"] == {"narrow": 1, "wide": 1}


def test_flappy_pipe_corner_uses_circle_rectangle_contact():
    env = make_game("flappy")
    env.reset(seed=71)
    state = env.snapshot()
    # At x=10 the nearest corner is sqrt(2) away, beyond radius 1.2.
    state["state"].update(y=15., vy=-.15,
                          pipes=[[10.55, 19., False, 10., False], [28.55, 19., False, 10., False]])
    env.restore(state)
    assert not env.step(0)[2]
    # At x=9.6 the same corner is sqrt(1+.6**2), so it is physical contact.
    state["state"]["pipes"][0][0] = 10.15
    env.restore(state)
    assert env.step(0)[2]


def test_flappy_snapshot_replays_procedural_pipe_replacement_rng():
    env = make_game("flappy")
    env.reset(seed=76)
    saved = env.snapshot()
    saved["state"].update(y=16., vy=-.15,
                          pipes=[[-3.1, 16., True, 10., True], [14., 16., False, 10., False],
                                 [32., 17., False, 9., False]])
    env.restore(saved)
    first = env.step(0)
    expected = env.snapshot()
    assert not first[2] and len(expected["state"]["pipes"]) == 3
    assert expected["state"]["pipes"][-1][0] == pytest.approx(49.45)
    assert expected["rng"] != saved["rng"]
    env.restore(saved)
    actual = env.step(0)
    assert np.array_equal(first[0], actual[0]) and first[1:4] == actual[1:4]
    assert env.snapshot() == expected


def test_racing_course_is_seeded_finite_and_fixed_lateral_position_cannot_win():
    env = make_game("racing")
    tracks = set()
    for seed in range(140, 160):
        _, info = env.reset(seed=seed)
        s = env.state
        tracks.add((s["phase"], s["amplitude"], s["wavelength"]))
        assert s["course_span"] == pytest.approx(2*math.pi*s["wavelength"])
        assert 75 < s["course_span"] <= info["track_length"] < 140
        positions = np.linspace(0., s["course_span"], 10001)
        factors = np.sqrt(1+(s["amplitude"]/s["wavelength"]*
                               np.cos(positions/s["wavelength"]+s["phase"]))**2)
        reference_length = np.sum((factors[1:]+factors[:-1])*.5*np.diff(positions))
        assert info["track_length"] == pytest.approx(reference_length, rel=1e-5)
        assert np.ptp([env._center(u) for u in positions[::50]]) > 2*env.road_half_width
        assert info["progress_normalized"] == info["lap_completed"] == 0
    assert len(tracks) == 20


def test_racing_finish_is_a_physical_terminal_and_not_a_step_limit():
    env = make_game("racing", max_steps=20)
    env.reset(seed=74)
    state = env.snapshot()
    s = state["state"]
    s.update(position=s["course_span"]-.1, speed=.3, x=env._center(s["course_span"]))
    env.restore(state)
    _, reward, done, truncated, info = env.step(0)
    assert done and not truncated and reward == 1
    assert info["terminal_reason"] == "lap_completed" and info["lap_completed"] == 1
    assert info["progress_normalized"] == 1 and info["progress_distance"] == info["track_length"]
    assert info["distance"] == pytest.approx(.1)


def test_racing_stationary_steering_braking_and_speed_by_road_geometry():
    env = make_game("racing")
    env.reset(seed=73)
    state = env.snapshot()
    state["state"].update(speed=0.)
    env.restore(state)
    x = env.state["x"]
    assert not env.step(3)[2]
    assert env.state["x"] == x and env.state["position"] == env.state["distance"] == 0
    assert env.step(2)[4]["effective_brakes"] == 0
    env.step(1)
    info = env.step(2)[4]
    assert info["effective_brakes"] == 1 and info["braking_speed_reduction"] > 0
    for angle, zone in ((math.pi/2, "curved"), (0., "straight")):
        state = env.snapshot()
        s = state["state"]
        position = ((angle-s["phase"]) % (2*math.pi))*s["wavelength"]
        s.update(position=position, x=env._center(position), speed=.5)
        before_sum, before_steps = s[f"{zone}_speed_sum"], s[f"{zone}_steps"]
        env.restore(state)
        info = env.step(2)[4]
        assert not info["collision"]
        assert info[f"{zone}_steps"] == before_steps+1
        assert info[f"{zone}_speed_sum"] == pytest.approx(before_sum+.38)


@pytest.mark.parametrize("name", GAMES)
def test_incomplete_or_nonfinite_snapshot_rejected_atomically(name):
    env = make_game(name)
    env.reset(seed=64)
    original = env.snapshot()
    damaged = copy.deepcopy(original)
    del damaged["state"][next(iter(damaged["state"]))]
    with pytest.raises(ValueError, match="physical snapshot"):
        env.restore(damaged)
    assert env.snapshot() == original
    damaged = copy.deepcopy(original)
    damaged["state"]["unexpected"] = float("nan")
    with pytest.raises(ValueError, match="physical snapshot"):
        env.restore(damaged)
    assert env.snapshot() == original
    damaged = copy.deepcopy(original)
    damaged["finished"] = "false"
    with pytest.raises(ValueError, match="completion"):
        env.restore(damaged)
    assert env.snapshot() == original
