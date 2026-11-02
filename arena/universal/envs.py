"""Prepared local games sharing pixels and joypad controls, never a player policy.

Rules, rendering and task rewards belong here. No environment supplies danger
features, target directions, preferred actions or a winning controller to the
learner. Opponents are part of the simulator, explicitly separate from the agent.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np


ACTION_NAMES = ("noop", "up", "down", "left", "right", "up_left", "up_right",
                "down_left", "down_right", "button")
DIRECTIONS = np.array(((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
                       (-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0)), dtype=float)
DIRECTIONS[5:9] /= math.sqrt(2)
BACKGROUND = (18, 22, 30)
PLAYER = (70, 175, 250)
FOOD = (245, 210, 65)
WALL = (205, 210, 220)
OTHER = (225, 90, 100)
Y_GRID, X_GRID = np.mgrid[0:32, 0:32]


@dataclass(frozen=True)
class VisualConfig:
    max_steps: int = 240
    difficulty: float = 1.0

    def __post_init__(self):
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 2000:
            raise ValueError("max_steps must be 1..2000")
        if not math.isfinite(self.difficulty) or not .25 <= self.difficulty <= 2:
            raise ValueError("difficulty must be .25..2")


def canvas():
    frame = np.empty((32, 32, 3), dtype=np.uint8)
    frame[:] = BACKGROUND
    return frame


def rect(frame, x0, y0, x1, y1, color):
    a, b = max(0, int(math.floor(x0))), min(32, int(math.ceil(x1)))
    c, d = max(0, int(math.floor(y0))), min(32, int(math.ceil(y1)))
    if b > a and d > c:
        frame[c:d, a:b] = color


def circle(frame, x, y, radius, color):
    frame[(X_GRID - x) ** 2 + (Y_GRID - y) ** 2 <= radius ** 2] = color


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class PixelGame(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    game = "abstract"
    available_actions = (0,)

    def __init__(self, *, max_steps=240, difficulty=1.0, render_mode=None):
        self.config = VisualConfig(max_steps=max_steps, difficulty=difficulty)
        if render_mode not in (None, "rgb_array"):
            raise ValueError("only rgb_array rendering is supported")
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(0, 255, (32, 32, 3), dtype=np.uint8)
        self.steps = 0
        self.finished = True
        self.state: dict[str, Any] = {}

    @property
    def identity(self):
        return f"visual-{self.game}-v1"

    def action_mask(self):
        # Constant input availability, never a mask of safe/preferred actions.
        mask = np.zeros(10, dtype=bool)
        mask[list(self.available_actions)] = True
        return mask

    def _info(self, **extra):
        return {"environment_id": self.identity, "steps": self.steps,
                "action_mask": self.action_mask(), **extra}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("use explicit versioned config, not reset overrides")
        self.steps, self.finished = 0, False
        self._reset_state()
        return self.render(), self._info(**self.metrics())

    def step(self, action):
        if self.finished:
            raise RuntimeError("reset is required before another step")
        if not self.action_space.contains(action):
            raise ValueError("action is outside the joypad space")
        # An unbound button has no physical effect, as on a real controller.
        # It remains excluded from the learner's available-input mask.
        physical_action = int(action) if self.action_mask()[int(action)] else 0
        self.steps += 1
        reward, terminated, reason = self._advance(physical_action)
        truncated = self.steps >= self.config.max_steps and not terminated
        self.finished = bool(terminated or truncated)
        return (self.render(), float(reward), bool(terminated), bool(truncated),
                self._info(terminal_reason=reason if terminated else
                           "step_limit" if truncated else "", **self.metrics()))

    def snapshot(self):
        return {"schema_version": 1, "environment_id": self.identity,
                "config": asdict(self.config), "steps": self.steps,
                "finished": self.finished, "state": _json_value(self.state),
                "rng": _json_value(self.np_random.bit_generator.state)}

    def restore(self, snapshot):
        if (snapshot.get("schema_version") != 1 or
                snapshot.get("environment_id") != self.identity or
                snapshot.get("config") != asdict(self.config)):
            raise ValueError("snapshot environment/schema/config mismatch")
        if type(snapshot["steps"]) is not int or not 0 <= snapshot["steps"] <= self.config.max_steps:
            raise ValueError("invalid snapshot step counter")
        old = (self.steps, self.finished, copy.deepcopy(self.state),
               copy.deepcopy(self.np_random.bit_generator.state))
        try:
            self.steps = snapshot["steps"]
            self.finished = bool(snapshot["finished"])
            self.state = copy.deepcopy(snapshot["state"])
            self.np_random.bit_generator.state = copy.deepcopy(snapshot["rng"])
            self._validate_state()
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            self.steps, self.finished, self.state, rng = old
            self.np_random.bit_generator.state = rng
            raise ValueError("invalid physical snapshot state") from exc
        return self.render()

    def _validate_state(self):
        # Snapshots are verified by a trusted outer digest before restore.
        # This still rejects nonfinite values and an incomplete state schema.
        def finite(value):
            if isinstance(value, (float, int)):
                return math.isfinite(value)
            if isinstance(value, (list, tuple)):
                return all(finite(x) for x in value)
            if isinstance(value, dict):
                return all(finite(x) for x in value.values())
            return isinstance(value, (str, bool)) or value is None
        if not isinstance(self.state, dict) or not finite(self.state):
            raise ValueError("invalid physical snapshot state")
        self.render()  # validates required physical fields without an episode.


class VisualSnake(PixelGame):
    """A new image-observation lineage, not the historical 11-feature Snake."""
    game = "snake"
    available_actions = (0, 1, 2, 3, 4)

    def _reset_state(self):
        n = 8
        direction = int(self.np_random.integers(4))
        vectors = ((0, -1), (0, 1), (-1, 0), (1, 0))
        dx, dy = vectors[direction]
        head = (n // 2, n // 2)
        self.state = {"size": n, "body": [[head[0]-i*dx, head[1]-i*dy]
                      for i in range(3)], "direction": direction+1,
                      "food": None, "foods": 0, "collision": ""}
        self._food()

    def _food(self):
        s = self.state
        free = [[x, y] for y in range(s["size"]) for x in range(s["size"])
                if [x, y] not in s["body"]]
        s["food"] = free[int(self.np_random.integers(len(free)))] if free else None

    def _advance(self, action):
        s = self.state
        opposite = {1: 2, 2: 1, 3: 4, 4: 3}
        if action and action != opposite[s["direction"]]:
            s["direction"] = action
        dx, dy = DIRECTIONS[s["direction"]]
        head = [s["body"][0][0]+int(dx), s["body"][0][1]+int(dy)]
        eats = head == s["food"]
        if not all(0 <= p < s["size"] for p in head):
            s["collision"] = "wall"
        elif head in (s["body"] if eats else s["body"][:-1]):
            s["collision"] = "body"
        if s["collision"]:
            return -1., True, s["collision"]
        s["body"].insert(0, head)
        if eats:
            s["foods"] += 1
            self._food()
            return 1., s["food"] is None, "filled" if s["food"] is None else ""
        s["body"].pop()
        return -.002, False, ""

    def metrics(self):
        return {"score": self.state["foods"], "foods": self.state["foods"],
                "wall_collision": int(self.state["collision"] == "wall"),
                "body_collision": int(self.state["collision"] == "body")}

    def render(self):
        s, f = self.state, canvas()
        k = 30/s["size"]
        rect(f, 0, 0, 32, 1, WALL); rect(f, 0, 31, 32, 32, WALL)
        rect(f, 0, 0, 1, 32, WALL); rect(f, 31, 0, 32, 32, WALL)
        if s["food"] is not None:
            x, y = s["food"]
            rect(f, 1+x*k, 1+y*k, 1+(x+1)*k, 1+(y+1)*k, FOOD)
        for i, (x, y) in enumerate(s["body"]):
            rect(f, 1+x*k, 1+y*k, 1+(x+1)*k, 1+(y+1)*k,
                 PLAYER if i == 0 else (35, 100, 150))
        return f


class VisualPong(PixelGame):
    game = "pong"
    available_actions = (0, 1, 2)

    def _serve(self):
        s = self.state
        s["ball"] = [16., float(self.np_random.uniform(7, 25))]
        # Serve towards the learned player: scoring requires an actual return.
        s["velocity"] = [-.75,
                         float(self.np_random.choice([-.65, -.35, .35, .65]))]

    def _reset_state(self):
        self.state = {"left": 16., "right": 16., "ball": [16., 16.],
                      "velocity": [.75, .35], "points": 0, "conceded": 0,
                      "hits": 0, "incoming": 0}
        self._serve()

    def _advance(self, action):
        s = self.state
        s["left"] = float(np.clip(s["left"] + DIRECTIONS[action, 1]*1.1, 3, 29))
        # Explicit simulator opponent, not the learned player.
        if self.steps % 2:
            s["right"] += float(np.clip(s["ball"][1]-s["right"], -.65, .65))
        x, y = s["ball"]; vx, vy = s["velocity"]
        x, y = x+vx, y+vy
        if y < 1 or y > 31:
            y = 2-y if y < 1 else 62-y
            vy = -vy
        reward = -.001
        if vx < 0 and x <= 3:
            s["incoming"] += 1
            if abs(y-s["left"]) <= 3:
                x, vx = 6-x, abs(vx)
                s["hits"] += 1
                reward = .15
            else:
                s["conceded"] += 1
                reward = -1.
                self._serve()
                return reward, s["conceded"] >= 3, "points" if s["conceded"] >= 3 else ""
        if vx > 0 and x >= 29:
            if abs(y-s["right"]) <= 3:
                x, vx = 58-x, -abs(vx)
            else:
                s["points"] += 1
                self._serve()
                return 1., s["points"] >= 3, "points" if s["points"] >= 3 else ""
        s["ball"], s["velocity"] = [x, y], [vx, vy]
        return reward, False, ""

    def metrics(self):
        s = self.state
        return {"score": s["points"]-s["conceded"], "points": s["points"],
                "conceded": s["conceded"], "hits": s["hits"],
                "incoming": s["incoming"]}

    def render(self):
        s, f = self.state, canvas()
        rect(f, 1, s["left"]-3, 3, s["left"]+3, PLAYER)
        rect(f, 29, s["right"]-3, 31, s["right"]+3, OTHER)
        circle(f, *s["ball"], 1.1, FOOD)
        rect(f, 0, 0, 32, 1, WALL); rect(f, 0, 31, 32, 32, WALL)
        return f


class FlappyBird(PixelGame):
    game = "flappy"
    available_actions = (0, 9)

    def _reset_state(self):
        self.state = {"y": float(self.np_random.uniform(12, 20)), "vy": 0.,
                      "pipes": [], "passed": 0, "collision": False, "flaps": 0}
        for x in (24., 42., 60.):
            self.state["pipes"].append([x, float(self.np_random.uniform(9, 23)), False])

    def _advance(self, action):
        s = self.state
        if action == 9:
            s["vy"] = -1.05
            s["flaps"] += 1
        s["vy"] = min(1.8, s["vy"]+.15)
        s["y"] += s["vy"]
        speed = .55*self.config.difficulty
        gap = 10.
        reward = .005
        for pipe in s["pipes"]:
            pipe[0] -= speed
            if not pipe[2] and pipe[0]+2 < 7:
                pipe[2] = True
                s["passed"] += 1
                reward += 1.
            if pipe[0]-2 <= 8.2 and pipe[0]+2 >= 5.8 and abs(s["y"]-pipe[1]) > gap/2-1.2:
                s["collision"] = True
        if s["pipes"][0][0] < -3:
            s["pipes"].pop(0)
            s["pipes"].append([s["pipes"][-1][0]+18,
                                  float(self.np_random.uniform(9, 23)), False])
        s["collision"] |= not 1.2 <= s["y"] <= 30.8
        return (-1. if s["collision"] else reward), s["collision"], "collision" if s["collision"] else ""

    def metrics(self):
        s = self.state
        return {"score": s["passed"], "pipes_passed": s["passed"],
                "collision": int(s["collision"]), "flaps": s["flaps"]}

    def render(self):
        s, f = self.state, canvas()
        for x, y, _ in s["pipes"]:
            rect(f, x-2, 0, x+2, y-5, WALL)
            rect(f, x-2, y+5, x+2, 32, WALL)
        circle(f, 7, s["y"], 1.4, PLAYER)
        return f


class AgarLike(PixelGame):
    game = "agar"
    available_actions = tuple(range(9))

    def _reset_state(self):
        self.state = {"player": [16., 16., 10.], "foods": 0, "prey_eaten": 0,
                      "eaten": False, "large_near_steps": 0,
                      "large_distance_change": 0., "small_near_steps": 0,
                      "small_distance_change": 0.,
                      "pellets": self.np_random.uniform(2, 30, (18, 2)).tolist(),
                      "bots": []}
        for mass in (4., 5., 25.):
            # Larger bot is initially separated, but always reaches the field.
            angle = float(self.np_random.uniform(0, 2*math.pi))
            x, y = 16+13*math.cos(angle), 16+13*math.sin(angle)
            self.state["bots"].append([x, y, mass,
                                      float(self.np_random.uniform(0, 2*math.pi))])

    @staticmethod
    def _radius(mass):
        return .38*math.sqrt(mass)

    def _advance(self, action):
        s = self.state
        p = s["player"]
        old = np.array(p[:2])
        speed = .95/math.sqrt(p[2]/10)
        new = np.clip(old + DIRECTIONS[action]*speed, 1, 31)
        p[0], p[1] = float(new[0]), float(new[1])
        reward = -.001
        for i, food in enumerate(s["pellets"]):
            if np.linalg.norm(new-food) <= self._radius(p[2])+.45:
                s["foods"] += 1
                p[2] += .3
                reward += .15
                s["pellets"][i] = self.np_random.uniform(1, 31, 2).tolist()
        for bot in s["bots"]:
            before = math.dist(old, bot[:2])
            after = math.dist(new, bot[:2])
            larger = bot[2] > p[2]*1.2
            smaller = p[2] > bot[2]*1.2
            if before < 9 and larger:
                s["large_near_steps"] += 1
                s["large_distance_change"] += after-before
            if before < 9 and smaller:
                s["small_near_steps"] += 1
                s["small_distance_change"] += after-before
            # Environmental opponents: pursuit by larger cells, seeded wandering
            # by small cells. These controls are never used for the player.
            if larger:
                vector = new-np.array(bot[:2])
                length = float(np.linalg.norm(vector))
                velocity = vector/max(length, 1e-8)*.50*self.config.difficulty
            else:
                bot[3] += float(self.np_random.normal(0, .18))
                velocity = np.array((math.cos(bot[3]), math.sin(bot[3])))*.45
            pos = np.clip(np.array(bot[:2])+velocity, 1, 31)
            bot[0], bot[1] = float(pos[0]), float(pos[1])
            distance = math.dist(new, bot[:2])
            if larger and distance < self._radius(bot[2]):
                s["eaten"] = True
                return -1., True, "eaten_by_larger"
            if smaller and distance < self._radius(p[2]):
                s["prey_eaten"] += 1
                p[2] += bot[2]*.6
                reward += 1.
                bot[0], bot[1] = map(float, self.np_random.uniform(1, 31, 2))
                bot[2] = float(self.np_random.uniform(3, 6))
        return reward, False, ""

    def metrics(self):
        s = self.state
        return {"score": s["foods"]*.15+s["prey_eaten"], "foods": s["foods"],
                "prey_eaten": s["prey_eaten"], "eaten": int(s["eaten"]),
                "mass": s["player"][2], "large_near_steps": s["large_near_steps"],
                "large_distance_change": s["large_distance_change"],
                "small_near_steps": s["small_near_steps"],
                "small_distance_change": s["small_distance_change"]}

    def render(self):
        s, f = self.state, canvas()
        for x, y in s["pellets"]:
            circle(f, x, y, .65, FOOD)
        for x, y, mass, _ in s["bots"]:
            circle(f, x, y, self._radius(mass), OTHER)
        circle(f, *s["player"][:2], self._radius(s["player"][2]), PLAYER)
        return f


class Racing(PixelGame):
    game = "racing"
    available_actions = tuple(range(9))

    def _center(self, position):
        s = self.state
        return 16 + s["amplitude"]*math.sin(position/s["wavelength"]+s["phase"])

    def _reset_state(self):
        self.state = {"position": 0., "speed": .25,
                      "phase": float(self.np_random.uniform(0, 2*math.pi)),
                      "amplitude": float(self.np_random.uniform(4, 7)),
                      "wavelength": float(self.np_random.uniform(12, 22))/self.config.difficulty,
                      "x": 16., "collision": False, "brakes": 0,
                      "accelerations": 0, "turns": 0,
                      "curved_steps": 0, "curved_distance": 0.}
        self.state["x"] = self._center(0)

    def _advance(self, action):
        s = self.state
        dx, dy = DIRECTIONS[action]
        if dy < 0:
            s["speed"] += .07
            s["accelerations"] += 1
        elif dy > 0:
            s["speed"] -= .12
            s["brakes"] += 1
        else:
            s["speed"] -= .004
        s["speed"] = float(np.clip(s["speed"], 0, 1.7))
        # Limited tyre grip makes turning at high speed harder. Slowing down
        # is useful on bends; a permanent full-throttle controller is not enough.
        s["x"] += float(dx)*.85/(1+s["speed"]**2)
        s["turns"] += int(dx != 0)
        s["position"] += s["speed"]
        curvature = abs(math.sin(s["position"]/s["wavelength"]+s["phase"]))
        if curvature > .6:
            s["curved_steps"] += 1
            s["curved_distance"] += s["speed"]
        s["collision"] = abs(s["x"]-self._center(s["position"])) > 4.8
        if s["collision"]:
            return -1., True, "off_track"
        return .035*s["speed"]-.003, False, ""

    def metrics(self):
        s = self.state
        return {"score": s["position"], "distance": s["position"],
                "speed": s["speed"], "collision": int(s["collision"]),
                "accelerations": s["accelerations"], "brakes": s["brakes"],
                "turns": s["turns"], "curved_steps": s["curved_steps"],
                "curved_distance": s["curved_distance"]}

    def render(self):
        s, f = self.state, canvas()
        for y in range(3, 32):
            center = self._center(s["position"]+(26-y)*1.5)
            rect(f, center-6, y, center+6, y+1, (55, 60, 65))
            rect(f, center-6, y, center-5, y+1, WALL)
            rect(f, center+5, y, center+6, y+1, WALL)
        rect(f, s["x"]-1, 25, s["x"]+1, 28, PLAYER)
        # A visible speedometer is part of the game display, not an optimal action.
        rect(f, 1, 0, 1+29*s["speed"]/1.7, 2, FOOD)
        return f


GAMES = {"snake": VisualSnake, "pong": VisualPong, "agar": AgarLike,
         "flappy": FlappyBird, "racing": Racing}


def make_game(name, **config):
    if name not in GAMES:
        raise ValueError(f"unknown local game: {name}")
    return GAMES[name](**config)
