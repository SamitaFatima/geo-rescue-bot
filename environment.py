"""
environment.py - RescueBot (Member 1)
Supports multiple victims, environment-specific hazards, and dynamic grid updates.
"""
import random
from collections import deque

EMPTY, WALL, HAZARD, VICTIM, EXIT = 0, 1, 2, 3, 4

ENV_CONFIGS = {
    "Building Fire":       {"wall": 0.12, "hazard": 0.12, "hazard_name": "fire"},
    "Earthquake":          {"wall": 0.22, "hazard": 0.04, "hazard_name": "aftershock zone"},
    "Flood":               {"wall": 0.08, "hazard": 0.20, "hazard_name": "deep water"},
    "Industrial Accident": {"wall": 0.12, "hazard": 0.14, "hazard_name": "toxic area"},
}

SYMBOLS = {EMPTY: ".", WALL: "#", HAZARD: "~", VICTIM: "V", EXIT: "E"}


class Environment:
    def __init__(self, env_type="Building Fire", rows=10, cols=10,
                 num_victims=3, seed=None):
        if env_type not in ENV_CONFIGS:
            raise ValueError(f"Unknown environment: {env_type}")
        self.env_type = env_type
        self.rows = rows
        self.cols = cols
        self.num_victims = num_victims
        self.config = ENV_CONFIGS[env_type]
        self.hazard_name = self.config["hazard_name"]
        self.rng = random.Random(seed)

        self.grid = []
        self.agent_start = (0, 0)
        self.exit_pos = (rows - 1, 0)
        self.victims = []          # List of remaining victim coordinates: [(r, c), ...]
        self.priorities = {}       # {victim_pos: "critical" or "normal"}
        self.generate()

    def generate(self):
        for _ in range(300):
            self._build_random_grid()
            if self._is_solvable():
                return
        self._build_random_grid(empty=True)

    def _build_random_grid(self, empty=False):
        self.grid = [[EMPTY] * self.cols for _ in range(self.rows)]
        if not empty:
            for r in range(self.rows):
                for c in range(self.cols):
                    x = self.rng.random()
                    if x < self.config["wall"]:
                        self.grid[r][c] = WALL
                    elif x < self.config["wall"] + self.config["hazard"]:
                        self.grid[r][c] = HAZARD

        sr, sc = self.agent_start
        er, ec = self.exit_pos
        self.grid[sr][sc] = EMPTY
        self.grid[er][ec] = EXIT

        free = [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.grid[r][c] == EMPTY and (r, c) != self.agent_start and (r, c) != self.exit_pos]
        self.rng.shuffle(free)
        self.victims = free[:self.num_victims]
        self.priorities = {}
        for i, v in enumerate(self.victims):
            self.grid[v[0]][v[1]] = VICTIM
            self.priorities[v] = "critical" if i == 0 else "normal"

    def _is_solvable(self):
        reachable = self._reachable_from(self.agent_start)
        return self.exit_pos in reachable and all(v in reachable for v in self.victims)

    def _reachable_from(self, start):
        seen = {start}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nxt in self.get_valid_neighbors(cur):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    @property
    def victim_pos(self):
        """Returns the next priority victim position, or None if all are rescued."""
        if not self.victims:
            return None
        # Return critical priority victim if present, otherwise first available
        for v in self.victims:
            if self.priorities.get(v) == "critical":
                return v
        return self.victims[0]

    def in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def cell_type(self, pos):
        return self.grid[pos[0]][pos[1]]

    def is_blocked(self, pos):
        return self.cell_type(pos) in (WALL, HAZARD)

    def get_neighbors(self, pos):
        r, c = pos
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (r + dr, c + dc)
            if self.in_bounds(nxt):
                result.append(nxt)
        return result

    def get_valid_neighbors(self, pos):
        return [n for n in self.get_neighbors(pos) if not self.is_blocked(n)]

    def block_cell(self, pos):
        if pos in self.victims or pos == self.exit_pos or pos == self.agent_start:
            return False
        self.grid[pos[0]][pos[1]] = HAZARD
        return True

    def block_cell_on_path(self, path, avoid=(), agent_pos=None):
        options = [p for p in path if self.cell_type(p) == EMPTY and p not in avoid]
        self.rng.shuffle(options)
        for pos in options:
            self.grid[pos[0]][pos[1]] = HAZARD
            if agent_pos is None or self._still_solvable_from(agent_pos):
                return pos
            self.grid[pos[0]][pos[1]] = EMPTY
        return None

    def _still_solvable_from(self, pos):
        reachable = self._reachable_from(pos)
        return self.exit_pos in reachable and all(v in reachable for v in self.victims)

    def remove_victim(self, pos):
        if pos in self.victims:
            self.victims.remove(pos)
            if pos in self.priorities:
                del self.priorities[pos]
            self.grid[pos[0]][pos[1]] = EMPTY

    # ------------------------------------------------------------------
    # RescueBot 2.0 (Member 3): time-stepped flood hazards.
    # Purely additive - nothing above this line was changed.
    # ------------------------------------------------------------------
    def apply_hazard_snapshot(self, risk_grid, protected=()):
        """Load one flood snapshot from Member 2's hazard engine.

        risk_grid is a rows x cols list of 0-3 ints (0 safe, 1 low,
        2 medium, 3 high/blocked) in the same (row, col) convention as
        this grid. Cells rated 3 become HAZARD so every existing search,
        the CSP and RescueAgent.step() treat them as blocked. Cells that
        an EARLIER snapshot turned into HAZARD and that are no longer
        rated 3 are restored to EMPTY; ordinary random hazards and walls
        are never touched. Victims, the exit, the agent start and any
        `protected` cell (e.g. where the robot currently stands) are never
        overwritten, mirroring block_cell().

        Returns the list of (row, col) cells newly blocked by this
        snapshot (useful for explaining a route change).
        """
        if len(risk_grid) != self.rows or any(len(row) != self.cols for row in risk_grid):
            raise ValueError("risk_grid size does not match the environment grid.")
        previous = getattr(self, "flood_cells", set())
        keep = set(protected) | set(self.victims) | {self.exit_pos, self.agent_start}

        for pos in previous:
            if self.grid[pos[0]][pos[1]] == HAZARD:
                self.grid[pos[0]][pos[1]] = EMPTY

        flooded = set()
        newly_blocked = []
        for r in range(self.rows):
            for c in range(self.cols):
                if risk_grid[r][c] >= 3 and (r, c) not in keep \
                        and self.grid[r][c] == EMPTY:
                    self.grid[r][c] = HAZARD
                    flooded.add((r, c))
                    if (r, c) not in previous:
                        newly_blocked.append((r, c))
        self.flood_cells = flooded
        self.risk_grid = [list(row) for row in risk_grid]
        return newly_blocked