import random

from environment import Environment, ENV_CONFIGS, WALL, HAZARD, EMPTY
from agent import RescueAgent, DONE, FAILED
from search_algorithms import bfs, dfs, bidirectional_search, ALGORITHMS


def check_valid_path(env, path, start, goal):
    
    assert path[0] == start and path[-1] == goal, "wrong start/goal"
    assert len(set(path)) == len(path), "path visits a cell twice"
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, f"jump between {a} and {b}"
        assert not env.is_blocked(b), f"path enters blocked cell {b}"


def test_interface():
    env = Environment("Earthquake", seed=1)
    for name, fn in ALGORITHMS.items():
        path, explored, seconds = fn(env, env.agent_start, env.exit_pos)
        assert isinstance(explored, int) and explored > 0, name
        assert isinstance(seconds, float) and seconds >= 0, name
        check_valid_path(env, path, env.agent_start, env.exit_pos)
    print("PASS interface  (path, explored, seconds) returned by all 3 algorithms")


def test_random_maps(n_maps=400):
    
    rng = random.Random(42)
    cases = dfs_longer = 0
    for i in range(n_maps):
        env_type = rng.choice(list(ENV_CONFIGS))
        env = Environment(env_type, rows=rng.randint(4, 15), cols=rng.randint(4, 15),
                          num_victims=rng.randint(1, 4), seed=i)
        # Add extra hazards so that many maps have NO path or long detours
        for _ in range(rng.randint(0, 12)):
            env.block_cell((rng.randrange(env.rows), rng.randrange(env.cols)))
        goals = list(env.victims) + [env.exit_pos]
        for goal in goals:
            p_bfs, _, _ = bfs(env, env.agent_start, goal)
            p_dfs, _, _ = dfs(env, env.agent_start, goal)
            p_bid, _, _ = bidirectional_search(env, env.agent_start, goal)
            assert (p_bfs is None) == (p_dfs is None) == (p_bid is None), \
                f"algorithms disagree on reachability (map {i})"
            if p_bfs is None:
                continue
            for p in (p_bfs, p_dfs, p_bid):
                check_valid_path(env, p, env.agent_start, goal)
            assert len(p_bid) == len(p_bfs), \
                f"Bidirectional not shortest: {len(p_bid)} vs BFS {len(p_bfs)} (map {i})"
            assert len(p_dfs) >= len(p_bfs)
            dfs_longer += len(p_dfs) > len(p_bfs)
            cases += 1
    print(f"PASS random maps ({cases} searches): valid paths, "
          f"Bidirectional == BFS length, DFS longer than BFS in {dfs_longer} cases")


def blank_env(rows=5, cols=5):
    env = Environment("Building Fire", rows=rows, cols=cols, num_victims=1, seed=0)
    for r in range(rows):
        for c in range(cols):
            if env.grid[r][c] in (WALL, HAZARD):
                env.grid[r][c] = EMPTY
    return env


def test_no_path():
    env = blank_env()
    for r in range(env.rows):                    # full wall across row 2
        env.grid[2][r] = WALL
    for name, fn in ALGORITHMS.items():
        path, explored, _ = fn(env, (0, 0), (4, 4))
        assert path is None, name
    print("PASS no path    all 3 return None when the goal is walled off")


def test_start_equals_goal():
    env = blank_env()
    for name, fn in ALGORITHMS.items():
        path, _, _ = fn(env, (1, 1), (1, 1))
        assert path == [(1, 1)], name
    print("PASS start==goal returns [start]")


def test_blocked_start():
    """Agent standing on a cell that just became a hazard must still be able to search."""
    env = blank_env()
    env.grid[0][0] = HAZARD
    for name, fn in ALGORITHMS.items():
        path, _, _ = fn(env, (0, 0), (4, 4))
        assert path is not None and path[0] == (0, 0), name
        check_valid_path(env, path, (0, 0), (4, 4))
        if name != "DFS":                        # DFS need not be shortest
            assert len(path) == 9, name
    print("PASS blocked start cell still works (agent standing on new hazard)")


def test_blocked_goal():
    env = blank_env()
    env.grid[4][4] = HAZARD
    for name, fn in ALGORITHMS.items():
        path, _, _ = fn(env, (0, 0), (4, 4))
        assert path is None, name
    print("PASS blocked goal returns None in all 3 algorithms")


def test_valid_cells_and_log():
    env = blank_env()
    allowed = {(r, c) for r in range(5) for c in range(5) if c == 0 or r == 4}   # L-shaped corridor
    for name, fn in ALGORITHMS.items():
        log = []
        path, explored, _ = fn(env, (0, 0), (4, 4), valid_cells=allowed, explored_log=log)
        assert set(path) <= allowed, name
        assert len(path) == 9, name
        assert len(log) == explored, name
    print("PASS optional valid_cells restricts the search; explored_log matches the count")


def test_big_grid_no_recursion_error():
    env = Environment("Flood", rows=120, cols=120, num_victims=1, seed=3)
    for name, fn in ALGORITHMS.items():
        path, _, _ = fn(env, env.agent_start, env.exit_pos)
        assert path is not None, name
    print("PASS 120 x 120 grid (iterative DFS, no recursion limit problem)")


def test_full_mission_with_agent():
    
    finished = 0
    for name, fn in ALGORITHMS.items():
        for seed in range(30):
            env = Environment("Building Fire", num_victims=3, seed=seed)
            agent = RescueAgent(env, max_energy=300)
            assert agent.plan(fn)
            blocked = False
            for _ in range(2000):
                if agent.steps_taken == 3 and agent.path and not blocked:
                    env.block_cell_on_path(agent.path[:4], avoid={agent.position},
                                           agent_pos=agent.position)
                    blocked = True
                status = agent.step()
                if status in ("REPLAN", "NEED_PLAN", "VICTIM_RESCUED"):
                    if not agent.plan(fn):
                        break
                if agent.state in (DONE, FAILED):
                    break
            assert agent.state == DONE, f"{name} seed {seed}: {agent.state} {agent.fail_reason}"
            assert agent.victims_rescued == 3
            finished += 1
    print(f"PASS full missions with Member 1's agent + mid-rescue hazard ({finished}/{finished} DONE)")


if __name__ == "__main__":
    test_interface()
    test_random_maps()
    test_no_path()
    test_start_equals_goal()
    test_blocked_start()
    test_blocked_goal()
    test_valid_cells_and_log()
    test_big_grid_no_recursion_error()
    test_full_mission_with_agent()
    print("\nAll tests passed.")
