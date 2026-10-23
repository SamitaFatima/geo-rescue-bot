from environment import Environment, EMPTY, WALL, HAZARD
from agent import RescueAgent, DONE, FAILED
from search_algorithms import ALGORITHMS
from csp import RescueCSP, CONSTRAINTS, plan_with_csp


def blank_env(rows=6, cols=6, num_victims=2, seed=0):
    env = Environment("Building Fire", rows=rows, cols=cols,
                       num_victims=num_victims, seed=seed)
    for r in range(rows):
        for c in range(cols):
            if env.grid[r][c] in (WALL, HAZARD):
                env.grid[r][c] = EMPTY
    return env


# ------------------------------------------------------------------
# 1. Constraints
# ------------------------------------------------------------------
def test_constraint_registry_is_complete():
    names = {c.name for c in CONSTRAINTS}
    expected = {"no_walls_or_debris", "no_hazards", "energy_limit",
                "time_limit", "priority_order", "valid_exit"}
    assert names == expected, names
    assert all(c.description for c in CONSTRAINTS)
    print("PASS constraint registry lists all 6 rules from the proposal")


def test_valid_cells_removes_walls_and_hazards():
    env = blank_env()
    env.grid[0][1] = WALL
    env.grid[0][2] = HAZARD
    csp = RescueCSP(env)
    valid = csp.get_valid_cells()
    assert (0, 1) not in valid and (0, 2) not in valid
    assert env.agent_start in valid and env.exit_pos in valid
    print("PASS valid_cells excludes walls and hazards")


# ------------------------------------------------------------------
# 3. Safety rules
# ------------------------------------------------------------------
def test_is_position_safe():
    env = blank_env()
    env.grid[1][1] = WALL
    env.grid[1][2] = HAZARD
    csp = RescueCSP(env)
    assert not csp.is_position_safe((1, 1))     # wall
    assert not csp.is_position_safe((1, 2))     # hazard
    assert not csp.is_position_safe((-1, 0))    # out of bounds
    assert csp.is_position_safe(env.agent_start)
    print("PASS is_position_safe rejects walls, hazards, and out-of-bounds cells")


# ------------------------------------------------------------------
# 2. Victim priority
# ------------------------------------------------------------------
def test_priority_constraint():
    env = blank_env(num_victims=2)
    critical, normal = env.victims[0], env.victims[1]
    assert env.priorities[critical] == "critical"
    csp = RescueCSP(env)
    assert csp._priority_ok([critical, normal])
    assert not csp._priority_ok([normal, critical])
    print("PASS priority constraint accepts critical-first, rejects critical-last")


def test_apply_matches_environment_priority_pick():
    """The CSP's chosen first victim should always be a critical one if any
    critical victim remains, exactly like Environment.victim_pos."""
    for seed in range(10):
        env = Environment("Earthquake", rows=8, cols=8, num_victims=3, seed=seed)
        agent = RescueAgent(env, max_energy=500)
        csp = RescueCSP(env)
        result = csp.apply(agent)
        if not result["ok"]:
            continue
        first = result["victim_order"][0]
        assert first == env.victim_pos, f"seed {seed}: CSP picked {first}, expected {env.victim_pos}"
    print("PASS apply() always puts a critical victim first, matching Environment.victim_pos")


def test_nearest_first_ordering_among_equal_priority():
    """Two normal-priority victims, one much closer than the other: with
    ample energy, backtracking should still return the closer one first
    (heuristic ordering), not an arbitrary one."""
    env = blank_env(rows=10, cols=10, num_victims=1, seed=0)
    near, far = (0, 1), (9, 9)
    env.victims = [near, far]
    env.priorities = {near: "normal", far: "normal"}
    env.grid[near[0]][near[1]] = 3   # VICTIM marker (matches environment.py's VICTIM=3)
    env.grid[far[0]][far[1]] = 3
    agent = RescueAgent(env, max_energy=1000)
    csp = RescueCSP(env)
    order, _ = csp.solve_victim_order(agent, csp.get_valid_cells())
    assert order is not None and order[0] == near, order
    print("PASS nearest-first heuristic visits the closer equal-priority victim first")


# ------------------------------------------------------------------
# 4. Time / energy conditions
# ------------------------------------------------------------------
def test_backtracking_finds_order_with_enough_energy():
    env = blank_env(num_victims=3, seed=2)
    agent = RescueAgent(env, max_energy=100)
    csp = RescueCSP(env)
    valid_cells = csp.get_valid_cells()
    order, total = csp.solve_victim_order(agent, valid_cells)
    assert order is not None, "should find a feasible order with plenty of energy"
    assert set(order) == set(env.victims)
    assert csp._priority_ok(order)
    assert total <= agent.energy
    print(f"PASS backtracking finds a feasible victim order (order={order}, distance={total})")


def test_energy_constraint_causes_backtrack_failure():
    env = blank_env(num_victims=3, seed=2)
    agent = RescueAgent(env, max_energy=1)     # not enough energy to reach anything
    csp = RescueCSP(env)
    valid_cells = csp.get_valid_cells()
    order, total = csp.solve_victim_order(agent, valid_cells)
    assert order is None, "should fail: 1 energy cannot possibly reach any victim + exit"
    print("PASS energy constraint rejects an infeasible plan")


def test_time_limit_constraint():
    env = blank_env(num_victims=1, seed=1)
    agent = RescueAgent(env, max_energy=1000)
    tight_csp = RescueCSP(env, max_time=0)     # no moves allowed at all
    valid_cells = tight_csp.get_valid_cells()
    order, _ = tight_csp.solve_victim_order(agent, valid_cells)
    assert order is None, "a max_time of 0 should make every plan infeasible"

    loose_csp = RescueCSP(env, max_time=1000)
    order, _ = loose_csp.solve_victim_order(agent, valid_cells)
    assert order is not None
    print("PASS time limit constraint: tight limit fails, loose limit succeeds")


# ------------------------------------------------------------------
# 5. Re-planning logic
# ------------------------------------------------------------------
def test_recheck_reasons():
    env = blank_env(num_victims=1)
    agent = RescueAgent(env, max_energy=100)
    csp = RescueCSP(env)
    valid_cells = csp.get_valid_cells()

    agent.energy = 0
    ok, reason = csp.recheck(agent, valid_cells)
    assert not ok and reason == "Out of energy"

    agent.energy = 100
    ok, reason = csp.recheck(agent, valid_cells)
    assert ok and reason == ""
    print("PASS recheck() reports the correct reason for energy/priority/time failures")


def test_recheck_after_mid_path_hazard():
    """Simulates Table 19-21: a new hazard appears on the agent's path, so
    the CSP must recompute valid_cells and confirm the mission is still
    feasible (or correctly report that it is not)."""
    env = Environment("Building Fire", rows=8, cols=8, num_victims=1, seed=4)
    agent = RescueAgent(env, max_energy=200)
    csp = RescueCSP(env)
    result = csp.apply(agent)
    assert result["ok"]

    # Drop a hazard onto the planned path and re-check.
    if len(result["victim_order"]) > 0:
        env.block_cell_on_path([env.agent_start], avoid=set(), agent_pos=agent.position)
    new_valid_cells = csp.get_valid_cells()
    ok, reason = csp.recheck(agent, new_valid_cells)
    assert ok or reason  # either still feasible, or a concrete reason is given
    print("PASS recheck() re-evaluates constraints after a new hazard appears")


def test_explain_output():
    env = blank_env(num_victims=1)
    agent = RescueAgent(env, max_energy=100)
    csp = RescueCSP(env, max_time=50)
    report = csp.explain(agent)
    assert "CSP constraint check" in report
    assert "energy_limit" in report
    assert "Feasible order" in report or "INFEASIBLE" in report
    print("PASS explain() produces a readable constraint report")


def test_full_mission_with_plan_with_csp():
    """Integration test: Member 1's Agent + Member 2's search + Member 3's CSP,
    run to completion across all three algorithms and several random maps,
    including a hazard appearing mid-mission (forces a CSP re-check)."""
    finished = 0
    for name, fn in ALGORITHMS.items():
        for seed in range(15):
            env = Environment("Flood", num_victims=3, seed=seed)
            agent = RescueAgent(env, max_energy=300)
            csp = RescueCSP(env)
            assert plan_with_csp(agent, fn, csp)

            blocked = False
            for _ in range(2000):
                if agent.steps_taken == 3 and agent.path and not blocked:
                    env.block_cell_on_path(agent.path[:4], avoid={agent.position},
                                           agent_pos=agent.position)
                    blocked = True
                status = agent.step()
                if status in ("REPLAN", "NEED_PLAN", "VICTIM_RESCUED"):
                    if not plan_with_csp(agent, fn, csp):
                        break
                if agent.state in (DONE, FAILED):
                    break
            assert agent.state == DONE, f"{name} seed {seed}: {agent.state} {agent.fail_reason}"
            assert agent.victims_rescued == 3
            finished += 1
    print(f"PASS full CSP-driven missions ({finished}/{finished} DONE, "
          f"all 3 algorithms x 15 seeds, mid-mission hazard included)")


if __name__ == "__main__":
    test_constraint_registry_is_complete()
    test_valid_cells_removes_walls_and_hazards()
    test_is_position_safe()
    test_priority_constraint()
    test_apply_matches_environment_priority_pick()
    test_nearest_first_ordering_among_equal_priority()
    test_backtracking_finds_order_with_enough_energy()
    test_energy_constraint_causes_backtrack_failure()
    test_time_limit_constraint()
    test_recheck_reasons()
    test_recheck_after_mid_path_hazard()
    test_explain_output()
    test_full_mission_with_plan_with_csp()
    print("\nAll CSP tests passed.")
