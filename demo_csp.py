"""
demo_csp.py - RescueBot (Member 3)
Terminal demo of the CSP module: prints the constraint set, shows the
backtracking victim order, and runs a full rescue mission driven by
plan_with_csp (a hazard is dropped on the path mid-mission to trigger a
CSP re-check).
"""
from environment import Environment
from agent import RescueAgent, DONE
from search_algorithms import bfs
from csp import RescueCSP, plan_with_csp


def run(env_type="Earthquake", seed=5, max_energy=100, max_time=None):
    env = Environment(env_type, rows=10, cols=10, num_victims=3, seed=seed)
    agent = RescueAgent(env, max_energy=max_energy)
    csp = RescueCSP(env, max_time=max_time)

    print(f"Environment       : {env_type} (seed={seed})")
    print(f"Victims (priority): " +
          ", ".join(f"{v}={env.priorities[v]}" for v in env.victims))
    print()
    print(csp.explain(agent))
    print()

    result = csp.apply(agent)
    if not result["ok"]:
        print(f"CSP says mission is already infeasible: {result['reason']}")
        return

    assert plan_with_csp(agent, bfs, csp)
    blocked_done = False
    while agent.state not in (DONE, "FAILED"):
        if not blocked_done and agent.steps_taken == 3 and agent.path:
            cell = env.block_cell_on_path(agent.path[:4], avoid={agent.position},
                                          agent_pos=agent.position)
            blocked_done = True
            if cell:
                print(f"  -> hazard appeared at {cell}, agent will re-plan through the CSP")

        status = agent.step()
        if status in ("REPLAN", "NEED_PLAN", "VICTIM_RESCUED"):
            if not plan_with_csp(agent, bfs, csp):
                break

    print("\nRESCUE COMPLETE" if agent.state == DONE else "\nRESCUE FAILED")
    print(f"Environment      : {env_type}")
    print(f"Algorithm        : BFS")
    print(f"Victims Rescued  : {agent.victims_rescued}/{agent.total_victims_target}")
    print(f"Path Length      : {agent.steps_taken}")
    print(f"Nodes Explored   : {agent.total_nodes_explored}")
    print(f"Time             : {agent.total_search_time:.5f} seconds")
    print(f"Re-plans         : {agent.replan_count}")
    print(f"Energy Left      : {agent.energy}")
    if agent.state != DONE:
        print(f"Fail Reason      : {agent.fail_reason}")


if __name__ == "__main__":
    print("=== Demo 1: normal energy budget ===")
    run(max_energy=100)

    print("\n=== Demo 2: energy budget too tight -> CSP reports infeasible up front ===")
    run(max_energy=3)

    print("\n=== Demo 3: optional time limit constraint too tight ===")
    run(max_energy=100, max_time=2)
