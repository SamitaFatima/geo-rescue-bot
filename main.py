"""
main.py - RescueBot (Member 1)
Terminal harness and integration test script.
"""
import time
from collections import deque

from environment import Environment
from agent import RescueAgent, DONE, FAILED


def simple_bfs(env, start, goal):
    t0 = time.perf_counter()
    queue = deque([start])
    parent = {start: None}
    explored = 0
    while queue:
        cur = queue.popleft()
        explored += 1
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1], explored, time.perf_counter() - t0
        for nxt in env.get_valid_neighbors(cur):
            if nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)
    return None, explored, time.perf_counter() - t0


def run_mission(env, agent, search_fn, block_at_step=None):
    if not agent.plan(search_fn):
        return

    blocked_done = False
    while agent.state not in (DONE, FAILED):
        if (block_at_step is not None and not blocked_done
                and agent.steps_taken == block_at_step and agent.path):
            cell = env.block_cell_on_path(agent.path[:4], avoid={agent.position},
                                          agent_pos=agent.position)
            blocked_done = True

        status = agent.step()
        if status in ("REPLAN", "NEED_PLAN", "VICTIM_RESCUED"):
            if not agent.plan(search_fn):
                break


if __name__ == "__main__":
    env = Environment("Building Fire", rows=10, cols=10, num_victims=3)
    agent = RescueAgent(env, max_energy=100)
    run_mission(env, agent, simple_bfs, block_at_step=3)
    print(f"Mission Status: {agent.state} | Victims Rescued: {agent.victims_rescued}")