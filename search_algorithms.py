import time
from collections import deque

def _neighbors(env, pos, valid_cells=None, allow=None):
   
    result = []
    for n in env.get_neighbors(pos):
        if n != allow:
            if env.is_blocked(n):
                continue
            if valid_cells is not None and n not in valid_cells:
                continue
        result.append(n)
    return result


def _chain(parent, node):
    """Follow parent links: [node, parent[node], ..., root]."""
    out = []
    while node is not None:
        out.append(node)
        node = parent[node]
    return out


def _log(explored_log, cell):
    if explored_log is not None:
        explored_log.append(cell)



# 1. Breadth-First Search

def bfs(env, start, goal, valid_cells=None, explored_log=None):
    """Explores the grid level by level using a FIFO queue.
    Every move costs 1, so the path found is the SHORTEST one."""
    t0 = time.perf_counter()
    queue = deque([start])
    parent = {start: None}          # also acts as the "visited" set
    explored = 0

    while queue:
        cur = queue.popleft()
        explored += 1
        _log(explored_log, cur)

        if cur == goal:
            path = _chain(parent, cur)[::-1]
            return path, explored, time.perf_counter() - t0

        for nxt in _neighbors(env, cur, valid_cells):
            if nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)

    return None, explored, time.perf_counter() - t0



# 2. Depth-First Search
def dfs(env, start, goal, valid_cells=None, explored_log=None):
    
    t0 = time.perf_counter()
    stack = [(start, None)]         # (cell, cell we came from)
    parent = {}                     # filled when a cell is expanded = visited
    explored = 0

    while stack:
        cur, came_from = stack.pop()
        if cur in parent:           # already expanded through another branch
            continue
        parent[cur] = came_from
        explored += 1
        _log(explored_log, cur)

        if cur == goal:
            path = _chain(parent, cur)[::-1]
            return path, explored, time.perf_counter() - t0

        # push in reverse so the FIRST neighbour is popped (explored) first
        for nxt in reversed(_neighbors(env, cur, valid_cells)):
            if nxt not in parent:
                stack.append((nxt, cur))

    return None, explored, time.perf_counter() - t0



# 3. Bidirectional Search

def bidirectional_search(env, start, goal, valid_cells=None, explored_log=None):
    #"""Runs BFS from start and goal until they meet."""
    t0 = time.perf_counter()

    if start == goal:
        _log(explored_log, start)
        return [start], 1, time.perf_counter() - t0

    # If the goal itself is not enterable, no path exists (same result as BFS).
    if env.is_blocked(goal) or (valid_cells is not None and goal not in valid_cells):
        return None, 0, time.perf_counter() - t0

    parent_f = {start: None}        # forward search: cell -> previous cell
    parent_b = {goal: None}         # backward search: cell -> next cell (towards goal)
    frontier_f = [start]
    frontier_b = [goal]
    explored = 0
    meeting = None

    while frontier_f and frontier_b and meeting is None:
        forward = len(frontier_f) <= len(frontier_b)
        if forward:
            frontier, own, other, allow = frontier_f, parent_f, parent_b, None
        else:
            frontier, own, other, allow = frontier_b, parent_b, parent_f, start

        next_frontier = []
        for cur in frontier:
            explored += 1
            _log(explored_log, cur)
            for nxt in _neighbors(env, cur, valid_cells, allow):
                if nxt in own:
                    continue
                own[nxt] = cur
                if nxt in other:    # the two searches have met
                    meeting = nxt
                    break
                next_frontier.append(nxt)
            if meeting is not None:
                break

        if forward:
            frontier_f = next_frontier
        else:
            frontier_b = next_frontier

    if meeting is None:
        return None, explored, time.perf_counter() - t0

    first_half = _chain(parent_f, meeting)[::-1]     # start ... meeting
    second_half = _chain(parent_b, meeting)          # meeting ... goal
    path = first_half + second_half[1:]
    return path, explored, time.perf_counter() - t0



# Registry used by the GUI drop-down and by the comparison module

ALGORITHMS = {
    "BFS": bfs,
    "DFS": dfs,
    "Bidirectional": bidirectional_search,
}

AUTO_ALGORITHM = "Auto (recommended)"


def select_algorithm(env, start=None, goal=None):
    #Choose a search strategy from the environment and current route.

    preferred = {
        "Building Fire": "BFS",
        "Earthquake": "DFS",
        "Flood": "Bidirectional",
        "Industrial Accident": "BFS",
    }
    name = preferred.get(env.env_type, "BFS")

    total_cells = env.rows * env.cols
    blocked_cells = sum(
        env.is_blocked((row, col))
        for row in range(env.rows)
        for col in range(env.cols)
    )
    blocked_ratio = blocked_cells / total_cells if total_cells else 0

    if start is not None and goal is not None:
        distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        # A long route through open space benefits from DFS's low memory use.
        if distance > total_cells * 0.55 and blocked_ratio < 0.10:
            name = "DFS"
        # Normal disaster maps can be 24-32% blocked; only unusually dense
        # maps should replace the disaster profile with Bidirectional Search.
        elif blocked_ratio > 0.38:
            name = "Bidirectional"

    return name
