
import copy
import time

from environment import Environment, ENV_CONFIGS
from agent import RescueAgent, DONE, FAILED
from search_algorithms import ALGORITHMS


# 1. Compare the algorithms on ONE search (e.g. agent -> victim)
def compare_searches(env, start, goal, algorithms=None, repeat=20):
    """Run every algorithm once on the same (start, goal) and compare.
    The search is repeated `repeat` times and the time is averaged, because a
    single search on a small grid takes only a few microseconds."""
    algorithms = algorithms or ALGORITHMS
    rows = []
    for name, fn in algorithms.items():
        path, explored, _ = fn(env, start, goal)
        total = 0.0
        for _ in range(repeat):
            total += fn(env, start, goal)[2]
        rows.append({
            "Algorithm": name,
            "Path Found": path is not None,
            "Path Length": len(path) - 1 if path else 0,     # number of moves
            "Nodes Explored": explored,
            "Time (s)": total / repeat,
        })
    return rows



# 2. "Compare All": a complete rescue mission for each algorithm
def simulate_mission(env, search_fn, max_energy=100, block_at_step=None, max_ticks=5000):
    """Run a whole rescue (all victims, then the exit) with Member 1's agent.
    Same loop as run_mission() in main.py, plus a safety limit against endless loops.
    If block_at_step is given, one hazard appears on the agent's path at that step
    (this triggers re-planning). Returns the finished agent."""
    agent = RescueAgent(env, max_energy=max_energy)
    if not agent.plan(search_fn):
        return agent

    blocked_done = False
    for _ in range(max_ticks):
        if (block_at_step is not None and not blocked_done
                and agent.steps_taken == block_at_step and agent.path):
            env.block_cell_on_path(agent.path[:4], avoid={agent.position},
                                   agent_pos=agent.position)
            blocked_done = True

        status = agent.step()
        if status in ("REPLAN", "NEED_PLAN", "VICTIM_RESCUED"):
            if not agent.plan(search_fn):
                break
        if agent.state in (DONE, FAILED):
            break
    else:
        agent.fail("Simulation step limit reached")
    return agent


def compare_all(env, algorithms=None, max_energy=100, block_at_step=None):
    """Solve the SAME environment with every algorithm and return one result row
    per algorithm. The original `env` is never changed: each algorithm gets
    its own deep copy of the map."""
    algorithms = algorithms or ALGORITHMS
    rows = []
    for name, fn in algorithms.items():
        agent = simulate_mission(copy.deepcopy(env), fn, max_energy, block_at_step)
        rows.append({
            "Environment": env.env_type,
            "Algorithm": name,
            "Status": agent.state,
            "Victims Rescued": f"{agent.victims_rescued}/{agent.total_victims_target}",
            "Path Length": agent.steps_taken,                # total moves walked
            "Nodes Explored": agent.total_nodes_explored,
            "Time (s)": agent.total_search_time,
            "Re-plans": agent.replan_count,
            "Energy Left": agent.energy,
            "Fail Reason": agent.fail_reason,
        })
    return rows



# 3. Benchmark: average over many random maps (for reliable graphs)
def benchmark(env_types=None, runs=30, num_victims=3, rows=10, cols=10,
              max_energy=100, first_seed=0):
    """For every environment type, run `runs` random maps with every algorithm.
    Each environment gets its own range of seeds, so the environments are
    tested on different random maps. Averages use the missions that finished
    successfully; the success rate shows how often the rescue was completed."""
    env_types = env_types or list(ENV_CONFIGS)
    results = []
    for index, env_type in enumerate(env_types):
        seed_start = first_seed + index * 1000
        stats = {name: {"ok": 0, "path": 0, "nodes": 0, "time": 0.0} for name in ALGORITHMS}
        for seed in range(seed_start, seed_start + runs):
            env = Environment(env_type, rows=rows, cols=cols,
                              num_victims=num_victims, seed=seed)
            for name, fn in ALGORITHMS.items():
                agent = simulate_mission(copy.deepcopy(env), fn, max_energy)
                if agent.state == DONE:
                    s = stats[name]
                    s["ok"] += 1
                    s["path"] += agent.steps_taken
                    s["nodes"] += agent.total_nodes_explored
                    s["time"] += agent.total_search_time
        for name, s in stats.items():
            ok = max(s["ok"], 1)
            results.append({
                "Environment": env_type,
                "Algorithm": name,
                "Runs": runs,
                "Success Rate (%)": 100.0 * s["ok"] / runs,
                "Avg Path Length": s["path"] / ok,
                "Avg Nodes Explored": s["nodes"] / ok,
                "Avg Time (ms)": 1000.0 * s["time"] / ok,
            })
    return results



# 4. Output: terminal table and graphs

def print_table(rows, title=None):
    if not rows:
        return

    def fmt(col, v):
        if isinstance(v, float):
            return f"{v:.5f}" if "(s)" in col else f"{v:.1f}"
        return str(v)

    cols = list(rows[0].keys())
    table = [[fmt(c, r[c]) for c in cols] for r in rows]
    widths = [max(len(c), *(len(line[i]) for line in table)) for i, c in enumerate(cols)]
    line = "-+-".join("-" * w for w in widths)
    if title:
        print("\n" + title)
    print(" | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print(line)
    for row in table:
        print(" | ".join(v.ljust(w) for v, w in zip(row, widths)))


# (column in the rows, title of the graph, multiplier)
MISSION_METRICS = [("Path Length", "Path length (moves)", 1),
                   ("Nodes Explored", "Nodes explored", 1),
                   ("Time (s)", "Search time (ms)", 1000)]
BENCHMARK_METRICS = [("Avg Path Length", "Avg path length (moves)", 1),
                     ("Avg Nodes Explored", "Avg nodes explored", 1),
                     ("Avg Time (ms)", "Avg search time (ms)", 1)]
COLORS = {"BFS": "#2563EB", "DFS": "#F59E0B", "Bidirectional": "#10B981"}


def plot_comparison(rows, title="Algorithm comparison", save_path=None, show=True):
    """Draw one bar graph per metric (path length, nodes explored, time).
    Works with the rows from compare_all() and from benchmark(). A hatched bar
    means that mission FAILED. Needs:  pip install matplotlib
    Returns the Matplotlib figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Graphs need Matplotlib. Install it with:  pip install matplotlib")

    metrics = BENCHMARK_METRICS if "Avg Path Length" in rows[0] else MISSION_METRICS
    groups = list(dict.fromkeys(r.get("Environment", "") for r in rows))
    names = list(dict.fromkeys(r["Algorithm"] for r in rows))
    width = 0.8 / len(names)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    for ax, (col, label, scale) in zip(axes, metrics):
        for i, name in enumerate(names):
            xs, ys, failed = [], [], []
            for g, group in enumerate(groups):
                row = next(r for r in rows
                           if r["Algorithm"] == name and r.get("Environment", "") == group)
                xs.append(g + (i - (len(names) - 1) / 2) * width)
                ys.append(row[col] * scale)
                failed.append(row.get("Status") == FAILED)
            bars = ax.bar(xs, ys, width, label=name, color=COLORS.get(name))
            for bar, f in zip(bars, failed):
                if f:
                    bar.set_hatch("//")
            if len(groups) == 1:                    # print the value above each bar
                for bar, y in zip(bars, ys):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{y:.2f}" if y < 10 else f"{y:.0f}",
                            ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=15 if len(groups) > 1 else 0)
        ax.set_title(label)
        ax.margins(y=0.12)
        ax.grid(axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(names), frameon=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig



# Demo

if __name__ == "__main__":
    env = Environment("Earthquake", rows=10, cols=10, num_victims=3, seed=5)

    goal = env.victim_pos
    print_table(compare_searches(env, env.agent_start, goal),
                f"SEARCH 1 only: agent {env.agent_start} -> first victim {goal}")

    mission_rows = compare_all(env, block_at_step=3)
    print_table(mission_rows, "COMPARE ALL: full rescue mission (a hazard appears at step 3)")

    print("\nRunning benchmark (30 random maps x 4 environments x 3 algorithms) ...")
    bench_rows = benchmark(runs=30)
    print_table(bench_rows, "BENCHMARK: averages over 30 random maps per environment")

    try:
        plot_comparison(mission_rows, "Compare All - Earthquake",
                        save_path="compare_all.png", show=False)
        plot_comparison(bench_rows, "Benchmark - 30 random maps per environment",
                        save_path="benchmark.png", show=True)
        print("\nGraphs saved as compare_all.png and benchmark.png")
    except ImportError as err:
        print("\n" + str(err))
