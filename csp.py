"""
csp.py - RescueBot (Member 3)
Constraint Satisfaction Problem module: constraints, victim priority,
safety rules, energy/time conditions, and re-planning logic (proposal
sections 7 and 18.2).

This module does not move the agent and does not search for paths itself.
It sits between Member 1's Agent and Member 2's search algorithms:

    Member 1 (Agent) --percepts (position, hazards, energy)--> Member 3 (CSP)
    Member 3 (CSP)   --valid cells + victim order-->            Member 2 (Search)

CSP formulation (matches proposal section 7 and Table 1):
    Variables   : the position of each victim in the rescue sequence
                  (1st victim to rescue, 2nd victim, ...)
    Domains     : the victims that have not been placed in the sequence yet
    Constraints : the six rules in CONSTRAINTS below

Everything the spec asks of Member 3 lives in one of these five places:
    1. Constraints            -> CONSTRAINTS, list_constraints(), get_valid_cells()
    2. Victim priority        -> _priority_ok(), the priority-first search order
                                  inside solve_victim_order()
    3. Safety rules            -> is_position_safe(), get_valid_cells()
    4. Time/energy conditions  -> _energy_ok(), _time_ok()
    5. Re-planning logic       -> recheck(), apply(), plan_with_csp()
"""

from collections import namedtuple

from search_algorithms import bfs


# ----------------------------------------------------------------------
# 1. CONSTRAINTS - the explicit rule set from proposal section 7. Kept as
#    data (not just buried inside if-statements) so the whole team / a
#    grader can print the constraint list and see exactly what is enforced.
# ----------------------------------------------------------------------
Constraint = namedtuple("Constraint", ["name", "description"])

CONSTRAINTS = [
    Constraint("no_walls_or_debris",
               "The agent cannot enter blocked cells (walls, debris, collapsed structures)."),
    Constraint("no_hazards",
               "The agent cannot enter fire, unsafe water, or other hazardous zones."),
    Constraint("energy_limit",
               "Total energy used for the rest of the mission must not exceed the agent's remaining energy."),
    Constraint("time_limit",
               "The rescue must be completed within the mission time limit, if one is set."),
    Constraint("priority_order",
               "Critical victims must be rescued before normal-priority victims."),
    Constraint("valid_exit",
               "The mission can only end by reaching a reachable, safe exit cell."),
]


def _distance(env, start, goal, valid_cells):
    """Shortest number of MOVES from start to goal, restricted to valid_cells.
    Returns None if unreachable. Reuses Member 2's BFS purely as a distance
    oracle for the CSP -- it is not the algorithm used to actually move the
    agent (that choice is still made by whichever algorithm the user picked)."""
    path, _, _ = bfs(env, start, goal, valid_cells=valid_cells)
    if path is None:
        return None
    return len(path) - 1


class RescueCSP:
    """Applies the rescue mission's constraints for one Environment/Agent pair."""

    def __init__(self, env, max_time=None):
        self.env = env
        self.max_time = max_time     # optional time-limit constraint (section 13)
        self._dist_cache = {}

    # ------------------------------------------------------------------
    # 1. Constraints - list them, and restrict the domain to safe cells
    # ------------------------------------------------------------------
    def list_constraints(self):
        """Returns the full constraint set as (name, description) pairs,
        for printing in a demo/report -- see CONSTRAINTS above."""
        return list(CONSTRAINTS)

    def is_position_safe(self, pos):
        """Safety rule (no_walls_or_debris + no_hazards): a cell may only be
        entered if it is inside the grid and is not a wall or a hazard."""
        return self.env.in_bounds(pos) and not self.env.is_blocked(pos)

    def get_valid_cells(self):
        """'Apply CSP constraints' step: restrict the domain of usable cells
        to the ones that pass the safety rule. Everything else in this
        module assumes the agent may only occupy a cell in this set."""
        return {(r, c)
                for r in range(self.env.rows)
                for c in range(self.env.cols)
                if self.is_position_safe((r, c))}

    # ------------------------------------------------------------------
    # 2. Victim priority
    # ------------------------------------------------------------------
    def _priority_ok(self, order):
        """priority_order constraint: no normal-priority victim may be
        placed before a critical-priority victim that is still waiting."""
        seen_normal = False
        for v in order:
            if self.env.priorities.get(v) == "critical":
                if seen_normal:
                    return False
            else:
                seen_normal = True
        return True

    # ------------------------------------------------------------------
    # 4. Time / energy conditions
    # ------------------------------------------------------------------
    def _energy_ok(self, agent, total_distance):
        """energy_limit constraint."""
        return total_distance <= agent.energy

    def _time_ok(self, agent, total_distance):
        """time_limit constraint (only active if max_time was set)."""
        if self.max_time is None:
            return True
        return agent.steps_taken + total_distance <= self.max_time

    def _cached_distance(self, start, goal, valid_cells):
        key = (start, goal)
        if key not in self._dist_cache:
            self._dist_cache[key] = _distance(self.env, start, goal, valid_cells)
        return self._dist_cache[key]

    # ------------------------------------------------------------------
    # Backtracking CSP solver: choose a feasible victim visiting order
    # ------------------------------------------------------------------
    def solve_victim_order(self, agent, valid_cells):
        """
        Backtracking search over victim visiting orders.

        At every partial assignment the priority, energy and time
        constraints are checked immediately, so an invalid branch is
        pruned as soon as it is built ("If a partial plan violates a
        constraint, the agent backtracks and tries another assignment" --
        proposal section 7).

        Within each step, candidate victims are tried nearest-first (a
        simple heuristic) so that when several orders are constraint-valid,
        backtracking finds a short, sensible one quickly instead of an
        arbitrary one -- it does not change which orders are legal, only
        the order branches are explored in.

        The exit is added as the final "stop" after all victims so the
        energy/time check reflects the WHOLE remaining mission (valid_exit
        constraint), not just the next victim.

        Returns (order, total_distance):
            order          - list of victim positions in visiting order,
                             or None if no order satisfies every constraint
            total_distance - moves needed for order + final leg to the exit
        """
        victims = list(self.env.victims)
        self._dist_cache = {}

        def backtrack(remaining, current_pos, order, distance_so_far):
            if not remaining:
                # All victims placed: close the plan with the leg to the exit
                # (valid_exit constraint).
                leg = self._cached_distance(current_pos, self.env.exit_pos, valid_cells)
                if leg is None:
                    return None, None
                total = distance_so_far + leg
                if not self._energy_ok(agent, total) or not self._time_ok(agent, total):
                    return None, None
                return order, total

            # Nearest-first heuristic ordering (unreachable victims sort last).
            candidates = sorted(
                remaining,
                key=lambda v: self._cached_distance(current_pos, v, valid_cells)
                if self._cached_distance(current_pos, v, valid_cells) is not None
                else float("inf"))

            for v in candidates:
                candidate_order = order + [v]
                if not self._priority_ok(candidate_order):
                    continue  # priority_order constraint failed -> prune

                leg = self._cached_distance(current_pos, v, valid_cells)
                if leg is None:
                    continue  # unreachable -> prune (safety/domain constraint)

                new_distance = distance_so_far + leg
                if not self._energy_ok(agent, new_distance):
                    continue  # energy_limit constraint failed -> backtrack
                if not self._time_ok(agent, new_distance):
                    continue  # time_limit constraint failed -> backtrack

                new_remaining = [x for x in remaining if x != v]
                result_order, result_total = backtrack(
                    new_remaining, v, candidate_order, new_distance)
                if result_order is not None:
                    return result_order, result_total
                # else: this branch could not be completed -> backtrack and
                # try the next candidate victim (the loop IS the backtrack step)

            return None, None

        return backtrack(victims, agent.position, [], 0)

    # ------------------------------------------------------------------
    # 5. Re-planning logic (proposal section 18.2 / Table 21):
    # "Energy left? Time left? Priority still correct?"
    # ------------------------------------------------------------------
    def recheck(self, agent, valid_cells):
        """Called by the agent whenever it needs to re-plan (its next cell
        just became blocked, e.g. new fire/debris/water). Re-verifies every
        constraint against the agent's CURRENT state. Returns (ok, reason);
        ok=False means the CSP has proved the mission can no longer be
        completed from here."""
        if agent.energy <= 0:
            return False, "Out of energy"
        if self.max_time is not None and agent.steps_taken >= self.max_time:
            return False, "Time limit exceeded"
        if self.env.victims:
            order, _ = self.solve_victim_order(agent, valid_cells)
            if order is None:
                return False, "No victim order satisfies energy/time/priority constraints"
        else:
            if self._cached_distance(agent.position, self.env.exit_pos, valid_cells) is None:
                return False, "No safe path to the exit"
        return True, ""

    # ------------------------------------------------------------------
    # Main entry point: "Apply CSP constraints" box in the workflow diagram.
    # Used for BOTH the first plan and every re-plan afterwards -- each
    # call re-derives valid_cells from the current grid and re-runs recheck(),
    # so it always reflects the agent's latest energy/position/hazards.
    # ------------------------------------------------------------------
    def apply(self, agent):
        """
        Runs the full constraint pipeline for the agent's current situation:
          1. restrict the domain to safe (non-blocked) cells
          2. decide a constraint-satisfying victim visiting order
          3. re-check energy / time / priority feasibility if none is found

        Returns a dict:
            valid_cells  - set of cells the search algorithm may use
            victim_order - feasible visiting order (empty list once all
                           victims are rescued)
            ok           - False if the CSP proved the mission is no
                           longer completable
            reason       - failure reason when ok is False
        """
        valid_cells = self.get_valid_cells()

        if not self.env.victims:
            ok, reason = self.recheck(agent, valid_cells)
            return {"valid_cells": valid_cells, "victim_order": [],
                    "ok": ok, "reason": reason}

        order, _ = self.solve_victim_order(agent, valid_cells)
        if order is None:
            ok, reason = self.recheck(agent, valid_cells)
            return {"valid_cells": valid_cells, "victim_order": [],
                    "ok": False, "reason": reason or "No feasible victim order"}

        return {"valid_cells": valid_cells, "victim_order": order,
                "ok": True, "reason": ""}

    # ------------------------------------------------------------------
    # Reporting helper for demos / write-ups: shows each constraint being
    # checked against the agent's current situation, in plain English.
    # ------------------------------------------------------------------
    def explain(self, agent):
        """Returns a human-readable, multi-line report of the constraint
        check for the agent's current state -- handy for a viva/demo to
        show the CSP is actually doing something, not a black box."""
        valid_cells = self.get_valid_cells()
        result = self.apply(agent)
        lines = ["CSP constraint check:"]
        for name, desc in CONSTRAINTS:
            lines.append(f"  - {name}: {desc}")
        lines.append(f"Usable cells      : {len(valid_cells)} / {self.env.rows * self.env.cols}")
        lines.append(f"Agent energy left : {agent.energy}")
        if self.max_time is not None:
            lines.append(f"Steps used/limit  : {agent.steps_taken}/{self.max_time}")
        if result["ok"]:
            if result["victim_order"]:
                lines.append(f"Feasible order    : {result['victim_order']}")
            else:
                lines.append("Feasible order    : (no victims left, heading to exit)")
        else:
            lines.append(f"INFEASIBLE        : {result['reason']}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Integration helper (kept OUTSIDE the Agent class on purpose): plugs the
# CSP into Member 1's Agent + Member 2's search functions without changing
# a single line of agent.py or search_algorithms.py.
# ----------------------------------------------------------------------
def plan_with_csp(agent, search_fn, csp):
    """
    Drop-in replacement for agent.plan(search_fn) that sources the goal and
    the set of usable cells from the CSP instead of Environment.victim_pos.
    Only uses agent's existing public interface (env, position, fail(),
    set_path(), total_nodes_explored, total_search_time), so Member 1's
    RescueAgent does not need any modification. Call this again every time
    the agent needs to re-plan -- csp.apply() re-runs the full constraint
    check (this IS the re-planning logic in Table 21/22).

    search_fn must be one of Member 2's functions (bfs/dfs/bidirectional_search
    from search_algorithms.py, or ALGORITHMS[name]) since they accept the
    extra valid_cells keyword; main.simple_bfs does not.
    """
    result = csp.apply(agent)
    if not result["ok"]:
        agent.fail(result["reason"])
        return False

    goal = result["victim_order"][0] if agent.env.victims else agent.env.exit_pos

    path, explored, seconds = search_fn(
        agent.env, agent.position, goal, valid_cells=result["valid_cells"])
    agent.total_nodes_explored += explored
    agent.total_search_time += seconds

    if path is None:
        agent.fail("No safe path to target.")
        return False

    agent.set_path(path)
    return True
