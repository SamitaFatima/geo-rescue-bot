"""
agent.py - RescueBot (Member 1)
Manages multi-victim searching, rescue sequences, and state transitions.
"""

SEARCHING = "SEARCHING"
RESCUING = "RESCUING"
DONE = "DONE"
FAILED = "FAILED"


class RescueAgent:
    def __init__(self, env, max_energy=100):
        self.env = env
        self.position = env.agent_start
        self.state = SEARCHING
        self.carrying_victims = 0
        self.total_victims_target = len(env.victims)
        self.path = []
        self.max_energy = max_energy
        self.energy = max_energy
        self.fail_reason = ""

        # Statistics
        self.steps_taken = 0
        self.replan_count = 0
        self.total_nodes_explored = 0
        self.total_search_time = 0.0
        self.victims_rescued = 0
        self.trail = [self.position]

    def perceive(self):
        return {
            "position": self.position,
            "victims": list(self.env.victims),
            "exit": self.env.exit_pos,
            "free_neighbors": self.env.get_valid_neighbors(self.position),
            "energy": self.energy,
            "victims_rescued": self.victims_rescued,
            "remaining_victims": len(self.env.victims)
        }

    def current_goal(self):
        """If unrescued victims remain, return target victim; else target exit."""
        if self.env.victims:
            return self.env.victim_pos
        return self.env.exit_pos

    def plan(self, search_fn):
        goal = self.current_goal()
        if goal is None:
            self.fail("No valid targets remaining.")
            return False

        path, explored, seconds = search_fn(self.env, self.position, goal)
        self.total_nodes_explored += explored
        self.total_search_time += seconds

        if path is None:
            self.fail("No safe path to target.")
            return False

        self.set_path(path)
        return True

    def set_path(self, path):
        path = list(path)
        if path and path[0] == self.position:
            path = path[1:]
        self.path = path

    def move(self, new_pos):
        self.position = new_pos
        self.energy -= 1
        self.steps_taken += 1
        self.trail.append(new_pos)

    def rescue(self):
        """Pick up victim if standing on one."""
        if self.position in self.env.victims:
            self.env.remove_victim(self.position)
            self.carrying_victims += 1
            self.victims_rescued += 1
            self.path = []  # Reset path to re-plan next victim or exit
            
            if self.env.victims:
                self.state = SEARCHING
            else:
                self.state = RESCUING
            return True
        return False

    def fail(self, reason):
        self.state = FAILED
        self.fail_reason = reason

    def step(self):
        if self.state in (DONE, FAILED):
            return self.state

        if self.energy <= 0:
            self.fail("Out of energy")
            return FAILED

        if not self.path:
            return "NEED_PLAN"

        nxt = self.path[0]
        if self.env.is_blocked(nxt):
            self.path = []
            self.replan_count += 1
            return "REPLAN"

        self.path.pop(0)
        self.move(nxt)

        if self.rescue():
            return "VICTIM_RESCUED"

        if len(self.env.victims) == 0 and self.position == self.env.exit_pos:
            self.state = DONE
            return DONE

        return "MOVED"