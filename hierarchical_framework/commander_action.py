"""
hierarchical_framework/commander_action.py

Purpose of this module:
- Define the role IDs used across the hierarchical framework.
- Define the small legal set of team role configurations.
- Provide a scripted commander baseline that picks role assignments
  for the blue team using simple game-state rules.

Why this file exists:
- Even though the commander is not trainable yet, we still want the
  "commander decision" logic to live in one clear place.
- Later, when you replace the scripted commander with a learned
  commander policy, you can keep the same role IDs and role configs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math


# Role IDs used everywhere else in the framework.
ATTACK = 0
DEFEND = 1
INTERCEPT = 2

ROLE_NAMES = {
    ATTACK: "ATTACK",
    DEFEND: "DEFEND",
    INTERCEPT: "INTERCEPT",
}

# Small legal set of role configurations for a 3-agent team.
# These are intentionally limited so the commander operates over a
# manageable tactical space instead of arbitrary assignments.
ROLE_CONFIGS: List[Tuple[int, int, int]] = [
    (ATTACK, DEFEND, INTERCEPT),    # 0 balanced default
    (ATTACK, ATTACK, DEFEND),       # 1 offense tilt
    (ATTACK, DEFEND, DEFEND),       # 2 defense tilt / escort return
    (ATTACK, INTERCEPT, INTERCEPT), # 3 recovery / chase
    (DEFEND, DEFEND, INTERCEPT),    # 4 heavy defense
    (ATTACK, ATTACK, INTERCEPT),    # 5 aggressive pressure
]


def euclidean_distance(a, b) -> float:
    """
    Compute Euclidean distance between two 2D positions.

    Why this method exists in this file:
    - The scripted commander often decides which worker should intercept
      or defend by checking distance to a carrier or key position.
    """
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


class ScriptedCommander:
    """
    Scripted commander baseline.

    Purpose of this class:
    - Look at environment state every role refresh interval.
    - Decide which blue worker should be ATTACK / DEFEND / INTERCEPT.
    - Return both:
        1) a role config index for logging/debugging, and
        2) explicit per-agent role assignments to apply in the wrapper.

    Why this class is useful now:
    - It lets you train the shared worker PPO under role-conditioned
      behavior before introducing a learned commander policy.
    """

    def __init__(self, blue_team: List[str], red_team: List[str]):
        """
        Store which agents belong to each team.

        Context:
        - The wrapper calls this commander for blue team role assignment.
        - Team membership is needed to inspect carriers, threats, and distances.
        """
        self.blue_team = list(blue_team)
        self.red_team = list(red_team)

    def _idx(self, agents: List[str], agent_id: str) -> int:
        """
        Convert an agent ID into its index in the environment's agent list.

        Why this method exists:
        - Most Pyquaticus state arrays are indexed by agent position in the
          master agent list, so we need a consistent lookup helper.
        """
        return agents.index(agent_id)

    def _find_flag_carrier(
        self,
        state: dict,
        agents: List[str],
        team_ids: List[str],
    ) -> Optional[str]:
        """
        Find which agent in a team is currently carrying the enemy flag.

        Why this method exists:
        - Carrier detection is central to role assignment:
          * if blue has the flag -> protect the return
          * if red has the flag -> chase and recover
        """
        for aid in team_ids:
            i = self._idx(agents, aid)
            if bool(state["agent_has_flag"][i]):
                return aid
        return None

    def _closest_blue_to_position(
        self,
        state: dict,
        agents: List[str],
        target_position,
        exclude: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Find the closest blue agent to a given position.

        Why this method exists:
        - In chase/recovery situations, the closest blue worker is often the
          best INTERCEPT candidate.
        """
        exclude = exclude or []
        best_agent = None
        best_distance = float("inf")

        for aid in self.blue_team:
            if aid in exclude:
                continue
            i = self._idx(agents, aid)
            d = euclidean_distance(state["agent_position"][i], target_position)
            if d < best_distance:
                best_distance = d
                best_agent = aid

        return best_agent

    def _count_red_pressure_on_blue_side(self, state: dict, agents: List[str]) -> int:
        """
        Count how many red agents are currently on the blue side.

        Why this method exists:
        - It gives the scripted commander a simple pressure signal.
        - If multiple red agents are pushing deep, the commander can shift
        to a more defensive formation.

        Important note:
        - Some branches expose agent_on_sides, some may not.
        - If that field is unavailable, we return 0 instead of crashing.
        """
        if "agent_on_sides" not in state:
            return 0

        pressure = 0
        for aid in self.red_team:
            i = self._idx(agents, aid)
            if int(state["agent_on_sides"][i]) == 0:
                pressure += 1
        return pressure

    def choose_config_index(self, state: dict, agents: List[str]) -> Tuple[int, str]:
        """
        Pick a role configuration index using a small rule tree.

        Returns:
        - config index into ROLE_CONFIGS
        - a human-readable reason string for debugging

        Why this method exists:
        - The wrapper wants a compact tactical mode.
        - The GUI/eval code can log why the scripted commander made a choice.

        Important implementation note:
        - We avoid depending on state["team_has_flag"] because some local
        Pyquaticus branches do not expose that key at reset.
        - Instead, we infer team-level possession from per-agent flag carriers.
        """
        blue_carrier = self._find_flag_carrier(state, agents, self.blue_team)
        red_carrier = self._find_flag_carrier(state, agents, self.red_team)

        blue_has_flag = blue_carrier is not None
        red_has_flag = red_carrier is not None

        # If blue has the enemy flag, protect the return.
        if blue_has_flag:
            return 2, "BLUE_HAS_FLAG"

        # If red has blue's flag, emphasize interception/recovery.
        if red_has_flag:
            return 3, "RED_HAS_FLAG"

        # If red pressure is high on blue side, lean defensive.
        red_pressure = self._count_red_pressure_on_blue_side(state, agents)
        if red_pressure >= 2:
            return 4, "RED_PRESSURE"

        # Otherwise remain in a stable balanced formation.
        return 0, "BALANCED_DEFAULT"

    def assign_roles(self, state: dict, agents: List[str]) -> Tuple[Dict[str, int], int, str]:
        """
        Produce explicit blue-team role assignments.

        Returns:
        - assignments: dict mapping blue worker id -> role id
        - config index: which ROLE_CONFIGS entry was chosen
        - reason: string explaining the tactical mode

        Why this method exists:
        - A config like (ATTACK, DEFEND, INTERCEPT) is not enough by itself;
          we still need to decide which specific blue agent gets which role.
        """
        config_index, reason = self.choose_config_index(state, agents)
        config = ROLE_CONFIGS[config_index]

        # Start with a simple fixed ordering.
        assignments = {
            self.blue_team[0]: config[0],
            self.blue_team[1]: config[1],
            self.blue_team[2]: config[2],
        }

        # If blue is carrying the flag, the carrier should be ATTACK so the
        # shared worker learns "ATTACK while carrying = finish the return".
        blue_carrier = self._find_flag_carrier(state, agents, self.blue_team)
        if blue_carrier is not None:
            assignments[blue_carrier] = ATTACK
            others = [aid for aid in self.blue_team if aid != blue_carrier]
            if len(others) == 2:
                assignments[others[0]] = DEFEND
                assignments[others[1]] = INTERCEPT
            return assignments, config_index, reason

        # If red is carrying blue's flag, make the closest blue worker the interceptor.
        red_carrier = self._find_flag_carrier(state, agents, self.red_team)
        if red_carrier is not None:
            red_i = self._idx(agents, red_carrier)
            red_pos = state["agent_position"][red_i]

            closest = self._closest_blue_to_position(state, agents, red_pos)
            if closest is not None:
                assignments[closest] = INTERCEPT
                remaining = [aid for aid in self.blue_team if aid != closest]
                if len(remaining) == 2:
                    assignments[remaining[0]] = DEFEND
                    assignments[remaining[1]] = ATTACK
            return assignments, config_index, reason

        return assignments, config_index, reason