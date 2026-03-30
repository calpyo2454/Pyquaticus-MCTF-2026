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
        Store team membership and initialize a simple formation cycle.

        Why this method exists:
        - The scripted commander should not stay in one formation forever when
        the game state is quiet.
        - Cycling formations is a deliberate baseline trick that helps verify
        the shared worker actually responds to role changes.
        """
        self.blue_team = list(blue_team)
        self.red_team = list(red_team)

        # Counts how many commander decisions have been made.
        self.decision_count = 0

        # Cycle through a few legal formations when no high-priority event is present.
        self.default_cycle = [0, 1, 2, 5]  # balanced, offense, defense, aggressive pressure
        self.default_cycle_reason = {
            0: "BALANCED_DEFAULT",
            1: "OFFENSE_CYCLE",
            2: "DEFENSE_CYCLE",
            5: "PRESSURE_CYCLE",
        }

        # Hold each default formation for a few commander refreshes before switching.
        # With role_period=5 and default_hold=3, each formation lasts 15 env steps.
        self.default_hold = 3

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

    def _get_team_flag_position(self, state: dict, team_idx: int):
        """
        Return a team's flag position, or None if unavailable.

        Why this method exists:
        - Quiet-state role assignment should use geometry:
        * attacker closer to enemy flag
        * defender closer to own flag
        """
        if "flag_position" not in state:
            return None
        try:
            pos = state["flag_position"][team_idx]
            return (float(pos[0]), float(pos[1]))
        except Exception:
            return None


    def _sorted_blue_by_distance_to(self, state: dict, agents: List[str], target_pos) -> List[str]:
        """
        Return blue agents sorted by distance to a target position.

        Why this method exists:
        - Quiet-state role assignment should pick roles based on geometry instead
        of fixed agent ordering whenever possible.
        """
        scored = []
        for aid in self.blue_team:
            i = self._idx(agents, aid)
            pos = state["agent_position"][i]
            d = euclidean_distance(pos, target_pos)
            scored.append((d, aid))
        scored.sort(key=lambda x: x[0])
        return [aid for _, aid in scored]

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

        Priority:
        1) If blue has the enemy flag -> defense tilt / escort return
        2) If red has blue's flag -> chase / recovery
        3) If red pressure is high on blue side -> heavy defense
        4) Otherwise cycle through several safe default formations

        Why this method exists:
        - The previous version stayed in BALANCED_DEFAULT too often.
        - This version intentionally varies the commander output even when no
        flag event has happened yet, which is useful for testing whether the
        shared worker obeys role conditioning.
        """
        self.decision_count += 1

        blue_carrier = self._find_flag_carrier(state, agents, self.blue_team)
        red_carrier = self._find_flag_carrier(state, agents, self.red_team)

        blue_has_flag = blue_carrier is not None
        red_has_flag = red_carrier is not None

        # High-priority event: blue is returning with flag.
        if blue_has_flag:
            return 2, "BLUE_HAS_FLAG"

        # High-priority event: red is carrying blue's flag.
        if red_has_flag:
            return 3, "RED_HAS_FLAG"

        # Secondary event: red pressure near blue side.
        red_pressure = self._count_red_pressure_on_blue_side(state, agents)
        if red_pressure >= 2:
            return 4, "RED_PRESSURE"

        # Quiet-state fallback: rotate among several formations.
        cycle_index = ((self.decision_count - 1) // self.default_hold) % len(self.default_cycle)
        config_index = self.default_cycle[cycle_index]
        reason = self.default_cycle_reason[config_index]
        return config_index, reason

    def assign_roles(self, state: dict, agents: List[str]) -> Tuple[Dict[str, int], int, str]:
        """
        Produce explicit blue-team role assignments.

        Returns:
        - assignments: dict mapping blue worker id -> role id
        - config index: which ROLE_CONFIGS entry was chosen
        - reason: string explaining the tactical mode

        Why this method exists:
        - The commander chooses a formation first.
        - Then this method decides which specific blue worker gets which role.
        - In quiet states, we rotate role ownership among agents so evaluation
        can confirm the shared worker reacts to role changes.
        """
        config_index, reason = self.choose_config_index(state, agents)
        config = ROLE_CONFIGS[config_index]

        # Case 1: blue carrier should always be ATTACK.
        blue_carrier = self._find_flag_carrier(state, agents, self.blue_team)
        if blue_carrier is not None:
            assignments = {aid: DEFEND for aid in self.blue_team}
            assignments[blue_carrier] = ATTACK

            others = [aid for aid in self.blue_team if aid != blue_carrier]
            if len(others) == 2:
                assignments[others[0]] = DEFEND
                assignments[others[1]] = INTERCEPT

            return assignments, config_index, reason

        # Case 2: if red has blue's flag, nearest blue becomes INTERCEPT.
        red_carrier = self._find_flag_carrier(state, agents, self.red_team)
        if red_carrier is not None:
            red_i = self._idx(agents, red_carrier)
            red_pos = state["agent_position"][red_i]

            closest = self._closest_blue_to_position(state, agents, red_pos)
            assignments = {aid: ATTACK for aid in self.blue_team}

            if closest is not None:
                assignments[closest] = INTERCEPT
                remaining = [aid for aid in self.blue_team if aid != closest]
                if len(remaining) == 2:
                    assignments[remaining[0]] = DEFEND
                    assignments[remaining[1]] = ATTACK

            return assignments, config_index, reason

        # Case 3: quiet state -> assign roles using geometry first, then fall back to rotation.
        own_flag_pos = self._get_team_flag_position(state, 0)   # blue flag
        enemy_flag_pos = self._get_team_flag_position(state, 1) # red flag

        # If flag positions are unavailable, keep the old rotation fallback.
        if own_flag_pos is None or enemy_flag_pos is None:
            rotation = (self.decision_count - 1) % len(self.blue_team)
            rotated_blue = self.blue_team[rotation:] + self.blue_team[:rotation]
            assignments = {
                rotated_blue[0]: config[0],
                rotated_blue[1]: config[1],
                rotated_blue[2]: config[2],
            }
            return assignments, config_index, reason

        # Sort blue agents by tactical relevance.
        by_enemy_flag = self._sorted_blue_by_distance_to(state, agents, enemy_flag_pos)
        by_own_flag = self._sorted_blue_by_distance_to(state, agents, own_flag_pos)

        remaining = list(self.blue_team)
        assignments: Dict[str, int] = {}

        attack_slots = sum(1 for r in config if r == ATTACK)
        defend_slots = sum(1 for r in config if r == DEFEND)
        intercept_slots = sum(1 for r in config if r == INTERCEPT)

        # Assign ATTACK roles to those closest to enemy flag.
        for aid in by_enemy_flag:
            if attack_slots <= 0:
                break
            if aid in remaining:
                assignments[aid] = ATTACK
                remaining.remove(aid)
                attack_slots -= 1

        # Assign DEFEND roles to those closest to own flag from remaining workers.
        for aid in by_own_flag:
            if defend_slots <= 0:
                break
            if aid in remaining:
                assignments[aid] = DEFEND
                remaining.remove(aid)
                defend_slots -= 1

        # Any remaining workers become INTERCEPT.
        for aid in list(remaining):
            if intercept_slots > 0:
                assignments[aid] = INTERCEPT
                remaining.remove(aid)
                intercept_slots -= 1

        # Safety fallback: if any workers remain for any reason, give them ATTACK.
        for aid in remaining:
            assignments[aid] = ATTACK

        return assignments, config_index, reason