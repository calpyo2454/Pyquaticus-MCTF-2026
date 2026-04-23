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

    def __init__(self, my_team: List[str], enemy_team: List[str], home_side_idx: int):
        """
        Generic scripted commander for one team.

        my_team:
        - the team this commander assigns roles for

        enemy_team:
        - the opposing team

        home_side_idx:
        - which side index in agent_on_sides corresponds to this team's home side
        (0 for blue, 1 for red)
        """
        self.my_team = list(my_team)
        self.enemy_team = list(enemy_team)
        self.home_side_idx = int(home_side_idx)

        self.decision_count = 0
        self.default_config_index = 0  # (ATTACK, DEFEND, INTERCEPT)
        self.last_config_index = self.default_config_index
        self.last_reason = "BALANCED_DEFAULT"
        self.min_event_dwell = 3
        self.remaining_event_dwell = 0

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

        for aid in self.my_team:
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

        Why this method exists:_closest_blue_to_position
        - Quiet-state role assignment should pick roles based on geometry instead
        of fixed agent ordering whenever possible.
        """
        scored = []
        for aid in self.my_team:
            i = self._idx(agents, aid)
            pos = state["agent_position"][i]
            d = euclidean_distance(pos, target_pos)
            scored.append((d, aid))
        scored.sort(key=lambda x: x[0])
        return [aid for _, aid in scored]

    def _count_enemy_pressure_on_home_side(self, state: dict, agents: List[str]) -> int:
        """
        Count how many enemy agents are currently on this commander's home side.
        """
        if "agent_on_sides" not in state:
            return 0

        pressure = 0
        for aid in self.enemy_team:
            i = self._idx(agents, aid)
            try:
                if int(state["agent_on_sides"][i]) == self.home_side_idx:
                    pressure += 1
            except Exception:
                pass
        return pressure

    def choose_config_index(self, state: dict, agents: List[str]) -> Tuple[int, str]:
        """
        Event-driven formation selection for whichever team this commander controls.
        """
        self.decision_count += 1

        my_carrier = self._find_flag_carrier(state, agents, self.my_team)
        enemy_carrier = self._find_flag_carrier(state, agents, self.enemy_team)

        my_has_flag = my_carrier is not None
        enemy_has_flag = enemy_carrier is not None

        desired_config = self.default_config_index
        desired_reason = "BALANCED_DEFAULT"

        if my_has_flag:
            desired_config = 2
            desired_reason = "MY_TEAM_HAS_FLAG"
        elif enemy_has_flag:
            desired_config = 3
            desired_reason = "ENEMY_HAS_FLAG"
        else:
            enemy_pressure = self._count_enemy_pressure_on_home_side(state, agents)
            if enemy_pressure >= 2:
                desired_config = 4
                desired_reason = "ENEMY_PRESSURE"

        if desired_config == self.last_config_index:
            if self.remaining_event_dwell > 0:
                self.remaining_event_dwell -= 1
            self.last_reason = desired_reason
            return desired_config, desired_reason

        if desired_reason in {"MY_TEAM_HAS_FLAG", "ENEMY_HAS_FLAG", "ENEMY_PRESSURE"}:
            self.last_config_index = desired_config
            self.last_reason = desired_reason
            self.remaining_event_dwell = self.min_event_dwell - 1
            return desired_config, desired_reason

        if self.remaining_event_dwell > 0:
            self.remaining_event_dwell -= 1
            return self.last_config_index, self.last_reason

        self.last_config_index = desired_config
        self.last_reason = desired_reason
        return desired_config, desired_reason

    def assign_roles(self, state: dict, agents: List[str]) -> Tuple[Dict[str, int], int, str]:
        """
        Assign ATTACK / DEFEND / INTERCEPT roles to this commander's team.
        """
        config_index, reason = self.choose_config_index(state, agents)
        config = ROLE_CONFIGS[config_index]

        # If my team has carrier, carrier becomes ATTACK.
        my_carrier = self._find_flag_carrier(state, agents, self.my_team)
        if my_carrier is not None:
            assignments = {aid: DEFEND for aid in self.my_team}
            assignments[my_carrier] = ATTACK

            others = [aid for aid in self.my_team if aid != my_carrier]
            if len(others) == 2:
                assignments[others[0]] = DEFEND
                assignments[others[1]] = INTERCEPT

            return assignments, config_index, reason

        # If enemy has carrier, nearest teammate becomes INTERCEPT.
        enemy_carrier = self._find_flag_carrier(state, agents, self.enemy_team)
        if enemy_carrier is not None:
            enemy_i = self._idx(agents, enemy_carrier)
            enemy_pos = state["agent_position"][enemy_i]

            closest = None
            best_distance = float("inf")
            for aid in self.my_team:
                i = self._idx(agents, aid)
                d = euclidean_distance(state["agent_position"][i], enemy_pos)
                if d < best_distance:
                    best_distance = d
                    closest = aid

            assignments = {aid: ATTACK for aid in self.my_team}
            if closest is not None:
                assignments[closest] = INTERCEPT
                remaining = [aid for aid in self.my_team if aid != closest]
                if len(remaining) == 2:
                    assignments[remaining[0]] = DEFEND
                    assignments[remaining[1]] = ATTACK

            return assignments, config_index, reason

        # Quiet state -> assign by geometry to home/enemy flags if possible.
        own_flag_pos = self._get_team_flag_position(state, self.home_side_idx)
        enemy_flag_pos = self._get_team_flag_position(state, 1 - self.home_side_idx)

        if own_flag_pos is None or enemy_flag_pos is None:
            assignments = {
                self.my_team[0]: config[0],
                self.my_team[1]: config[1],
                self.my_team[2]: config[2],
            }
            return assignments, config_index, reason

        by_enemy_flag = self._sorted_blue_by_distance_to(state, agents, enemy_flag_pos)
        by_own_flag = self._sorted_blue_by_distance_to(state, agents, own_flag_pos)

        remaining = list(self.my_team)
        assignments: Dict[str, int] = {}

        attack_slots = sum(1 for r in config if r == ATTACK)
        defend_slots = sum(1 for r in config if r == DEFEND)
        intercept_slots = sum(1 for r in config if r == INTERCEPT)

        for aid in by_enemy_flag:
            if attack_slots <= 0:
                break
            if aid in remaining:
                assignments[aid] = ATTACK
                remaining.remove(aid)
                attack_slots -= 1

        for aid in by_own_flag:
            if defend_slots <= 0:
                break
            if aid in remaining:
                assignments[aid] = DEFEND
                remaining.remove(aid)
                defend_slots -= 1

        for aid in list(remaining):
            if intercept_slots > 0:
                assignments[aid] = INTERCEPT
                remaining.remove(aid)
                intercept_slots -= 1

        for aid in remaining:
            assignments[aid] = ATTACK

        return assignments, config_index, reason