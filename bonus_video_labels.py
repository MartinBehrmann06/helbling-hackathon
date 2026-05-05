"""
Labels for the bonus videos, extracted from:
  bonus_videos/20250225_Example Videos labeled SLU.xlsx

Severity scale: 0 = present (unrated), 1 = light, 2 = medium, 3 = heavy.
Timestamps are MM:SS into the video.
"""

from __future__ import annotations

# channel -> list of timestamped label dicts
ANNOTATIONS: dict[int, list[dict]] = {
    1: [
        {'time': '0:37', 'Lime deposit': 1},
        {'time': '2:02', 'No Vision (Splash Water)': 0},
        {'time': '2:48', 'No Vision (Lens covered)': 0},
        {'time': '16:20', 'Gravel': 1},
        {'time': '16:44', 'Gravel': 1},
        {'time': '23:28', 'Lime deposit': 2},
    ],
    2: [
        {'time': '2:19', 'Lime deposit': 2, 'comment': 'Under water'},
    ],
    3: [
        {'time': '2:02', 'No Obstruction': 0},
        {'time': '2:26', 'No Obstruction': 0},
    ],
    4: [
        {'time': '1:06', 'Sludge': 1},
        {'time': '3:55', 'No Vision (Lens covered)': 0},
        {'time': '8:09', 'No Obstruction': 0},
    ],
    5: [
        {'time': '1:17', 'No Obstruction': 0},
        {'time': '1:51', 'Lime deposit': 1},
        {'time': '2:14', 'Lime deposit': 1, 'Sand': 1},
        {'time': '2:23', 'Gravel': 1, 'Lime deposit': 1, 'Sand': 1},
        {'time': '3:20', 'No Vision (Lens covered)': 0},
        {'time': '10:43', 'No Obstruction': 0},
        {'time': '17:37', 'Gravel': 1},
        {'time': '18:00', 'Gravel': 1},
        {'time': '19:01', 'Lime deposit': 2},
    ],
    6: [
        {'time': '6:42', 'Gravel': 1},
    ],
    7: [
        {'time': '1:28', 'No Obstruction': 0},
        {'time': '2:01', 'Gravel': 1, 'Sludge': 1},
    ],
    8: [
        {'time': '3:14', 'Lime deposit': 1},
    ],
    9: [
        {'time': '5:21', 'Sludge': 2},
        {'time': '5:38', 'Sludge': 2},
        {'time': '7:35', 'Gravel': 2},
        {'time': '9:12', 'Gravel': 2},
        {'time': '14:57', 'No Obstruction': 0},
        {'time': '15:40', 'No Obstruction': 0, 'comment': 'Nice example cleaning quality'},
    ],
    10: [
        {'time': '1:22', 'Sludge': 1},
        {'time': '3:47', 'Sludge': 2},
    ],
    11: [
        {'time': '1:23', 'No Obstruction': 0},
        {'time': '2:11', 'No Vision (Splash Water)': 0, 'comment': 'Going to fast'},
        {'time': '2:13', 'No Obstruction': 0},
        {'time': '2:16', 'Lime deposit': 1},
    ],
    12: [
        {'time': '1:10', 'Roots': 1},
        {'time': '2:46', 'Grease': 2, 'Sludge': 2},
        {'time': '2:48', 'Roots': 1},
        {'time': '3:06', 'Roots': 1},
        {'time': '3:11', 'Sludge': 2},
        {'time': '3:37', 'Grease': 3, 'Sludge': 3, 'Roots': 2},
        {'time': '4:54', 'comment': 'Broken Pipe'},
    ],
    13: [
        {'time': '1:35', 'Gravel': 2},
        {'time': '2:44', 'Sludge': 1},
        {'time': '6:56', 'No Obstruction': 0},
        {'time': '7:32', 'No Vision (Under Water)': 0},
        {'time': '7:55', 'No Vision (Splash Water)': 0},
        {'time': '9:10', 'No Obstruction': 0},
    ],
    14: [
        {'time': '1:17', 'No Obstruction': 0, 'Sludge': 1},
        {'time': '1:38', 'No Vision (Lens covered)': 0},
        {'time': '1:45', 'Roots': 3, 'comment': 'Blockage 100%'},
        {'time': '2:08', 'Roots': 3, 'comment': 'Blockage 100%'},
        {'time': '2:45', 'No Obstruction': 0, 'Sludge': 1},
    ],
    15: [
        {'time': '0:51', 'Sludge': 1},
        {'time': '2:19', 'No Vision (Splash Water)': 0},
        {'time': '2:52', 'No Obstruction': 0},
        {'time': '3:15', 'No Obstruction': 0},
    ],
    16: [
        {'time': '2:59', 'Grease': 1, 'Sludge': 1},
        {'time': '3:26', 'No Vision (Lens covered)': 0},
    ],
    17: [
        {'time': '1:44', 'No Obstruction': 0},
        {'time': '6:11', 'Roots': 2},
    ],
    18: [
        {'time': '0:14', 'Roots': 2},
        {'time': '0:51', 'Roots': 1},
        {'time': '0:54', 'Roots': 2},
        {'time': '1:28', 'Roots': 1},
        {'time': '1:59', 'Roots': 2},
        {'time': '2:49', 'No Obstruction': 0},
        {'time': '2:55', 'No Obstruction': 0},
        {'time': '3:00', 'Roots': 1},
    ],
    19: [
        {'time': '0:16', 'Gravel': 1},
        {'time': '0:40', 'No Vision (Splash Water)': 0},
        {'time': '1:38', 'Gravel': 2},
        {'time': '2:18', 'No Obstruction': 0},
        {'time': '2:34', 'No Obstruction': 0},
        {'time': '5:13', 'No Obstruction': 0},
    ],
    20: [
        {'time': '0:46', 'No Vision (Under Water)': 0},
        {'time': '1:57', 'Sludge': 1},
        {'time': '2:24', 'Sludge': 1},
        {'time': '3:00', 'Concrete': 3, 'comment': 'Blockage 50%'},
    ],
}

# channel -> frozenset of obstruction types present (excluding "No Obstruction")
OBSTRUCTIONS: dict[int, frozenset[str]] = {
    1: frozenset(['Gravel', 'Lime deposit', 'No Vision (Lens covered)', 'No Vision (Splash Water)']),
    2: frozenset(['Lime deposit']),
    3: frozenset([]),
    4: frozenset(['No Vision (Lens covered)', 'Sludge']),
    5: frozenset(['Gravel', 'Lime deposit', 'No Vision (Lens covered)', 'Sand']),
    6: frozenset(['Gravel']),
    7: frozenset(['Gravel', 'Sludge']),
    8: frozenset(['Lime deposit']),
    9: frozenset(['Gravel', 'Sludge']),
    10: frozenset(['Sludge']),
    11: frozenset(['Lime deposit', 'No Vision (Splash Water)']),
    12: frozenset(['Grease', 'Roots', 'Sludge']),
    13: frozenset(['Gravel', 'No Vision (Splash Water)', 'No Vision (Under Water)', 'Sludge']),
    14: frozenset(['No Vision (Lens covered)', 'Roots', 'Sludge']),
    15: frozenset(['No Vision (Splash Water)', 'Sludge']),
    16: frozenset(['Grease', 'No Vision (Lens covered)', 'Sludge']),
    17: frozenset(['Roots']),
    18: frozenset(['Roots']),
    19: frozenset(['Gravel', 'No Vision (Splash Water)']),
    20: frozenset(['Concrete', 'No Vision (Under Water)', 'Sludge']),
}
