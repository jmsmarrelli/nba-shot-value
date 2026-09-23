"""Half-court drawing in the same units as the shot data.

The shot files put the rim at (0, 5.25) and measure in feet, x from -25 to 25
(sideline to sideline) and y from 0 (baseline) up the floor. Everything here is
drawn in those units so shot coordinates can be plotted without a transform.
"""

import numpy as np
from matplotlib.patches import Arc, Circle, Rectangle

RIM_Y = 5.25
RIM_R = 0.75
THREE_R = 23.75          # arc radius from the centre of the rim
CORNER_X = 22.0          # corner three distance from centre line
COURT_HALF_W = 25.0

# y at which the arc meets the corner-three line
CORNER_Y = RIM_Y + np.sqrt(THREE_R**2 - CORNER_X**2)


def draw(ax, color="#8a8880", lw=1.0, ymax=40.0):
    """Draw the offensive half court onto ax and fix the limits."""
    ax.add_patch(Rectangle((-COURT_HALF_W, 0), 2 * COURT_HALF_W, ymax,
                           fill=False, ec=color, lw=lw))
    # Backboard and rim
    ax.plot([-3, 3], [4.0, 4.0], color=color, lw=lw * 1.4, solid_capstyle="butt")
    ax.add_patch(Circle((0, RIM_Y), RIM_R, fill=False, ec=color, lw=lw * 1.4))
    # Paint (16 ft wide, 19 ft deep) and free-throw circle
    ax.add_patch(Rectangle((-8, 0), 16, 19, fill=False, ec=color, lw=lw))
    ax.add_patch(Rectangle((-6, 0), 12, 19, fill=False, ec=color, lw=lw))
    ax.add_patch(Arc((0, 19), 12, 12, theta1=0, theta2=180, ec=color, lw=lw))
    ax.add_patch(Arc((0, 19), 12, 12, theta1=180, theta2=360, ec=color, lw=lw,
                     linestyle=(0, (3, 3))))
    # Restricted area
    ax.add_patch(Arc((0, RIM_Y), 8, 8, theta1=0, theta2=180, ec=color, lw=lw))
    # Three-point line
    ax.plot([-CORNER_X, -CORNER_X], [0, CORNER_Y], color=color, lw=lw)
    ax.plot([CORNER_X, CORNER_X], [0, CORNER_Y], color=color, lw=lw)
    theta = np.degrees(np.arcsin(CORNER_X / THREE_R))
    ax.add_patch(Arc((0, RIM_Y), 2 * THREE_R, 2 * THREE_R,
                     theta1=theta, theta2=180 - theta, ec=color, lw=lw))

    ax.set_xlim(-COURT_HALF_W - 0.6, COURT_HALF_W + 0.6)
    ax.set_ylim(-0.6, ymax + 0.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax
