import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def rot(n, x, y, rx, ry):
    """Rotates and flips the quadrant appropriately."""
    if ry == 0:
        if rx == 1:
            x, y = n - 1 - x, n - 1 - y
        return y, x
    return x, y

def d2xy(n, d):
    """
    Decodes a 1D Hilbert index 'd' into (x, y) coordinates.
    n: side length of the square (2^bits)
    d: index along the curve
    """
    t = d
    x = y = 0
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y

# Configuration
bits = 5
side_length = 2**bits
n_points = side_length**2



points = np.array([d2xy(side_length, i) for i in range(n_points)])

segments = np.zeros((n_points - 1, 2, 2))
segments[:, 0, :] = points[:-1] # Start points
segments[:, 1, :] = points[1:]  # End points

colors = np.linspace(0, 1, n_points)

fig, ax = plt.subplots(figsize=(10, 10))

x_coords, y_coords = points[:, 0], points[:, 1]
ax.scatter(x_coords, y_coords, c=colors, cmap="plasma", s=10, zorder=2)
lc = LineCollection(segments, cmap="plasma", array=colors, linewidths=2)
line = ax.add_collection(lc)

ax.set_xlim(-0.5, side_length - 0.5)
ax.set_ylim(-0.5, side_length - 0.5)
ax.set_aspect("equal")
ax.set_title("2D Order 5 Hilbert Curve")
ax.set_xlabel("X Dimension")
ax.set_ylabel("Y Dimension")
ax.grid(True, linestyle=':', alpha=0.5)

cbar = fig.colorbar(line, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Normalized Hilbert Index")

plt.show()
