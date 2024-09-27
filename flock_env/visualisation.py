import chex
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def animate_agents(
    positions: chex.Array,
    headings: chex.Array,
    rewards: chex.Array,
    cmap: str = "winter",
    border: float = 0.01,
) -> animation.FuncAnimation:

    # Scale rewards to use as colours for the plot
    rewards = 255 * (rewards - rewards.min()) / (rewards.max() - rewards.min())

    d = np.hstack(
        [
            jnp.swapaxes(positions, 1, 2),
            headings[:, jnp.newaxis],
            rewards[:, jnp.newaxis],
        ]
    )

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.subplots_adjust(
        top=1.0 - border,
        bottom=border,
        right=1.0 - border,
        left=border,
        hspace=0,
        wspace=0,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    q = ax.quiver(
        d[0][0],
        d[0][1],
        jnp.cos(d[0][2]),
        jnp.sin(d[0][2]),
        d[0][3],
        cmap=plt.get_cmap(cmap),
        pivot="middle",
    )

    def update_quiver(f):
        """Updates the values of the quiver plot"""
        q.set_offsets(f[:2].T)
        q.set_UVC(np.cos(f[2]), np.sin(f[2]), f[3])
        return (q,)

    anim = animation.FuncAnimation(
        fig, update_quiver, frames=d[1:], interval=50, blit=False
    )

    return anim
