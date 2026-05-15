"""
Generates mpl_animator_showcase.gif — a 3x2 grid of animated plots
for the mpl-animator LinkedIn launch post.
"""

import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from PIL import Image

FRAMES = 72
FPS = 20
DPI = 62
OUTPUT = "mpl_animator_showcase.gif"
BG = '#FFFFFF'
ACCENT = '#0969DA'
GRID_C = '#E8EBEF'
EDGE_C = '#D0D7DE'
LABEL_C = '#57606A'
TEXT_C = '#24292F'
FOOTER_C = '#8C959F'

PALETTE = ['#E03E3E', '#0891B2', '#0369A1', '#15803D', '#7C3AED', '#B45309']

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': BG,
    'axes.edgecolor': EDGE_C,
    'axes.labelcolor': LABEL_C,
    'xtick.color': LABEL_C,
    'ytick.color': LABEL_C,
    'grid.color': GRID_C,
    'grid.linewidth': 0.5,
    'text.color': TEXT_C,
    'font.size': 9,
})


def make_frame(frame_idx: int) -> Image.Image:
    t = 2 * np.pi * frame_idx / FRAMES  # 0 → 2π per loop

    fig = plt.figure(figsize=(14, 9.5), facecolor=BG)

    # "mpl-animator" sits in the horizontal gap between the two rows.
    # With top=0.96, bottom=0.04 the gap centre is exactly (0.96+0.04)/2 = 0.50.
    fig.text(0.5, 0.505, 'mpl-animator',
             ha='center', va='center', fontsize=26, fontweight='bold',
             color=ACCENT, fontfamily='monospace')
    fig.text(0.5, 0.468,
             'pip install mpl-animator  ·  Zero-boilerplate matplotlib → animated GIF/MP4',
             ha='center', va='center', fontsize=10, color=LABEL_C)

    gs = GridSpec(2, 3, figure=fig,
                  hspace=0.62, wspace=0.30,
                  top=0.96, bottom=0.04, left=0.06, right=0.97)

    # ── 1: Wave Harmonics (top-left) ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.linspace(0, 4 * np.pi, 500)
    n_harm = 1 + 4 * (0.5 + 0.5 * np.sin(t * 0.7))
    y = np.zeros_like(x)
    for k in range(1, 6):
        amp = float(np.clip(n_harm - k + 1, 0, 1))
        y += amp / k * np.sin(k * x + t * k * 0.25)
    y /= 2.5
    ax1.fill_between(x, y, alpha=0.20, color=PALETTE[0])
    ax1.plot(x, y, color=PALETTE[0], linewidth=2.0)
    ax1.set_xlim(0, 4 * np.pi)
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_title('Wave Harmonics', fontsize=11, pad=7, color=TEXT_C)
    ax1.grid(True)
    ax1.set_xticks([])

    # ── 2: 3-D Ripple Surface (top-middle) ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.set_facecolor(BG)
    lin3 = np.linspace(-3, 3, 42)
    X2, Y2 = np.meshgrid(lin3, lin3)
    R2 = np.sqrt(X2**2 + Y2**2)
    Z2 = np.sin(R2 - t * 1.5) * np.exp(-R2 * 0.22)
    ax2.plot_surface(X2, Y2, Z2, cmap='plasma', alpha=0.92,
                     linewidth=0, antialiased=True, rcount=42, ccount=42)
    ax2.set_zlim(-1, 1)
    ax2.set_title('3D Ripple Surface', fontsize=11, pad=5, color=TEXT_C)
    ax2.view_init(elev=28, azim=30 + 50 * np.sin(t * 0.45))
    ax2.tick_params(labelsize=6)
    for pane in (ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(EDGE_C)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])

    # ── 3: Polar Rose (top-right) ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2], projection='polar')
    ax3.set_facecolor(BG)
    theta_p = np.linspace(0, 8 * np.pi, 3000)
    k = 2 + 3 * (0.5 + 0.5 * np.sin(t * 0.55))
    r_rose = np.abs(np.cos(k * theta_p))
    ax3.fill(theta_p, r_rose, alpha=0.22, color=PALETTE[2])
    ax3.plot(theta_p, r_rose, color=PALETTE[2], linewidth=1.3, alpha=0.9)
    ax3.set_ylim(0, 1.05)
    ax3.set_title('Polar Rose', fontsize=11, pad=10, color=TEXT_C)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['polar'].set_color(EDGE_C)
    ax3.grid(color=GRID_C, linewidth=0.5)

    # ── 4: Lissajous Figure (bottom-left) ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    s = np.linspace(0, 2 * np.pi, 1200)
    b_ratio = 2 + 0.8 * np.sin(t * 0.5)
    lx = np.sin(3 * s + t)
    ly = np.sin(b_ratio * s)
    step = 6
    seg_idx = np.arange(0, len(s) - step, step)
    cmap_liss = plt.cm.cool
    for j, i in enumerate(seg_idx):
        frac = j / len(seg_idx)
        ax4.plot(lx[i:i+step+1], ly[i:i+step+1],
                 color=cmap_liss(frac), linewidth=1.8,
                 alpha=0.4 + 0.6 * frac, solid_capstyle='round')
    ax4.set_xlim(-1.3, 1.3)
    ax4.set_ylim(-1.3, 1.3)
    ax4.set_aspect('equal', adjustable='box')
    ax4.set_title('Lissajous Figure', fontsize=11, pad=7, color=TEXT_C)
    ax4.grid(True)
    ax4.set_xticks([])
    ax4.set_yticks([])

    # ── 5: Spiral Galaxy (bottom-middle) ─────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(42)
    n_stars = 220
    phi_arm = np.linspace(0, 3 * np.pi, n_stars)
    r_arm = 0.08 + 0.75 * phi_arm / (3 * np.pi)
    for arm_offset, cmap_name in [(0.0, 'cool'), (np.pi, 'autumn')]:
        angles = phi_arm + arm_offset + t
        xs = r_arm * np.cos(angles) + rng.normal(0, 0.04, n_stars)
        ys = r_arm * np.sin(angles) + rng.normal(0, 0.04, n_stars)
        sz = 3 + 28 * rng.random(n_stars)
        cols = plt.get_cmap(cmap_name)(np.linspace(0.1, 0.9, n_stars))
        ax5.scatter(xs, ys, s=sz, c=cols, alpha=0.78, linewidths=0)
    ax5.set_xlim(-1.1, 1.1)
    ax5.set_ylim(-1.1, 1.1)
    ax5.set_aspect('equal', adjustable='box')
    ax5.set_title('Spiral Galaxy', fontsize=11, pad=7, color=TEXT_C)
    ax5.set_xticks([])
    ax5.set_yticks([])

    # ── 6: Wave Interference (bottom-right) ──────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    xi = np.linspace(-3, 3, 110)
    yi = np.linspace(-3, 3, 110)
    Xi, Yi = np.meshgrid(xi, yi)
    d1 = np.sqrt((Xi - 1.0)**2 + Yi**2)
    d2 = np.sqrt((Xi + 1.0)**2 + Yi**2)
    Zi = np.sin(2.6 * d1 - t * 1.2) + np.sin(2.6 * d2 - t * 0.85)
    ax6.contourf(Xi, Yi, Zi, levels=24, cmap='RdBu_r', alpha=0.93)
    ax6.contour(Xi, Yi, Zi, levels=8, colors=LABEL_C, alpha=0.35, linewidths=0.4)
    ax6.set_aspect('equal', adjustable='box')
    ax6.set_title('Wave Interference', fontsize=11, pad=7, color=TEXT_C)
    ax6.set_xticks([])
    ax6.set_yticks([])

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor=BG,
                bbox_inches='tight', pad_inches=0.05)
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    buf.close()
    return img


def main():
    print(f"Generating {FRAMES}-frame showcase GIF at {FPS}fps …")
    frames = []
    for i in range(FRAMES):
        print(f"  frame {i+1:>3}/{FRAMES}", end='\r', flush=True)
        frames.append(make_frame(i))

    print(f"\nQuantizing colours …")
    # Build a global palette from a mid-animation frame so all frames share it.
    # Shared palette + no dithering → longer pixel runs → better LZW compression.
    palette_src = frames[len(frames) // 2].convert('RGB').quantize(colors=128)
    frames_q = [
        f.convert('RGB').quantize(palette=palette_src, dither=Image.Dither.NONE)
        for f in frames
    ]

    print(f"Saving → {OUTPUT}")
    ms_per_frame = 1000 // FPS
    frames_q[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames_q[1:],
        optimize=True,
        duration=ms_per_frame,
        loop=0,
    )
    import os
    size_mb = os.path.getsize(OUTPUT) / 1_048_576
    w, h = frames[0].size
    print(f"Done!  {w}×{h}px  {FRAMES} frames  {ms_per_frame}ms/frame  {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
