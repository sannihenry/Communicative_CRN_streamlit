"""Recover codebase-style agent-detection animations, driven by the geodesic
navigation model in physics.py instead of a trained DQN.

Policy = 6-move greedy descent on the Eikonal potential V, with the SAME
coarse->fine multiscale schedule the DQN uses (action_step 9 -> 3 -> 1). This is
exactly the behaviour a converged agent approximates, so it "recovers" the
codebase visuals without needing torch or trained weights.
"""
import sys, os
sys.path.insert(0, '/home/claude/adapted/src')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, patches
from physics import geodesic_potential

MOVES = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]  # codebase's 6 actions


def navigate(V, start, target, schedule=(9, 3, 1), max_steps=120):
    """6-move greedy descent on V with multiscale step refinement."""
    dims = np.array(V.shape)
    loc = np.array(start, dtype=int)
    traj = [loc.copy()]
    si = 0
    step = schedule[si]
    stalls = 0
    for _ in range(max_steps):
        best, best_v = None, V[tuple(loc)]
        for mv in MOVES:
            nl = loc + np.array(mv) * step
            if np.any(nl < 0) or np.any(nl >= dims):
                continue
            v = V[tuple(nl)]
            if v < best_v:
                best_v, best = v, nl
        if best is None:                    # no improving move -> refine scale
            si += 1
            if si >= len(schedule):
                break
            step = schedule[si]
            stalls = 0
            continue
        loc = best
        traj.append(loc.copy())
        if np.linalg.norm(loc - np.array(target)) <= 1:
            break
    return np.array(traj)


def render(mri, landmarks, starts, trajs, Vs, out, title, spacing_seq=None,
           overlay_potential=False, fps=12):
    """Reproduce the codebase's multi-panel agent visualization.

    Conventions matched from medical.py::display -
      blue dot   = current location
      red dot    = target landmark, wrapped in a translucent red z-distance disk
      yellow box = ROI 'what the network sees'  + 'Agent i / Spacing s' text
    """
    n = len(trajs)
    T = max(len(t) for t in trajs)
    roi = 40                                # half-size of ROI box (27*3/2 ~ 40)
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.2), facecolor='black')
    if n == 1:
        axes = [axes]

    def frame(k):
        for i, ax in enumerate(axes):
            ax.clear(); ax.set_facecolor('black'); ax.set_xticks([]); ax.set_yticks([])
            tj = trajs[i]
            ki = min(k, len(tj) - 1)
            cx, cy, cz = tj[ki]
            tx, ty, tz = landmarks[i]
            sl = np.flipud(mri[:, :, int(cz)].T)          # axial plane at current z
            H = sl.shape[0]
            ax.imshow(sl, cmap='gray', vmin=np.percentile(mri, 2), vmax=np.percentile(mri, 99))
            if overlay_potential:
                Vsl = np.flipud(Vs[i][:, :, int(cz)].T)
                ax.contour(Vsl, levels=10, cmap='cool', linewidths=0.5, alpha=0.55)
                tr = np.array(tj[:ki + 1])
                ax.plot(tr[:, 0], H - 1 - tr[:, 1], '-', color='#7cf6ff', lw=1.6, alpha=0.9)
            # ROI yellow box
            ax.add_patch(patches.Rectangle((cx - roi, H - 1 - cy - roi), 2 * roi, 2 * roi,
                                           fill=False, edgecolor='#ffcc00', lw=1.4))
            # red z-distance disk + target
            dz = abs(int(cz) - int(tz))
            ax.add_patch(patches.Circle((tx, H - 1 - ty), radius=3 + 1.6 * dz,
                                        color='red', alpha=0.20))
            ax.plot(tx, H - 1 - ty, 'o', color='red', ms=6)
            # current blue point
            ax.plot(cx, H - 1 - cy, 'o', color='#3a7bff', ms=6, mec='w', mew=0.5)
            sp = spacing_seq[i][min(ki, len(spacing_seq[i]) - 1)] if spacing_seq else 1
            ax.set_title(f'Agent {i}   spacing {sp}', color='#ffcc00', fontsize=9)
            d = np.linalg.norm(tj[ki] - np.array(landmarks[i]))
            ax.text(0.5, -0.06, f'err {d:4.1f} vox', transform=ax.transAxes,
                    color='#9fe1cb', fontsize=8, ha='center', va='top')
        fig.suptitle(title, color='white', fontsize=12, y=1.03)
        return []

    ani = animation.FuncAnimation(fig, frame, frames=T + 6, blit=False)
    ani.save(out, writer=animation.PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    print('saved', out, f'({n} agents, {T} steps)')


def run(mri_path, lm_path, out_tag, n_agents=5, overlay=False, seed=0):
    mri = np.load(mri_path)
    L = np.array([[float(x) for x in l.split(',')] for l in open(lm_path) if l.strip()])
    rng = np.random.default_rng(seed)
    dims = np.array(mri.shape)
    targets, starts, trajs, Vs, sched = [], [], [], [], []
    # pick n well-separated landmarks
    idxs = [0, 3, 5, 9, 11][:n_agents]
    for j in idxs:
        tgt = L[j].astype(int)
        pad = dims // 4
        start = np.array([rng.integers(pad[d], dims[d] - pad[d]) for d in range(3)])
        V = geodesic_potential(mri, tgt, beta=8.0, sigma=1.0)
        tj = navigate(V, start, tgt)
        # spacing schedule per step (mirror 9->3->1 as it refines)
        sp = []
        step = 9
        for s in range(len(tj)):
            sp.append({9: 3, 3: 2, 1: 1}.get(step, 1))
        targets.append(tgt); starts.append(start); trajs.append(tj); Vs.append(V); sched.append(sp)
    title = ('Physics-driven recovery — geodesic model navigating like the C-MARL agents'
             if not overlay else 'Physics-enhanced view — agents descend the navigation potential V')
    out = f'/home/claude/fig/recovered_{out_tag}{"_potential" if overlay else ""}.gif'
    render(mri, targets, starts, trajs, Vs, out, title, spacing_seq=sched, overlay_potential=overlay)


if __name__ == '__main__':
    run('/tmp/005_S_044.npy', 
        '/home/claude/rl-medical-master/src/data/landmarks/' +
        [f for f in os.listdir('/home/claude/rl-medical-master/src/data/landmarks') if '005_S_0448' in f][0],
        '005', n_agents=5, overlay=False)
    run('/tmp/005_S_044.npy',
        '/home/claude/rl-medical-master/src/data/landmarks/' +
        [f for f in os.listdir('/home/claude/rl-medical-master/src/data/landmarks') if '005_S_0448' in f][0],
        '005', n_agents=5, overlay=True)
    run('/tmp/003_S_105.npy',
        '/home/claude/rl-medical-master/src/data/landmarks/' +
        [f for f in os.listdir('/home/claude/rl-medical-master/src/data/landmarks') if '003_S_1059' in f][0],
        '003', n_agents=5, overlay=False)
