"""Dirichlet evidence accumulation object map (second vision approach).

Per grid cell c, a Dirichlet over K = (object classes + Background) holds
evidence beta[c]. The posterior class distribution is

    p(o_k | z_c) = beta[c, k] / sum_j beta[c, j]

Changes with respect to the confusion-matrix sample (psgsl_dirichlet_map.py):

1. Confusion-matrix evidence: every detection adds `xi * CONF[d]` to its
   cells' footprint, where `xi = confidence * conf_temper` and CONF[d] is
   the predicted class's full likelihood row over true classes (object
   classes AND Background) -- not a flat per-class indicator.
2. Uniform prior: beta starts at prior_strength / K for every class
   (objects and Background alike).
3. Active, FOV-wide background: every frame (including frames with zero
   detections), every visible occupied cell that isn't inside a detected
   object's footprint receives Background evidence, decayed by distance
   from the camera and capped by an assumed detector false-negative rate:
       bg_strength = (1 - eta) / (1 + lambda * dist)
4. Dynamic footprint: the footprint radius of each detection is half its
   metric bounding box width (bbox width in px * depth / fx / 2), clipped
   to [min_radius, max_radius], instead of a fixed FOOTPRINT_RADIUS.
5. Overlap handling: within one frame, detections of the SAME predicted
   class that overlap in cells are deduplicated by taking the max
   confidence per cell (not summed) before the confusion-matrix row is
   applied, so repeated/duplicate detections of one object don't inflate
   evidence. Detections of DIFFERENT predicted classes still contribute
   independently in shared cells.

Source probability given vision:

    sem[c]         = sum_k p(o_k | z_c) * max(0, sim(class_k, goal))
    P(src | V)[c]  = sem[c] / sum_c sem[c]

Background similarity is fixed to `background_similarity` (0 by default:
empty space doesn't emit odor) instead of embedding the word "Background".

Grid cells are centred on the same x_points / z_points as the Bayesian
olfactory map (z = ROS map y, see sOSL_tb4Functions.py), so all maps align.
"""

import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import digamma  # noqa: E402

BACKGROUND = 'Background'

def confusion_matrix_from_normalized(classes, matrix, labels, bg_fp_rate=0.05,
                                     background=BACKGROUND, bg_label='background'):
    """C[pred][true] (columns sum to 1) in DirichletObjectMap.classes order
    (objects in the given order, Background last) from an Ultralytics
    *normalized* confusion matrix (rows = predicted, columns = true, both in `labels` order).

    Object columns are P(pred | true object), including the miss row (-> Background), and are
    re-normalized for rounding. The background column of the Ultralytics matrix is the share of
    false positives per class (it sums to 1), not a rate, so it is scaled by `bg_fp_rate`
    (probability that a background region yields any false detection).
    """
    M = np.asarray(matrix, dtype=float)
    labels = list(labels)
    if M.shape != (len(labels), len(labels)):
        raise ValueError(f"matrix shape {M.shape} does not match {len(labels)} labels")
    objs = [c for c in classes if c != background]
    if set(objs) != set(labels) - {bg_label}:
        raise ValueError(f"model classes {sorted(objs)} != matrix classes {sorted(set(labels) - {bg_label})}; "
                         f"fix the matrix labels or yolo_exclude_classes")
    order = [labels.index(c) for c in objs] + [labels.index(bg_label)]
    C = M[np.ix_(order, order)].copy()
    n = len(objs)

    share = C[:n, n]
    share = share / share.sum() if share.sum() > 0 else np.full(n, 1.0 / n)
    C[:n, n] = bg_fp_rate * share
    C[n, n] = 1.0 - bg_fp_rate

    for j in range(n):                      # rounding: 0.91 + 0.04 + 0.04 = 0.99
        C[:, j] /= C[:, j].sum()
    return C

class DirichletObjectMap:

    def __init__(self, x_points, z_points, classes, confusion_matrix,
                 background=BACKGROUND, conf_temper=1.0,
                 bg_false_neg_rate=0.15, bg_dist_decay=0.5,
                 prior_strength=1.0, min_radius=None, max_radius=1.0):
        """
        Parameters
        ----------
        confusion_matrix : (K, K) array
            C[pred][true] = p(detector predicts `pred` | object truly `true`).
            Row/column order MUST match `self.classes` (object classes in the
            order given, with `background` moved to the last index) -- same
            convention as CONF in psgsl_dirichlet_map.py. Columns must sum to 1.
        conf_temper : float
            Scales detection confidence before it's used as evidence weight
            (xi = confidence * conf_temper). <1.0 down-weights evidence, e.g.
            to account for correlated consecutive frames.
        bg_false_neg_rate : float
            eta: assumed probability the detector misses a real object that
            is actually in view. Caps how strongly "no detection" counts as
            background evidence (lower eta -> stronger background evidence).
        bg_dist_decay : float
            lambda: how fast background evidence falls off with distance
            from the camera. 0 = no decay.
        """
        self.x_points = np.asarray(x_points, dtype=float)
        self.z_points = np.asarray(z_points, dtype=float)
        self.classes = [c for c in classes if c != background] + [background]
        self.background = background
        self.bg_idx = len(self.classes) - 1
        self.cls_idx = {c: i for i, c in enumerate(self.classes)}
        self.K = len(self.classes)
        self.H, self.W = len(self.z_points), len(self.x_points)

        self.confusion_matrix = np.asarray(confusion_matrix, dtype=float)
        if self.confusion_matrix.shape != (self.K, self.K):
            raise ValueError(
                f"confusion_matrix shape {self.confusion_matrix.shape} must be "
                f"({self.K}, {self.K}) matching self.classes order {self.classes}")
        if not np.allclose(self.confusion_matrix.sum(axis=0), 1.0, atol=1e-6):
            raise ValueError("confusion_matrix columns must sum to 1")

        self.res = float(self.x_points[1] - self.x_points[0]) if self.W > 1 else 0.25
        self.conf_temper = conf_temper
        self.bg_false_neg_rate = bg_false_neg_rate
        self.bg_dist_decay = bg_dist_decay
        self.min_radius = self.res / 2.0 if min_radius is None else min_radius
        self.max_radius = max_radius

        # (2) uniform prior over objects + background
        self.prior_value = prior_strength / self.K
        self.beta = np.full((self.H, self.W, self.K), self.prior_value)
        self.observed = np.zeros((self.H, self.W), dtype=bool)

        self.Xc, self.Zc = np.meshgrid(self.x_points, self.z_points)   # cell centres, (H, W)
        self.class_weights = None
        self.similarity_table = None

    # ------------------------------------------------------------------
    # Evidence accumulation
    # ------------------------------------------------------------------

    def _nearest_cell(self, x, z):
        return int(np.abs(self.z_points - z).argmin()), int(np.abs(self.x_points - x).argmin())

    def _in_bounds(self, x, z):
        h = self.res / 2.0
        return (self.x_points[0] - h <= x <= self.x_points[-1] + h
                and self.z_points[0] - h <= z <= self.z_points[-1] + h)

    def footprint_mask(self, x, z, radius):
        """(4) Cells whose centre lies within `radius` of (x, z), at least the containing cell."""
        mask = np.hypot(self.Xc - x, self.Zc - z) <= radius
        if self._in_bounds(x, z):
            mask[self._nearest_cell(x, z)] = True
        return mask

    def fov_mask(self, cam_xz, heading, hfov, max_range=None):
        """Cells inside the camera's horizontal field of view (and range)."""
        dx, dz = self.Xc - cam_xz[0], self.Zc - cam_xz[1]
        bearing = np.arctan2(dz, dx) - heading
        bearing = (bearing + np.pi) % (2.0 * np.pi) - np.pi
        mask = np.abs(bearing) <= hfov / 2.0
        if max_range is not None:
            mask &= np.hypot(dx, dz) <= max_range
        return mask

    def update(self, detections, cam_xz, heading, hfov, max_range=None):
        """Adds one frame of evidence.

        Parameters
        ----------
        detections : list[dict]
            From sOSL_tb4Functions.visionBranch -- expects objectType,
            map_x, map_y, radius, and confidence (YOLO box confidence, 0-1).
        cam_xz : tuple
            Camera position (map x, map y).
        heading : float
            Camera optical axis yaw in the map frame (rad, ROS convention).
        hfov : float
            Horizontal field of view (rad).

        Returns
        -------
        dict
            Number of object / background cells updated and skipped detections.
        """
        footprint_union = np.zeros((self.H, self.W), dtype=bool)
        contrib = {}   # predicted-class idx -> (H, W) max-confidence-weight map
        skipped = []

        for det in detections:
            name = det['objectType']
            if name not in self.cls_idx or name == self.background:
                skipped.append(name)
                continue
            obj_xz = (det['map_x'], det['map_y'])
            radius = float(np.clip(det.get('radius', self.min_radius), self.min_radius, self.max_radius))
            fp = self.footprint_mask(obj_xz[0], obj_xz[1], radius)

            d_idx = self.cls_idx[name]
            xi = float(det['Conf']) * self.conf_temper

            # (5) dedupe same-class overlapping detections in this frame:
            # keep the max confidence per cell, don't sum contributions.
            weight_map = contrib.get(d_idx)
            if weight_map is None:
                weight_map = np.zeros((self.H, self.W))
                contrib[d_idx] = weight_map
            np.maximum(weight_map, np.where(fp, xi, 0.0), out=weight_map)

            footprint_union |= fp

        # (1) apply confusion-matrix evidence, scaled by max confidence per cell
        for d_idx, weight_map in contrib.items():
            L = self.confusion_matrix[d_idx]          # (K,) likelihood row, includes background
            self.beta += weight_map[..., None] * L[None, None, :]

        # (3) background evidence: every visible cell not inside a detected
        # footprint, every frame, decayed by distance from the camera.
        visible = self.fov_mask(cam_xz, heading, hfov, max_range)
        bg_cells = visible & ~footprint_union

        dist = np.hypot(self.Xc - cam_xz[0], self.Zc - cam_xz[1])
        bg_strength = (1.0 - self.bg_false_neg_rate) / (1.0 + self.bg_dist_decay * dist)
        self.beta[bg_cells, self.bg_idx] += bg_strength[bg_cells]

        self.observed |= footprint_union | bg_cells

        return dict(object_cells=int(footprint_union.sum()), background_cells=int(bg_cells.sum()),
                    skipped=skipped)

    # ------------------------------------------------------------------
    # Posterior / uncertainty
    # ------------------------------------------------------------------

    def posterior(self):
        return self.beta / self.beta.sum(axis=2, keepdims=True)

    def mle_class(self):
        """argmax_k p(o_k | z_c); -1 for cells that never received evidence."""
        mle = np.argmax(self.posterior(), axis=2)
        mle[~self.observed] = -1
        return mle

    def shannon_entropy(self):
        p = self.posterior()
        return -np.sum(p * np.log(p + 1e-12), axis=2)

    def expected_entropy(self):
        """Expected Shannon entropy under the Dirichlet (Voxeland Eq. 7)."""
        a0 = self.beta.sum(axis=2)
        return digamma(a0 + 1.0) - np.sum(self.beta * digamma(self.beta + 1.0), axis=2) / a0

    def observed_classes(self):
        """Classes that received any evidence (Background included)."""
        gained = (self.beta > self.prior_value + 1e-9).any(axis=(0, 1))
        return [c for c, g in zip(self.classes, gained) if g]

    # ------------------------------------------------------------------
    # Semantic similarity -> P(source | vision)
    # ------------------------------------------------------------------

    def set_class_similarity(self, sentence_model, goal_phrase, background_similarity=0.0,
                             clip_negative=True):
        """Cosine similarity between goal_phrase and every class name (computed once)."""
        names = self.classes[:-1]
        goal = np.asarray(sentence_model.encode(goal_phrase, convert_to_tensor=False), dtype=float)
        embs = np.asarray(sentence_model.encode(names, convert_to_tensor=False), dtype=float)
        sims = embs @ goal / (np.linalg.norm(embs, axis=1) * np.linalg.norm(goal) + 1e-12)
        raw = np.append(sims, background_similarity)
        self.class_weights = np.clip(raw, 0.0, None) if clip_negative else raw
        self.similarity_table = pd.DataFrame({
            'class': self.classes, 'goal_phrase': goal_phrase,
            'cosine_similarity': raw, 'weight': self.class_weights,
        }).sort_values('cosine_similarity', ascending=False).reset_index(drop=True)
        return self.similarity_table

    def semantic_likelihood_map(self):
        """sum_k p(o_k | z_c) * weight_k, shape (H, W)."""
        return np.tensordot(self.posterior(), self.class_weights, axes=([2], [0]))

    def source_prob_given_vision(self):
        sem = self.semantic_likelihood_map()
        total = sem.sum()
        if total < 1e-12:
            return np.full_like(sem, 1.0 / sem.size)
        return sem / total

    def top_object_at(self, row, col):
        """Most probable non-background class at a cell (or 'N/A' if unobserved)."""
        if not self.observed[row, col]:
            return 'N/A'
        p = self.posterior()[row, col, :-1]
        return self.classes[int(np.argmax(p))]

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _extent(self):
        h = self.res / 2.0
        return [self.x_points[0] - h, self.x_points[-1] + h, self.z_points[0] - h, self.z_points[-1] + h]

    def plot_mle(self, save_path, title='MLE class  argmax p(o | z)', robot_xz=None):
        """Class map of observed cells; grey = never observed."""
        mle = self.mle_class()
        present = sorted(set(mle[mle >= 0].tolist()))
        fig, ax = plt.subplots(figsize=(7.5, 6))
        ax.set_facecolor((0.85, 0.85, 0.85))
        if present:
            lut = {k: i for i, k in enumerate(present)}
            img = np.full(mle.shape, np.nan)
            for k, i in lut.items():
                img[mle == k] = i
            cmap = plt.get_cmap('tab20', max(len(present), 2))
            im = ax.imshow(img, origin='lower', extent=self._extent(), cmap=cmap,
                           vmin=-0.5, vmax=max(len(present), 2) - 0.5, aspect='equal')
            cb = fig.colorbar(im, ax=ax, ticks=range(len(present)))
            cb.ax.set_yticklabels([self.classes[k] for k in present])
        if robot_xz is not None:
            ax.plot(robot_xz[0], robot_xz[1], marker='o', ms=8, mfc='lime', mec='black', ls='')
        ax.set_title(title)
        ax.set_xlabel('map x (m)')
        ax.set_ylabel('map y (m)')
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)

    def save_final(self, out_dir):
        """Per-class posteriors (observed classes), MLE, entropies, similarity table."""
        maps_dir = os.path.join(out_dir, 'dirichlet_maps')
        os.makedirs(maps_dir, exist_ok=True)
        post = self.posterior()
        np.savez_compressed(os.path.join(out_dir, 'dirichlet_final.npz'), beta=self.beta.astype(np.float32),
                            posterior=post.astype(np.float32), observed=self.observed,
                            x_points=self.x_points, z_points=self.z_points)
        with open(os.path.join(out_dir, 'dirichlet_classes.json'), 'w') as f:
            json.dump({'classes': self.classes, 'grid_HW': [self.H, self.W], 'res': self.res,
                       'extent': self._extent(), 'conf_temper': self.conf_temper,
                       'bg_false_neg_rate': self.bg_false_neg_rate,
                       'bg_dist_decay': self.bg_dist_decay,
                       'prior_value': self.prior_value, 'radius_clip': [self.min_radius, self.max_radius],
                       'confusion_matrix': self.confusion_matrix.tolist()},
                      f, indent=2)
        if self.similarity_table is not None:
            self.similarity_table.to_csv(os.path.join(out_dir, 'semantic_similarity_table.csv'), index=False)

        extent = self._extent()
        for cls in self.observed_classes():
            k = self.cls_idx[cls]
            fig, ax = plt.subplots(figsize=(7, 5.5))
            im = ax.imshow(post[:, :, k], origin='lower', extent=extent, cmap='magma', vmin=0, vmax=1,
                           aspect='equal')
            ax.set_title(f'p(o = {cls} | z)')
            fig.colorbar(im, ax=ax, label='probability')
            fig.tight_layout()
            fig.savefig(os.path.join(maps_dir, f"posterior__{cls.replace(' ', '_')}.png"), dpi=120)
            plt.close(fig)

        self.plot_mle(os.path.join(maps_dir, 'mle_class.png'))

        for name, arr, lab in [('entropy_shannon', self.shannon_entropy(), 'H[p(o|z)] (nats)'),
                               ('entropy_dirichlet', self.expected_entropy(), 'E[H] under Dirichlet (nats)'),
                               ('semantic_likelihood', self.semantic_likelihood_map(),
                                'sum_k p(o_k|z) * sim(class_k, goal)')]:
            fig, ax = plt.subplots(figsize=(7, 5.5))
            im = ax.imshow(arr, origin='lower', extent=extent, cmap='viridis', aspect='equal')
            ax.set_title(lab)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(maps_dir, f'{name}.png'), dpi=120)
            plt.close(fig)


def camera_pose_in_map(optical_to_map):
    """Camera (x, y) and optical-axis yaw in the map frame from an optical->map transform."""
    origin = np.asarray(optical_to_map(np.zeros(3)), dtype=float)
    ahead = np.asarray(optical_to_map(np.array([0.0, 0.0, 1.0])), dtype=float)
    return (origin[0], origin[1]), math.atan2(ahead[1] - origin[1], ahead[0] - origin[0])