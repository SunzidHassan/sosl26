"""Dirichlet object map: the vision branch of the fusion controller.

Every grid cell c holds Dirichlet evidence beta[c, k] over K classes (the
YOLO object classes + Background). The posterior class distribution is

    p(o_k | z_c) = beta[c, k] / sum_j beta[c, j]

How evidence is added (DirichletObjectMap.update, once per camera frame):

1. Uniform prior: beta starts at prior_strength / K for every class.
2. Detections: a detection of class d with confidence xi adds
   xi * conf_temper * C[d, :] to every cell of its footprint. C[d, :] is
   the confusion-matrix row of d (how likely each true class, Background
   included, is to be detected as d), so a detection is soft evidence.
3. Footprint: cells within half the metric bounding-box width of the
   detection (bbox width in px * depth / fx / 2), clipped to
   [min_radius, max_radius]; always at least the cell containing it.
4. Overlap: in one frame, overlapping detections of the SAME class keep the
   max confidence per cell (not the sum), so duplicate boxes of one object
   don't double its evidence. Different classes add up independently.
5. Background: every cell in the camera's field of view (within max range)
   that no detection covers gets Background evidence that weakens with
   distance and is capped by the detector's miss rate eta:
       (1 - eta) / (1 + lambda * distance)

Source probability given vision:

    sem[c]         = sum_k p(o_k | z_c) * max(0, sim(class_k, goal))
    P(src | V)[c]  = sem[c] / sum_c sem[c]      (0 outside the search boundary)

sim is the cosine similarity of sentence embeddings of the class name and the
goal phrase. Background similarity is fixed (0 by default: empty space does
not emit odor).

The map uses the same SearchGrid (cells, boundary) as the olfactory map, so
all maps align.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy.special import digamma  # noqa: E402

BACKGROUND = 'Background'


def confusion_matrix_from_normalized(classes, matrix, labels, bg_fp_rate=0.05, bg_label='background'):
    """Confusion matrix C[pred][true] for the map, from an Ultralytics *normalized* one.

    `matrix` has rows = predicted, columns = true, both in `labels` order (background
    last). The result is reordered to `classes` + Background and every column sums to 1.

    Object columns are P(pred | true object), including the miss row (-> Background),
    re-normalized for rounding. The Ultralytics background column is the share of
    false positives per class (it sums to 1), not a rate, so it is scaled by
    `bg_fp_rate`: the probability that a background region yields any false detection.
    """
    M = np.asarray(matrix, dtype=float)
    labels = list(labels)
    if M.shape != (len(labels), len(labels)):
        raise ValueError(f"matrix shape {M.shape} does not match {len(labels)} labels")
    if set(classes) != set(labels) - {bg_label}:
        raise ValueError(f"model classes {sorted(classes)} != matrix classes {sorted(set(labels) - {bg_label})}; "
                         f"fix the matrix labels or yolo_exclude_classes")
    order = [labels.index(c) for c in classes] + [labels.index(bg_label)]
    C = M[np.ix_(order, order)].copy()
    n = len(classes)

    # Background column: false-positive share per class, scaled to a rate
    share = C[:n, n]
    share = share / share.sum() if share.sum() > 0 else np.full(n, 1.0 / n)
    C[:n, n] = bg_fp_rate * share
    C[n, n] = 1.0 - bg_fp_rate

    for j in range(n):                      # rounding: 0.91 + 0.04 + 0.04 = 0.99
        C[:, j] /= C[:, j].sum()
    return C


class DirichletObjectMap:

    def __init__(self, grid, classes, confusion_matrix, conf_temper=1.0, bg_false_neg_rate=None,
                 bg_dist_decay=0.5, prior_strength=1.0, min_radius=None, max_radius=1.0,
                 class_colors=None, gt_positions=None):
        """
        Parameters
        ----------
        grid : sOSL_tb4Functions.SearchGrid
            Cells and search boundary (shared with the olfactory map).
        classes : list[str]
            Object classes (YOLO names); Background is appended as the last class.
        confusion_matrix : (K, K) array
            C[pred][true] in `self.classes` order, columns summing to 1
            (see confusion_matrix_from_normalized).
        conf_temper : float
            Evidence weight = YOLO confidence * conf_temper. < 1 down-weights
            evidence, e.g. for correlated consecutive frames.
        bg_false_neg_rate : float or None
            eta: probability the detector misses an object that is in view.
            Lower eta -> "no detection" is stronger Background evidence.
            None = the mean miss rate of the confusion matrix.
        bg_dist_decay : float
            lambda: how fast Background evidence falls off with distance from
            the camera. 0 = no decay.
        prior_strength : float
            Total prior evidence per cell, spread uniformly over the K classes.
        min_radius, max_radius : float
            Footprint radius clip (m); min_radius None = half a cell.
        class_colors : {class: matplotlib color} for the plots (case-insensitive, incl. 'Background').
        gt_positions : {class: (map x, map y)} ground-truth object positions drawn on the plots.
        """
        self.grid = grid
        self.classes = list(classes) + [BACKGROUND]
        self.bg_idx = len(self.classes) - 1
        self.cls_idx = {c: i for i, c in enumerate(self.classes)}
        self.K = len(self.classes)
        self.H, self.W = grid.inside.shape

        self.confusion_matrix = np.asarray(confusion_matrix, dtype=float)
        if self.confusion_matrix.shape != (self.K, self.K):
            raise ValueError(f"confusion_matrix shape {self.confusion_matrix.shape} must be "
                             f"({self.K}, {self.K}) matching {self.classes}")
        if not np.allclose(self.confusion_matrix.sum(axis=0), 1.0, atol=1e-6):
            raise ValueError("confusion_matrix columns must sum to 1")

        self.conf_temper = conf_temper
        # Mean of the Background (miss) row over the object columns = mean miss rate
        self.bg_false_neg_rate = (float(self.confusion_matrix[-1, :-1].mean())
                                  if bg_false_neg_rate is None else bg_false_neg_rate)
        self.bg_dist_decay = bg_dist_decay
        self.min_radius = grid.step / 2.0 if min_radius is None else min_radius
        self.max_radius = max_radius

        self.prior_value = prior_strength / self.K
        self.beta = np.full((self.H, self.W, self.K), self.prior_value)   # Dirichlet evidence per cell
        self.observed = np.zeros((self.H, self.W), dtype=bool)            # cells that received any evidence

        self.Xc, self.Zc = np.meshgrid(grid.x_points, grid.z_points)     # cell centres, (H, W)
        self.class_weights = None       # max(0, sim(class, goal)), set by set_class_similarity
        self.similarity_table = None

        self.class_colors = {str(k).lower(): v for k, v in (class_colors or {}).items()}
        self.gt_positions = dict(gt_positions or {})

    # ------------------------------------------------------------------
    # Evidence accumulation
    # ------------------------------------------------------------------

    def footprint_mask(self, x, z, radius):
        """Cells whose centre lies within `radius` of (x, z), plus the cell containing (x, z) if on the grid."""
        mask = np.hypot(self.Xc - x, self.Zc - z) <= radius
        x0, x1, z0, z1 = self.grid.extent()
        if x0 <= x <= x1 and z0 <= z <= z1:
            mask[self.grid.nearest_cell(x, z)] = True
        return mask

    def fov_mask(self, cam_xz, heading, hfov, max_range=None):
        """Cells inside the camera's horizontal field of view (and range)."""
        dx, dz = self.Xc - cam_xz[0], self.Zc - cam_xz[1]
        bearing = np.arctan2(dz, dx) - heading
        bearing = (bearing + np.pi) % (2.0 * np.pi) - np.pi     # wrap to [-pi, pi)
        mask = np.abs(bearing) <= hfov / 2.0
        if max_range is not None:
            mask &= np.hypot(dx, dz) <= max_range
        return mask

    def update(self, detections, cam_xz, heading, hfov, max_range=None):
        """Adds one frame of evidence.

        Parameters
        ----------
        detections : list[dict]
            From sOSL_tb4Functions.detect_objects: objectType, map_x, map_y,
            radius and Conf (YOLO box confidence, 0-1).
        cam_xz : tuple
            Camera position (map x, map y).
        heading : float
            Camera optical axis yaw in the map frame (rad, ROS convention).
        hfov : float
            Horizontal field of view (rad).
        max_range : float
            Cells farther than this from the camera are not considered seen.
        """
        footprint_union = np.zeros((self.H, self.W), dtype=bool)   # cells covered by any detection
        contrib = {}   # predicted-class idx -> (H, W) max confidence weight per cell

        for det in detections:
            k = self.cls_idx.get(det['objectType'])
            if k is None or k == self.bg_idx:
                continue   # class not in the map (e.g. excluded)
            radius = float(np.clip(det.get('radius', self.min_radius), self.min_radius, self.max_radius))
            fp = self.footprint_mask(det['map_x'], det['map_y'], radius)
            xi = float(det['Conf']) * self.conf_temper

            # Same class: keep the max confidence per cell instead of summing
            weight_map = contrib.setdefault(k, np.zeros((self.H, self.W)))
            np.maximum(weight_map, np.where(fp, xi, 0.0), out=weight_map)
            footprint_union |= fp

        # Detections: weight * confusion-matrix row (soft evidence over all true classes)
        for k, weight_map in contrib.items():
            self.beta += weight_map[..., None] * self.confusion_matrix[k][None, None, :]

        # Background: visible cells without a detection, weaker with distance
        bg_cells = self.fov_mask(cam_xz, heading, hfov, max_range) & ~footprint_union
        dist = np.hypot(self.Xc - cam_xz[0], self.Zc - cam_xz[1])
        bg_strength = (1.0 - self.bg_false_neg_rate) / (1.0 + self.bg_dist_decay * dist)
        self.beta[bg_cells, self.bg_idx] += bg_strength[bg_cells]

        self.observed |= footprint_union | bg_cells

    # ------------------------------------------------------------------
    # Posterior / uncertainty
    # ------------------------------------------------------------------

    def posterior(self):
        """p(o_k | z_c), shape (H, W, K)."""
        return self.beta / self.beta.sum(axis=2, keepdims=True)

    def mle_class(self):
        """argmax_k p(o_k | z_c); -1 for cells that never received evidence."""
        mle = np.argmax(self.posterior(), axis=2)
        mle[~self.observed] = -1
        return mle

    def shannon_entropy(self):
        """Entropy of the posterior mean p(o | z) per cell (nats)."""
        p = self.posterior()
        return -np.sum(p * np.log(p + 1e-12), axis=2)

    def expected_entropy(self):
        """Expected Shannon entropy under the Dirichlet (Voxeland Eq. 7): also falls as evidence grows."""
        a0 = self.beta.sum(axis=2)
        return digamma(a0 + 1.0) - np.sum(self.beta * digamma(self.beta + 1.0), axis=2) / a0

    def observed_classes(self):
        """Classes that received any evidence (Background included)."""
        gained = (self.beta > self.prior_value + 1e-9).any(axis=(0, 1))
        return [c for c, g in zip(self.classes, gained) if g]

    # ------------------------------------------------------------------
    # Semantic similarity -> P(source | vision)
    # ------------------------------------------------------------------

    def set_class_similarity(self, sentence_model, goal_phrase, background_similarity=0.0):
        """Cosine similarity between goal_phrase and every class name (computed once).

        Negative similarities are clipped to 0 (an unrelated class is not evidence against the source).
        """
        goal = np.asarray(sentence_model.encode(goal_phrase, convert_to_tensor=False), dtype=float)
        embs = np.asarray(sentence_model.encode(self.classes[:-1], convert_to_tensor=False), dtype=float)
        sims = embs @ goal / (np.linalg.norm(embs, axis=1) * np.linalg.norm(goal) + 1e-12)
        raw = np.append(sims, background_similarity)
        self.class_weights = np.clip(raw, 0.0, None)
        self.similarity_table = pd.DataFrame({
            'class': self.classes, 'goal_phrase': goal_phrase,
            'cosine_similarity': raw, 'weight': self.class_weights,
        }).sort_values('cosine_similarity', ascending=False).reset_index(drop=True)
        return self.similarity_table

    def semantic_likelihood_map(self):
        """sum_k p(o_k | z_c) * weight_k, shape (H, W)."""
        return np.tensordot(self.posterior(), self.class_weights, axes=([2], [0]))

    def source_prob_given_vision(self):
        """P(src | V): the semantic likelihood normalised over the cells inside the boundary (0 outside)."""
        inside = self.grid.inside
        sem = np.where(inside, self.semantic_likelihood_map(), 0.0)
        total = sem.sum()
        if total < 1e-12:
            return inside / inside.sum()
        return sem / total

    def top_object_at(self, row, col):
        """Most probable non-background class at a cell (or 'N/A' if unobserved)."""
        if not self.observed[row, col]:
            return 'N/A'
        return self.classes[int(np.argmax(self.posterior()[row, col, :-1]))]

    # ------------------------------------------------------------------
    # Plots / saving
    # ------------------------------------------------------------------

    def _color(self, name):
        c = self.class_colors.get(str(name).lower())
        return c if c is not None else plt.get_cmap('tab20')(self.cls_idx.get(name, 0) % 20)

    def overlay_ground_truth(self, ax):
        """Stars at the ground-truth object positions, filled with the class color."""
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        for name, (gx, gy) in self.gt_positions.items():
            ax.plot(gx, gy, marker='*', ms=18, mfc=self._color(name), mec='white', mew=1.5, ls='', zorder=6)
            ax.annotate(name, (gx, gy), xytext=(0, 10), textcoords='offset points', ha='center',
                        fontsize=8, zorder=7, bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.85))
        ax.set_xlim(xlim)   # positions outside the map are clipped, not added to the view
        ax.set_ylim(ylim)

    def plot_mle(self, save_path, title='MLE class  argmax p(o | z)', robot_xz=None):
        """Class map of observed cells in the class colors; grey = never observed."""
        mle = self.mle_class()
        present = sorted(set(mle[mle >= 0].tolist()))
        fig, ax = plt.subplots(figsize=(7.5, 6))
        ax.set_facecolor((0.85, 0.85, 0.85))
        if present:
            # Re-index the classes present to 0..n-1 so the colormap only holds their colors
            img = np.full(mle.shape, np.nan)
            for i, k in enumerate(present):
                img[mle == k] = i
            cmap = ListedColormap([self._color(self.classes[k]) for k in present])
            im = ax.imshow(img, origin='lower', extent=self.grid.extent(), cmap=cmap,
                           vmin=-0.5, vmax=len(present) - 0.5, aspect='equal')
            cb = fig.colorbar(im, ax=ax, ticks=range(len(present)))
            cb.ax.set_yticklabels([self.classes[k] for k in present])
        self.overlay_ground_truth(ax)
        if robot_xz is not None:
            ax.plot(robot_xz[0], robot_xz[1], marker='o', ms=8, mfc='lime', mec='black', ls='', zorder=8)
        ax.set_title(title)
        ax.set_xlabel('map x (m)')
        ax.set_ylabel('map y (m)')
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)

    def _save_map(self, arr, path, title, **imshow_kw):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        im = ax.imshow(arr, origin='lower', extent=self.grid.extent(), aspect='equal', **imshow_kw)
        self.overlay_ground_truth(ax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def save_final(self, out_dir):
        """End of run: raw arrays, settings, similarity table, and plots of the
        per-class posteriors, MLE map, entropies and semantic likelihood."""
        g = self.grid
        post = self.posterior()
        np.savez_compressed(os.path.join(out_dir, 'dirichlet_final.npz'), beta=self.beta.astype(np.float32),
                            posterior=post.astype(np.float32), observed=self.observed,
                            x_points=g.x_points, z_points=g.z_points, inside=g.inside)
        with open(os.path.join(out_dir, 'dirichlet_classes.json'), 'w') as f:
            json.dump({'classes': self.classes, 'grid_HW': [self.H, self.W], 'res': g.step,
                       'extent': g.extent(), 'conf_temper': self.conf_temper,
                       'bg_false_neg_rate': self.bg_false_neg_rate, 'bg_dist_decay': self.bg_dist_decay,
                       'prior_value': self.prior_value, 'radius_clip': [self.min_radius, self.max_radius],
                       'confusion_matrix': self.confusion_matrix.tolist()},
                      f, indent=2, default=float)
        if self.similarity_table is not None:
            self.similarity_table.to_csv(os.path.join(out_dir, 'semantic_similarity_table.csv'), index=False)

        maps_dir = os.path.join(out_dir, 'dirichlet_maps')
        os.makedirs(maps_dir, exist_ok=True)
        for cls in self.observed_classes():
            self._save_map(post[:, :, self.cls_idx[cls]],
                           os.path.join(maps_dir, f"posterior__{cls.replace(' ', '_')}.png"),
                           f'p(o = {cls} | z)', cmap='magma', vmin=0, vmax=1)
        self.plot_mle(os.path.join(maps_dir, 'mle_class.png'))
        self._save_map(self.shannon_entropy(), os.path.join(maps_dir, 'entropy_shannon.png'),
                       'H[p(o|z)] (nats)', cmap='viridis')
        self._save_map(self.expected_entropy(), os.path.join(maps_dir, 'entropy_dirichlet.png'),
                       'E[H] under Dirichlet (nats)', cmap='viridis')
        self._save_map(self.semantic_likelihood_map(), os.path.join(maps_dir, 'semantic_likelihood.png'),
                       'sum_k p(o_k|z) * sim(class_k, goal)', cmap='viridis')
