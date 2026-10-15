"""GraTok: 3D spatio-temporal graph token compression.

This is the single reference implementation of the method described in the
paper.  Both backbones in this repository (Qwen3-VL and LLaVA-OneVision-1.5)
used to carry their own copy of the merge loop, and the two copies had drifted
apart: the Qwen copy implemented the count-weighted merge described in
Algorithm 1 of the paper, while the LLaVA copy still contained the older
convex-combination merge (no ``represented_token_num``, and the score was
accumulated instead of averaged).  They are replaced by this module so that a
single, documented algorithm produces every number in the paper.

Algorithm (see Algorithm 1 of the paper)
----------------------------------------
1. Cosine similarity over the flattened tokens, split into an intra-frame part
   ``S_intra`` of shape ``(T, N, N)`` and an adjacent-frame part ``S_inter`` of
   shape ``(T-1, N, N)``.
2. Thresholded adjacency.  ``A_intra`` carries a self-loop on its diagonal, so
   that the degree of an isolated token is still defined; ``A_inter`` is
   strictly off-diagonal.  The self-loop is used for *degree only* and is never
   a merge candidate -- the source token is always removed from its own
   candidate set.
3. Importance score ``s = lam * w_st + (1 - lam) * (1 - dep_st)``, where ``w_st``
   combines the spatial saliency (deviation from the 8-neighbourhood mean
   similarity) and the temporal saliency (deviation from the previous/next
   frame), and ``dep_st`` is the normalised spatial/temporal degree.
4. Repeatedly remove the lowest-scoring active token and merge it into its most
   similar *other* active neighbour, weighting the feature by both the
   similarity and the number of original tokens each side represents.  If a
   token has no surviving neighbour it is *kept as is* by default
   (``keep_isolated``); ``GRATOK_DROP_ISOLATED=1`` restores the old behaviour
   of deleting it, which is what the ablation of that branch compares against.

Only three quantities change during merging: the destination feature, its
score and the number of original tokens it represents.  The graph, the
similarity matrix and all normalisation constants are computed once.

Parameters can be overridden without editing the source, e.g.::

    GRATOK_RETENTION=0.3 python -m lmms_eval ... --model qwen3_vl_compress

which keeps the retention ratio of every run reproducible from the command
line instead of a hand-edited constant.
"""

from __future__ import annotations

import logging
import json
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

#: 2 x 2 patch pooling performed by the vision encoder before the tokens reach
#: the compressor.  It is a property of the model, not of the method, so it is
#: a constructor argument rather than a hard-coded ``// 2``.
DEFAULT_SPATIAL_MERGE_SIZE = 2

_ENV_KEYS = {
    "tau": "GRATOK_TAU",
    "alpha": "GRATOK_ALPHA",
    "beta": "GRATOK_BETA",
    "lam": "GRATOK_LAMBDA",
    "p": "GRATOK_RETENTION",
    "spatial_merge_size": "GRATOK_SPATIAL_MERGE",
    "cpu_max_tokens": "GRATOK_CPU_MAX_TOKENS",
}

#: Environment variables arrive as text, so they have to be cast before the
#: ``__init__`` validation (which compares them numerically) runs.
_ENV_CASTS = {"spatial_merge_size": int, "cpu_max_tokens": int}

#: Ablation switches, read from the environment as "1"/"true"/"yes".
#: Their only purpose is to reproduce the leave-one-out ablations of Table III
#: and the two joint-removal rows, without editing the source between runs.
_ENV_SWITCHES = {
    "use_sw": "GRATOK_NO_SW",
    "use_tw": "GRATOK_NO_TW",
    "use_sd": "GRATOK_NO_SD",
    "use_td": "GRATOK_NO_TD",
    # Table IV variants: GRATOK_DROP_ISOLATED restores the pre-2026-10
    # "delete a token that has no surviving neighbour" behaviour, so the old
    # and the new semantics can be compared from the command line.
    "keep_isolated": "GRATOK_DROP_ISOLATED",
    "propagate": "GRATOK_NO_PROPAGATE",
}

#: String-valued switches (Table IV).  ``sequential_last`` is not in the task
#: book: it exists so that the "take the highest index instead of the lowest"
#: variant can be tried without editing the source if a number does not match.
_ENV_CHOICES = {
    "select": "GRATOK_SELECT",
    "merge_mode": "GRATOK_MERGE",
    "backend": "GRATOK_BACKEND",
}

_CHOICES = {
    "select": ("importance", "random", "sequential", "sequential_last"),
    "merge_mode": ("weighted", "sum"),
    "backend": ("auto", "cpu", "gpu"),
}

_TRUTHY = {"1", "true", "yes", "on"}


class GraTok:
    """Training-free spatio-temporal graph token compressor.

    Args:
        tau: similarity threshold that defines the graph neighbourhood.
        alpha: spatial/temporal balance of the intrinsic saliency term.
        beta: spatial/temporal balance of the degree term.
        lam: balance between saliency and (inverted) degree.
        p: fraction of tokens to keep (the retention ratio).
        spatial_merge_size: patch pooling factor of the vision encoder.
        eps: numerical guard used by the normalisations.
        keep_isolated: keep a token that has no surviving neighbour instead of
            deleting it (the default; ``False`` is the old behaviour).
        propagate: write the merged feature/score/count back into the target
            token (Table IV "Sim Merge" turns this off).
        select: how the next token to reduce is chosen -- ``importance``
            (lowest score, the default), ``random`` or ``sequential``.
        merge_mode: ``weighted`` (similarity- and count-weighted average, the
            default) or ``sum`` (Table IV "Sum Merge").
    """

    def __init__(
        self,
        tau: float = 0.7,
        alpha: float = 0.3,
        beta: float = 0.5,
        lam: float = 0.5,
        p: float = 0.5,
        spatial_merge_size: int = DEFAULT_SPATIAL_MERGE_SIZE,
        eps: float = 1e-8,
        use_sw: bool = True,
        use_tw: bool = True,
        use_sd: bool = True,
        use_td: bool = True,
        keep_isolated: bool = True,
        propagate: bool = True,
        select: str = "importance",
        merge_mode: str = "weighted",
        backend: str = "auto",
        cpu_max_tokens: int = 32768,
        profile: bool = False,
        profile_out: Optional[str] = None,
    ) -> None:
        if not 0.0 < p <= 1.0:
            raise ValueError(f"retention ratio p must be in (0, 1], got {p}")
        for name, value in (("alpha", alpha), ("beta", beta), ("lam", lam)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not -1.0 <= tau <= 1.0:
            raise ValueError(f"tau must be a cosine similarity in [-1, 1], got {tau}")
        if int(spatial_merge_size) < 1:
            raise ValueError(
                f"spatial_merge_size must be a positive integer, got {spatial_merge_size}"
            )
        for key, value in (
            ("select", select),
            ("merge_mode", merge_mode),
            ("backend", backend),
        ):
            if value not in _CHOICES[key]:
                raise ValueError(f"{key} must be one of {_CHOICES[key]}, got {value!r}")

        self.tau = float(tau)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lam = float(lam)
        self.p = float(p)
        self.spatial_merge_size = int(spatial_merge_size)
        self.eps = float(eps)

        # Ablation switches (Table III).  Turning a term off gives the
        # remaining term of that group the full weight, which is the same as
        # setting the corresponding balance coefficient to its extreme; see the
        # docstrings of the two scoring stages below.
        self.use_sw = bool(use_sw)
        self.use_tw = bool(use_tw)
        self.use_sd = bool(use_sd)
        self.use_td = bool(use_td)
        if not (self.use_sw or self.use_tw or self.use_sd or self.use_td):
            raise ValueError("at least one of SW/TW/SD/TD must remain enabled")

        # Table IV variants.
        self.keep_isolated = bool(keep_isolated)
        self.propagate = bool(propagate)
        self.select = str(select)
        self.merge_mode = str(merge_mode)
        # ``auto`` runs the heap-based host merge loop for the token counts
        # produced by a normal video and keeps the reference device loop for
        # very large inputs (where copying the token tensor to the host is the
        # smaller of the two costs).
        self.backend = str(backend)
        self.cpu_max_tokens = int(cpu_max_tokens)

        # Profiling (off by default: it synchronises the device at every stage).
        self.profile = bool(profile)
        self.profile_out = profile_out
        self.last_stats: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls, **defaults) -> "GraTok":
        """Build a compressor, letting ``GRATOK_*`` variables override defaults.

        Any keyword argument accepted by ``__init__`` may be given as a default;
        only the parameters whose environment variable is set are overridden.
        """
        kwargs = dict(defaults)
        for key, env_name in _ENV_KEYS.items():
            raw = os.environ.get(env_name)
            if raw is not None and raw.strip() != "":
                cast = _ENV_CASTS.get(key, float)
                try:
                    kwargs[key] = cast(raw.strip())
                except ValueError as exc:
                    raise ValueError(
                        f"{env_name}={raw!r} is not a valid {cast.__name__}"
                    ) from exc
        for key, env_name in _ENV_SWITCHES.items():
            raw = os.environ.get(env_name)
            if raw is not None and raw.strip() != "":
                kwargs[key] = raw.strip().lower() not in _TRUTHY
        for key, env_name in _ENV_CHOICES.items():
            raw = os.environ.get(env_name)
            if raw is not None and raw.strip() != "":
                value = raw.strip().lower()
                if value not in _CHOICES[key]:
                    raise ValueError(
                        f"{env_name}={raw!r} must be one of {_CHOICES[key]}"
                    )
                kwargs[key] = value
        if os.environ.get("GRATOK_PROFILE", "").strip() in _TRUTHY:
            kwargs["profile"] = True
        if os.environ.get("GRATOK_PROFILE_OUT"):
            kwargs["profile_out"] = os.environ["GRATOK_PROFILE_OUT"]
        return cls(**kwargs)

    def config(self) -> dict:
        """The effective configuration, for logging and for the run manifest."""
        return {
            "tau": self.tau,
            "alpha": self.alpha,
            "beta": self.beta,
            "lam": self.lam,
            "p": self.p,
            "spatial_merge_size": self.spatial_merge_size,
            "use_sw": self.use_sw,
            "use_tw": self.use_tw,
            "use_sd": self.use_sd,
            "use_td": self.use_td,
            "keep_isolated": self.keep_isolated,
            "propagate": self.propagate,
            "select": self.select,
            "merge_mode": self.merge_mode,
            "backend": self.backend,
            "cpu_max_tokens": self.cpu_max_tokens,
        }

    def __repr__(self) -> str:  # pragma: no cover - trivial
        args = ", ".join(f"{k}={v}" for k, v in self.config().items())
        return f"{type(self).__name__}({args})"

    # ------------------------------------------------------------------ #
    # stages
    # ------------------------------------------------------------------ #
    @staticmethod
    def _similarity(X_3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Intra-frame ``(T, N, N)`` and adjacent-frame ``(T-1, N, N)`` similarity."""
        T = X_3d.shape[0]
        X_norm = F.normalize(X_3d, p=2, dim=2)
        S_intra = torch.bmm(X_norm, X_norm.transpose(1, 2))
        if T > 1:
            S_inter = torch.bmm(X_norm[:-1], X_norm[1:].transpose(1, 2))
        else:
            S_inter = X_3d.new_empty((0,))
        return S_intra, S_inter

    def _adjacency(
        self, S_intra: torch.Tensor, S_inter: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Thresholded adjacency; ``A_intra`` carries a self-loop on its diagonal."""
        T, N = S_intra.shape[:2]
        A_intra = (S_intra > self.tau).to(S_intra.dtype)
        diag = torch.eye(N, device=S_intra.device, dtype=torch.bool)
        A_intra = torch.where(diag.unsqueeze(0), torch.ones_like(A_intra), A_intra)
        if S_inter.numel() > 0:
            A_inter = (S_inter > self.tau).to(S_inter.dtype)
        else:
            A_inter = S_inter
        return A_intra, A_inter

    @staticmethod
    def _eight_neighbour_offsets():
        for dh in (-1, 0, 1):
            for dw in (-1, 0, 1):
                if dh == 0 and dw == 0:
                    continue
                yield dh, dw

    @staticmethod
    def _sync_mark(X: torch.Tensor) -> float:
        """Wall-clock mark that is meaningful on an asynchronous device."""
        if X.is_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def _spatial_similarity(self, S_intra: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Mean similarity of every token to its 8 grid neighbours, shape ``(T, N)``.

        Boundary tokens use the neighbours that exist (3 at a corner, 5 on an
        edge), which is what the paper's Eq. (10) states.
        """
        T = S_intra.shape[0]
        S5 = S_intra.view(T, H, W, H, W)
        neighbour_sum = torch.zeros((T, H, W), device=S_intra.device, dtype=S_intra.dtype)
        neighbour_cnt = torch.zeros((H, W), device=S_intra.device, dtype=S_intra.dtype)
        h_idx = torch.arange(H, device=S_intra.device).view(H, 1).expand(H, W)
        w_idx = torch.arange(W, device=S_intra.device).view(1, W).expand(H, W)
        for dh, dw in self._eight_neighbour_offsets():
            src = (slice(max(0, -dh), min(H, H - dh)), slice(max(0, -dw), min(W, W - dw)))
            hh = h_idx[src[0], src[1]]
            ww = w_idx[src[0], src[1]]
            neighbour_sum[:, src[0], src[1]] += S5[:, hh, ww, hh + dh, ww + dw]
            neighbour_cnt[src[0], src[1]] += 1.0
        neighbour_mean = neighbour_sum / neighbour_cnt.clamp(min=1.0)
        return neighbour_mean.reshape(T, H * W)

    def _spatio_temporal_weight(
        self,
        S_intra: torch.Tensor,
        S_inter: torch.Tensor,
        N: int,
        grid_h: int,
        grid_w: int,
    ) -> Optional[torch.Tensor]:
        """``w_st`` from Eqs. (10)-(13), or ``None`` if both terms are disabled.

        With ``use_sw=False`` the whole weight comes from the temporal term, and
        with ``use_tw=False`` from the spatial term, which is the same as
        pushing ``alpha`` to 0 or 1.  This is what the leave-one-out rows of the
        ablation table correspond to.
        """
        if not (self.use_sw or self.use_tw):
            return None
        T = S_intra.shape[0]
        spatial_sim = self._spatial_similarity(S_intra, grid_h, grid_w)
        w_s = spatial_sim / (spatial_sim.max() + self.eps)

        if T > 1:
            idx = torch.arange(N, device=S_intra.device)
            diag_values = S_inter[:, idx, idx]  # similarity of a patch to itself one frame later
            w_t = torch.zeros((T, N), device=S_intra.device, dtype=S_intra.dtype)
            w_t[0] = 1.0 - diag_values[0]
            if T > 2:
                w_t[1:-1] = (1.0 - diag_values[:-1] + 1.0 - diag_values[1:]) / 2.0
            w_t[-1] = 1.0 - diag_values[-1]
            w_t = w_t / (w_t.max() + self.eps)
        else:
            w_t = torch.zeros((1, N), device=S_intra.device, dtype=S_intra.dtype)

        if self.use_sw and self.use_tw:
            return self.alpha * (1.0 - w_s) + (1.0 - self.alpha) * w_t
        if self.use_sw:
            return 1.0 - w_s
        return w_t

    def _dependency(
        self, A_intra: torch.Tensor, A_inter: torch.Tensor, N: int, T: int
    ) -> Optional[torch.Tensor]:
        """``dep_st`` from Eqs. (14)-(18), or ``None`` if both terms are disabled.

        ``dep_s`` is the row sum of ``A_intra`` (the self-loop is included, so
        the degree of an isolated token is 1 rather than 0).  ``dep_t`` sums the
        inter-frame edges incident on the token's *own* frame: the in-degree
        from the previous frame plus the out-degree to the next frame.

        ``use_sd`` / ``use_td`` drop one of the two degrees, which is the same
        as pushing ``beta`` to 0 or 1.
        """
        if not (self.use_sd or self.use_td):
            return None
        dep_s = A_intra.sum(dim=2)
        dep_s = dep_s / (dep_s.max() + self.eps)

        dep_t = torch.zeros((T, N), device=A_intra.device, dtype=A_intra.dtype)
        if T > 1 and A_inter.numel() > 0:
            in_from_prev = A_inter.sum(dim=1)  # (T-1, N): edges t-1 -> t
            out_to_next = A_inter.sum(dim=2)  # (T-1, N): edges t   -> t+1
            dep_t[1:] += in_from_prev  # token in frame t receives from frame t-1
            dep_t[:-1] += out_to_next  # token in frame t sends to frame t+1
        dep_t = dep_t / (dep_t.max() + self.eps)

        if self.use_sd and self.use_td:
            return self.beta * dep_s + (1.0 - self.beta) * dep_t
        if self.use_sd:
            return dep_s
        return dep_t

    @staticmethod
    def _lookups(
        S_intra: torch.Tensor,
        S_inter: torch.Tensor,
        A_intra: torch.Tensor,
        A_inter: torch.Tensor,
        T: int,
        N: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense per-token similarity/adjacency tables used by the merge loop.

        The original code filled these row by row in Python, i.e. ``2 * T * N``
        GPU slicing operations per video.  The block structure is the same, so
        the tables are built block by block instead -- ``T`` slices per table.
        (A top-K sparse neighbour list would shrink the ``(T*N)^2`` memory term;
        that is left to future work, as stated in the paper.)
        """
        device = S_intra.device
        sim_lookup = torch.zeros((T * N, T * N), device=device, dtype=S_intra.dtype)
        for t in range(T):
            rows = slice(t * N, (t + 1) * N)
            sim_lookup[rows, rows] = S_intra[t]
        if T > 1 and S_inter.numel() > 0:
            for t in range(T - 1):
                here = slice(t * N, (t + 1) * N)
                nxt = slice((t + 1) * N, (t + 2) * N)
                sim_lookup[here, nxt] = S_inter[t]
                sim_lookup[nxt, here] = S_inter[t].transpose(0, 1)

        adj_lookup = torch.zeros((T * N, T * N), device=device, dtype=torch.bool)
        for t in range(T):
            rows = slice(t * N, (t + 1) * N)
            adj_lookup[rows, rows] = A_intra[t].bool()
        if T > 1 and A_inter.numel() > 0:
            for t in range(T - 1):
                here = slice(t * N, (t + 1) * N)
                nxt = slice((t + 1) * N, (t + 2) * N)
                adj_lookup[here, nxt] = A_inter[t].bool()
                adj_lookup[nxt, here] = A_inter[t].transpose(0, 1).bool()
        return sim_lookup, adj_lookup

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #
    def _select_token(
        self, token_scores: torch.Tensor, pool: torch.Tensor, inf_score: torch.Tensor
    ) -> int:
        """Which token to reduce next.

        ``importance`` is the paper's rule (the lowest score wins); ``random``
        and ``sequential`` are the two control variants of Table IV.  Tokens
        outside ``pool`` are inactive or protected and are never chosen.
        """
        if self.select == "importance":
            masked_scores = torch.where(pool, token_scores, inf_score)
            return int(torch.argmin(masked_scores).item())
        indices = torch.where(pool)[0]
        if self.select == "random":
            pick = torch.randint(indices.numel(), (1,), device=indices.device)
            return int(indices[int(pick.item())].item())
        if self.select == "sequential_last":
            return int(indices[-1].item())
        return int(indices[0].item())

    def _reduce_gpu(
        self,
        X: torch.Tensor,
        sim_lookup: torch.Tensor,
        adj_lookup: torch.Tensor,
        scores: torch.Tensor,
        M: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Reference merge loop: every iteration scans all ``N_total`` scores.

        Kept as the fallback for very large inputs and as the reference the CPU
        fast path is checked against (see ``tools/verify_backends.py``).
        """
        device = X.device
        N_total = X.shape[0]
        tokens = X.clone()
        token_scores = scores.clone()
        represented = torch.ones(N_total, device=X.device, dtype=torch.float32)
        active = torch.ones(N_total, device=X.device, dtype=torch.bool)
        # Tokens that have no surviving neighbour.  With ``keep_isolated`` they
        # are pinned: they stay in the output but can never be selected again,
        # so the loop is free to keep reducing the rest.  Otherwise they are
        # deleted and their mass is accounted for below.
        frozen = torch.zeros(N_total, device=X.device, dtype=torch.bool)

        n_protected_without_neighbour = 0
        dropped_without_neighbour = 0
        # Mass (number of original tokens) that left the graph through the
        # drop-without-neighbour path.  It is *not* represented by any kept
        # token, so the conservation check below has to account for it -- that
        # is exactly the quantity JkNx asks about.
        dropped_mass = 0
        # Mass that left the graph through a merge with ``propagate=False``
        # (Table IV "Sim Merge"), where nothing is written back.
        discarded_mass = 0
        n_merged = 0
        inf_score = torch.tensor(float("inf"), device=X.device)
        removed = 0
        target_removals = N_total - M
        while removed < target_removals:
            pool = active & ~frozen
            if not bool(pool.any()):
                break  # only protected (isolated) tokens are left
            min_idx = self._select_token(token_scores, pool, inf_score)

            # Candidates: active neighbours, never the token itself.  The
            # self-loop stored on the diagonal of A_intra is a degree device and
            # must not turn a merge into a deletion.
            candidate_mask = adj_lookup[min_idx] & active
            candidates = torch.where(candidate_mask)[0]
            candidates = candidates[candidates != min_idx]
            if candidates.numel() == 0:
                if self.keep_isolated:
                    frozen[min_idx] = True
                    n_protected_without_neighbour += 1
                    continue
                active[min_idx] = False
                dropped_without_neighbour += 1
                dropped_mass += int(represented[min_idx].item())
                removed += 1
                continue

            sim_values = sim_lookup[min_idx, candidates]
            best_pos = int(torch.argmax(sim_values).item())
            best_neighbor = int(candidates[best_pos].item())
            best_sim = sim_values[best_pos].clamp(min=0.0)  # keep the merge weight positive

            n_i = represented[min_idx]
            n_j = represented[best_neighbor]
            if self.propagate:
                if self.merge_mode == "sum":
                    # Table IV "Sum Merge": plain addition, no weighting.
                    tokens[best_neighbor] = tokens[min_idx] + tokens[best_neighbor]
                    token_scores[best_neighbor] = (
                        token_scores[best_neighbor] + token_scores[min_idx]
                    )
                else:
                    # Feature merge: similarity- and count-weighted average.
                    tokens[best_neighbor] = (
                        best_sim * n_i * tokens[min_idx] + n_j * tokens[best_neighbor]
                    ) / (best_sim * n_i + n_j)
                    # Score merge: count-weighted average (not an accumulation).
                    token_scores[best_neighbor] = (
                        n_j * token_scores[best_neighbor] + n_i * token_scores[min_idx]
                    ) / (n_j + n_i)
                represented[best_neighbor] = n_i + n_j
            else:
                # Table IV "Sim Merge": the source leaves the set but nothing is
                # written back, so its mass leaves the accounting.
                discarded_mass += int(n_i.item())
            active[min_idx] = False
            removed += 1
            n_merged += 1

        kept_ids = torch.where(active)[0]
        return tokens, kept_ids, represented[kept_ids], {
            "n_merged": n_merged,
            "n_protected_without_neighbour": n_protected_without_neighbour,
            "dropped_without_neighbour": dropped_without_neighbour,
            "dropped_mass": dropped_mass,
            "discarded_mass": discarded_mass,
        }

    @staticmethod
    def _csr(
        sim_lookup: torch.Tensor, adj_lookup: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Neighbour lists and edge weights grouped by source token (CSR).

        ``cols[offsets[i]:offsets[i + 1]]`` is the neighbour list of token ``i``
        and ``vals[...]`` holds the matching similarity, both in ascending index
        order -- the same candidate order the reference loop produces with
        ``torch.where(adj_lookup[i] & active)``.  Built once per video, which is
        what removes the per-iteration scan of the whole adjacency row.
        """
        N_total = adj_lookup.shape[0]
        rows, cols = adj_lookup.nonzero(as_tuple=True)
        vals = sim_lookup[rows, cols]
        counts = torch.bincount(rows, minlength=N_total)
        offsets = torch.zeros(N_total + 1, dtype=torch.long, device=rows.device)
        offsets[1:] = counts.cumsum(0)
        return cols, vals, offsets

    @staticmethod
    def _store(values, round_to_bf16: bool):
        """Emulate the dtype rounding the device loop performs on assignment.

        ``tokens[j] = <float32 expression>`` writes into a bfloat16 tensor, so
        the device loop rounds every merged value once, with round-to-nearest-
        even.  Doing the same on the host is what keeps the two paths
        bit-comparable for a bfloat16 backbone.
        """
        import numpy as np

        arr = np.ascontiguousarray(values, dtype=np.float32)
        if not round_to_bf16:
            return arr
        bits = arr.view(np.uint32)
        lsb = (bits >> np.uint32(16)) & np.uint32(1)
        rounded = bits + np.uint32(0x7FFF) + lsb
        return (rounded & np.uint32(0xFFFF0000)).view(np.float32)

    def _reduce_cpu(
        self,
        X: torch.Tensor,
        sim_lookup: torch.Tensor,
        adj_lookup: torch.Tensor,
        scores: torch.Tensor,
        M: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Merge loop on the host, ordered by a binary heap.

        The reference loop rescans all ``N_total`` scores twice per removed
        token (``argmin`` plus the adjacency row) and pays a device
        synchronisation for every scalar it reads back: ``O((N-M) N)`` work and
        ``O(N-M)`` syncs.  Here the ordering lives in a lazy-deletion heap and
        the adjacency in a CSR neighbour list, so an iteration costs
        ``O(log N)`` plus the surviving neighbours, and the only device traffic
        is the token tensor itself.

        The arithmetic mirrors the device loop expression for expression and
        rounds the merged values to the same dtype, so both paths agree
        (``tools/verify_backends.py``).
        """
        import heapq

        import numpy as np

        device = X.device
        round_to_bf16 = X.dtype == torch.bfloat16
        csr_cols, csr_vals, csr_offsets = self._csr(sim_lookup, adj_lookup)
        cols = csr_cols.cpu().numpy()
        offsets = csr_offsets.cpu().numpy()
        # widening bfloat16 edge weights to float32 is lossless and keeps the
        # comparisons identical to the device loop's
        vals = csr_vals.float().cpu().numpy()
        N_total = X.shape[0]

        tokens = X.detach().float().cpu().numpy()
        token_scores = scores.detach().float().cpu().numpy()
        represented = np.ones(N_total, dtype=np.float32)
        active = np.ones(N_total, dtype=np.bool_)
        frozen = np.zeros(N_total, dtype=np.bool_)

        n_protected_without_neighbour = 0
        dropped_without_neighbour = 0
        dropped_mass = 0
        discarded_mass = 0
        n_merged = 0

        heap = [(float(token_scores[i]), i) for i in range(N_total)]
        heapq.heapify(heap)
        removed = 0
        target_removals = N_total - M
        while removed < target_removals:
            min_idx = -1
            while heap:
                score, i = heapq.heappop(heap)
                if not active[i] or frozen[i]:
                    continue  # already merged away or pinned
                if score != float(token_scores[i]):
                    continue  # stale entry: updated after it was pushed
                min_idx = i
                break
            if min_idx < 0:
                break  # only protected (isolated) tokens are left

            lo, hi = int(offsets[min_idx]), int(offsets[min_idx + 1])
            cand = cols[lo:hi]
            keep = active[cand] & (cand != min_idx)
            if not keep.any():
                if self.keep_isolated:
                    frozen[min_idx] = True
                    n_protected_without_neighbour += 1
                    continue
                active[min_idx] = False
                dropped_without_neighbour += 1
                dropped_mass += int(represented[min_idx])
                removed += 1
                continue

            cand = cand[keep]
            sim_values = vals[lo:hi][keep]
            best_pos = int(np.argmax(sim_values))
            best_neighbor = int(cand[best_pos])
            best_sim = max(0.0, float(sim_values[best_pos]))

            n_i = float(represented[min_idx])
            n_j = float(represented[best_neighbor])
            if self.propagate:
                if self.merge_mode == "sum":
                    # Table IV "Sum Merge": plain addition, no weighting.
                    tokens[best_neighbor] = self._store(
                        tokens[min_idx] + tokens[best_neighbor], round_to_bf16
                    )
                    token_scores[best_neighbor] = self._store(
                        token_scores[best_neighbor] + token_scores[min_idx], round_to_bf16
                    )
                else:
                    # Feature merge: similarity- and count-weighted average.
                    tokens[best_neighbor] = self._store(
                        (best_sim * n_i * tokens[min_idx] + n_j * tokens[best_neighbor])
                        / (best_sim * n_i + n_j),
                        round_to_bf16,
                    )
                    # Score merge: count-weighted average (not an accumulation).
                    token_scores[best_neighbor] = self._store(
                        (n_j * token_scores[best_neighbor] + n_i * token_scores[min_idx])
                        / (n_j + n_i),
                        round_to_bf16,
                    )
                represented[best_neighbor] = n_i + n_j
                heapq.heappush(
                    heap, (float(token_scores[best_neighbor]), best_neighbor)
                )
            else:
                # Table IV "Sim Merge": the source leaves the set but nothing is
                # written back, so its mass leaves the accounting.
                discarded_mass += int(n_i)
            active[min_idx] = False
            removed += 1
            n_merged += 1

        kept_ids = np.nonzero(active)[0]
        return (
            torch.from_numpy(tokens).to(device).to(X.dtype),
            torch.from_numpy(kept_ids).to(device),
            torch.from_numpy(represented[kept_ids]).to(device),
            {
                "n_merged": n_merged,
                "n_protected_without_neighbour": n_protected_without_neighbour,
                "dropped_without_neighbour": dropped_without_neighbour,
                "dropped_mass": dropped_mass,
                "discarded_mass": discarded_mass,
            },
        )

    def _reduce(
        self,
        X: torch.Tensor,
        sim_lookup: torch.Tensor,
        adj_lookup: torch.Tensor,
        scores: torch.Tensor,
        M: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Dispatch the merge loop (see ``GRATOK_BACKEND``/``GRATOK_CPU_MAX_TOKENS``)."""
        N_total = X.shape[0]
        if self.backend == "cpu" or (
            self.backend == "auto"
            and X.is_cuda
            and 256 <= N_total <= self.cpu_max_tokens
        ):
            return self._reduce_cpu(X, sim_lookup, adj_lookup, scores, M)
        return self._reduce_gpu(X, sim_lookup, adj_lookup, scores, M)

    def compress(
        self, X: torch.Tensor, grid_thw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress the visual tokens of one video.

        Args:
            X: visual tokens of shape ``(N_total, D)``, frame-major.
            grid_thw: ``(t, h, w)`` of the token grid *before* patch pooling.

        Returns:
            ``(tokens, kept_ids)`` where ``tokens`` keeps the original
            ``(N_total, D)`` layout -- surviving rows hold the merged features,
            removed rows are stale and must be filtered out by the caller using
            ``kept_ids`` (both call sites already do this).
        """
        T = int(grid_thw[0])
        merge = self.spatial_merge_size
        self.last_stats = None
        H = int(grid_thw[1]) // merge
        W = int(grid_thw[2]) // merge
        N_total, D = X.shape
        if T <= 0 or N_total % T != 0:
            raise ValueError(f"cannot split {N_total} tokens into {T} frames")
        N = N_total // T
        if H * W != N:
            raise ValueError(
                f"token grid {H}x{W}={H * W} does not match {N} tokens per frame; "
                f"check spatial_merge_size ({merge}) and grid_thw={tuple(int(v) for v in grid_thw)}"
            )
        M = max(1, int(self.p * N_total))
        if M >= N_total:
            return X.clone(), torch.arange(N_total, device=X.device)

        prof = self.profile
        marks = []
        if prof:
            base_alloc = torch.cuda.memory_allocated() if X.is_cuda else 0
            if X.is_cuda:
                # The compressor's own peak has to be measured from here: the
                # caller's high-water mark would otherwise be included.
                torch.cuda.reset_peak_memory_stats()
            marks.append(self._sync_mark(X))

        X_3d = X.view(T, N, D)
        S_intra, S_inter = self._similarity(X_3d)
        if prof:
            marks.append(self._sync_mark(X))
        A_intra, A_inter = self._adjacency(S_intra, S_inter)
        if prof:
            marks.append(self._sync_mark(X))

        w_st = self._spatio_temporal_weight(S_intra, S_inter, N, H, W)
        dep_st = self._dependency(A_intra, A_inter, N, T)
        if w_st is None and dep_st is None:
            raise ValueError(
                "all four terms (SW, TW, SD, TD) are disabled; there is nothing to score"
            )
        if w_st is None:            # dependency only
            scores = (1.0 - dep_st).reshape(-1)
        elif dep_st is None:        # saliency only
            scores = w_st.reshape(-1)
        else:
            scores = (self.lam * w_st + (1.0 - self.lam) * (1.0 - dep_st)).reshape(-1)
        if prof:
            marks.append(self._sync_mark(X))

        sim_lookup, adj_lookup = self._lookups(S_intra, S_inter, A_intra, A_inter, T, N)
        if prof:
            marks.append(self._sync_mark(X))

        tokens, kept_ids, kept_counts, counters = self._reduce(
            X, sim_lookup, adj_lookup, scores, M
        )
        if prof:
            marks.append(self._sync_mark(X))

        n_merged = counters["n_merged"]
        n_protected_without_neighbour = counters["n_protected_without_neighbour"]
        dropped_without_neighbour = counters["dropped_without_neighbour"]
        dropped_mass = counters["dropped_mass"]
        discarded_mass = counters["discarded_mass"]
        if prof:
            marks.append(self._sync_mark(X))

        # Invariants: cheap to check, and they catch exactly the class of silent
        # error that made the two old implementations diverge.  With
        # ``keep_isolated`` the output can exceed the target M, hence ``>=``.
        assert kept_ids.numel() >= M, (
            f"expected at least {M} kept tokens, got {kept_ids.numel()}"
        )
        kept_mass = int(kept_counts.sum().item())
        assert kept_mass + dropped_mass + discarded_mass == N_total, (
            f"token mass is not conserved: {kept_mass} kept + {dropped_mass} "
            f"dropped + {discarded_mass} discarded != {N_total}"
        )
        assert bool((kept_counts > 0).all()), "non-positive represented-token count"

        if prof:
            n_out = int(kept_ids.numel())
            self.last_stats = {
                "T": T, "N": N, "grid_h": H, "grid_w": W, "D": D,
                "n_tokens_in": N_total,
                "n_tokens_target": M,
                "n_tokens_out": n_out,
                "n_merged": n_merged,
                "n_protected_without_neighbour": n_protected_without_neighbour,
                "n_dropped_without_neighbour": dropped_without_neighbour,
                "n_kept_from_merging": int((kept_counts > 1).sum().item()),
                "max_represented": float(kept_counts.max().item()),
                "ms_similarity": 1e3 * (marks[1] - marks[0]),
                "ms_graph": 1e3 * (marks[2] - marks[1]),
                "ms_score": 1e3 * (marks[3] - marks[2]),
                # kept for backwards compatibility with earlier aggregations
                "ms_graph_and_scoring": 1e3 * (marks[3] - marks[1]),
                "ms_lookup": 1e3 * (marks[4] - marks[3]),
                "ms_merge": 1e3 * (marks[5] - marks[4]),
                "ms_total": 1e3 * (marks[5] - marks[0]),
                "peak_mem_bytes": (
                    torch.cuda.max_memory_allocated() - base_alloc if X.is_cuda else 0
                ),
                "config": self.config(),
            }
            if self.profile_out:
                with open(self.profile_out, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(self.last_stats) + "\n")

        if dropped_without_neighbour:
            fraction = dropped_without_neighbour / N_total
            log = logger.warning if fraction > 0.01 else logger.debug
            log(
                "GraTok: %d of %d tokens (%.2f%%) had no surviving neighbour and were "
                "dropped without merging (p=%.2f, tau=%.2f)",
                dropped_without_neighbour,
                N_total,
                100.0 * fraction,
                self.p,
                self.tau,
            )
        if n_protected_without_neighbour:
            logger.debug(
                "GraTok: %d of %d tokens (%.2f%%) had no surviving neighbour and were "
                "kept unmerged (p=%.2f, tau=%.2f)",
                n_protected_without_neighbour,
                N_total,
                100.0 * n_protected_without_neighbour / N_total,
                self.p,
                self.tau,
            )
        return tokens, kept_ids

    # Backwards-compatible alias for call sites written against the old name.
    refine_tokens_gpu_optimized2 = compress
