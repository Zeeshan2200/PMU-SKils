"""
Signal Reconstruction Comparison: PMU Interpolation vs. Compressed Sensing vs. AI
==================================================================================

Compares three approaches to reconstructing a sparse multi-tone power system
voltage waveform from limited/degraded measurements:

  1. PMU   - anti-alias filtered, uniformly sampled at 30 Hz, linearly interpolated
  2. CS    - compressed sensing recovery (Orthogonal Matching Pursuit) from random,
             sub-Nyquist samples of the raw signal
  3. AI    - a small MLP regressor trained ONLY on PMU-rate samples (no
             ground-truth leakage), then evaluated at full time resolution

Produces 5 figures matching the paper:
  Figure_1.png - 2x2 dashboard (original / PMU / CS / AI time-domain traces)
  Figure_2.png - overlay comparison of all traces
  Figure_3.png - frequency-spectrum comparison (2D overlay)
  Figure_4.png - CS RMSE sensitivity to measurement ratio M/N
  Figure_5.png - CS/PMU RMSE sensitivity to noise level

Requires: numpy, scipy, matplotlib, scikit-learn
    pip install numpy scipy matplotlib scikit-learn
"""

import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# ======================================================================
# 0. GLOBAL STYLE - large, print-legible fonts (used for every figure)
# ======================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 15,
    'axes.titlesize': 17,
    'axes.titleweight': 'bold',
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.titlesize': 18,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.edgecolor': '#444444',
})
DPI = 300
COL = {'orig': '#2c3e50', 'pmu': '#27ae60', 'cs': '#e74c3c', 'ai': '#3498db'}

# ======================================================================
# 1. PARAMETERS
# ======================================================================
Fs = 1000                          # sampling frequency (Hz)
T = 2                              # observation window (s)
t = np.arange(0, T, 1 / Fs)
N = len(t)

f_true = np.array([1, 12, 18, 20, 30])     # true tone frequencies (Hz)
Amp = np.array([1, 0.4, 0.3, 0.7, 0.5])    # true tone amplitudes
sigma_noise = 0.3                          # AWGN standard deviation

pmu_rate = 30                      # PMU report rate (Hz)
cutoff = 14                        # anti-alias filter cutoff (Hz), just below
                                    # the PMU Nyquist frequency (pmu_rate/2 = 15 Hz)
L = 101                            # FIR filter length (taps)

cs_ratio = 0.10                    # CS measurement ratio M/N used in the main comparison
K = 5                              # OMP sparsity (matches the 5 true tones)
freq_grid_step = 0.5               # CS dictionary spacing = 1/T (aligned to the
                                    # discrete Fourier resolution of the window)


# ======================================================================
# 2. SIGNAL GENERATION
# ======================================================================
def generate_signal(sigma, seed=42):
    """Five-tone test signal + additive white Gaussian noise."""
    Va = np.zeros(N)
    for f_i, A_i in zip(f_true, Amp):
        Va += A_i * np.cos(2 * np.pi * f_i * t)
    rng = np.random.RandomState(seed)
    Va_noisy = Va + sigma * rng.randn(N)
    return Va, Va_noisy


# ======================================================================
# 3. PMU PATH - anti-alias filter + uniform sampling + interpolation
# ======================================================================
def pmu_reconstruct(Va_noisy, cutoff=cutoff, pmu_rate=pmu_rate, L=L):
    """
    Apply anti-alias low-pass filter, downsample to PMU rate, then
    linearly interpolate back to full time resolution.
    """
    n = np.arange(-(L - 1) / 2, (L - 1) / 2 + 1)
    x = 2 * cutoff * n / Fs
    h = np.zeros_like(x)
    zero_idx = (x == 0)
    h[~zero_idx] = (2 * cutoff / Fs) * (np.sin(np.pi * x[~zero_idx]) / (np.pi * x[~zero_idx]))
    h[zero_idx] = 2 * cutoff / Fs
    w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(L) / (L - 1))   # Hamming window
    h = h * w
    h = h / np.sum(h)                                              # unit DC gain

    Va_filtered = sg.convolve(Va_noisy, h, mode='same')
    step = int(np.round(Fs / pmu_rate))
    Va_pmu = Va_filtered[::step]
    t_pmu = t[::step]
    Va_pmu_interp = np.interp(t, t_pmu, Va_pmu)
    return Va_pmu, t_pmu, Va_pmu_interp


# ======================================================================
# 4. COMPRESSED SENSING PATH - Orthogonal Matching Pursuit (OMP)
# ======================================================================
def omp(A, y, K):
    """Orthogonal Matching Pursuit: recover a K-sparse coefficient vector theta
    such that A @ theta ~= y."""
    r = y.copy()
    idx_set = []
    theta = np.zeros(A.shape[1])
    theta_ls = np.array([])
    for _ in range(K):
        proj = np.abs(A.T @ r)
        idx = np.argmax(proj)
        if idx not in idx_set:
            idx_set.append(idx)
        else:
            break                       # no new atom found; stop early
        A_sel = A[:, idx_set]
        theta_ls = np.linalg.pinv(A_sel) @ y
        r = y - A_sel @ theta_ls
    if idx_set:
        theta[idx_set] = theta_ls
    return theta


def build_cs_dictionary(freq_step=freq_grid_step, f_max=60):
    """Cosine dictionary aligned to the 1/T discrete Fourier grid."""
    freqs = np.arange(0, f_max + freq_step, freq_step)
    Psi = np.zeros((N, len(freqs)))
    for k, f_k in enumerate(freqs):
        Psi[:, k] = np.cos(2 * np.pi * f_k * t)
    return freqs, Psi


def cs_reconstruct(Va_noisy, Psi, M_ratio=cs_ratio, K=K, seed=42):
    rng = np.random.RandomState(seed)
    M = int(np.round(M_ratio * N))
    idx = rng.choice(N, M, replace=False)
    Phi = np.zeros((M, N))
    for i in range(M):
        Phi[i, idx[i]] = 1
    A_cs = Phi @ Psi
    y = Va_noisy[idx]
    theta = omp(A_cs, y, K)
    Va_cs = Psi @ theta
    return Va_cs, idx, y, theta


# ======================================================================
# 5. AI PATH - MLP trained ONLY on PMU-rate samples (no ground-truth leakage)
# ======================================================================
def ai_reconstruct(Va_noisy):
    """
    Train an MLP regressor on the raw noisy signal sampled uniformly at the
    PMU report rate (no anti-alias filtering). The network must learn to
    interpolate and (implicitly) handle aliased high-frequency components
    purely from the time-domain samples at 30 Hz.

    This is a distinct approach from the PMU path: the PMU applies an explicit
    analog-style anti-alias filter before sampling; the AI receives the raw
    (potentially aliased) samples and relies on its learned inductive bias.
    """
    step = int(np.round(Fs / pmu_rate))
    t_pmu = t[::step]
    Va_pmu_raw = Va_noisy[::step]  # raw noisy samples, NO anti-alias filtering

    t_mean, t_std = t_pmu.mean(), t_pmu.std()
    t_pmu_scaled = (t_pmu - t_mean) / t_std
    t_scaled = (t - t_mean) / t_std

    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='tanh',
                        max_iter=20000, random_state=0)
    mlp.fit(t_pmu_scaled.reshape(-1, 1), Va_pmu_raw)
    Va_ai = mlp.predict(t_scaled.reshape(-1, 1))
    return Va_ai


# ======================================================================
# 6. EVALUATION METRIC
# ======================================================================
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


# ======================================================================
# 7. RUN THE MAIN COMPARISON (matches the paper's headline numbers)
# ======================================================================
def main():
    print("Running main comparison...")
    Va, Va_noisy = generate_signal(sigma_noise)
    Va_pmu, t_pmu, Va_pmu_interp = pmu_reconstruct(Va_noisy)
    freqs, Psi = build_cs_dictionary()
    Va_cs, cs_idx, cs_y, cs_theta = cs_reconstruct(Va_noisy, Psi)
    Va_ai = ai_reconstruct(Va_noisy)

    # RMSE computed against the CLEAN ground-truth signal (Va), not the noisy
    # version. This measures how well each method recovers the true underlying
    # waveform, which is the standard metric in signal reconstruction.
    rmse_pmu = rmse(Va, Va_pmu_interp)
    rmse_cs = rmse(Va, Va_cs)
    rmse_ai = rmse(Va, Va_ai)

    print("\n" + "=" * 30)
    print(f"  RMSE PMU: {rmse_pmu:.4f}")
    print(f"  RMSE CS : {rmse_cs:.4f}")
    print(f"  RMSE AI : {rmse_ai:.4f}")
    print("=" * 30 + "\n")


    # ======================================================================
    # FIGURE 1: 2x2 time-domain dashboard
    # ======================================================================
    fig1, axs1 = plt.subplots(2, 2, figsize=(15, 9))
    fig1.suptitle('Signal Reconstruction Methods Breakdown', fontweight='bold', y=0.99)

    axs1[0, 0].plot(t, Va_noisy, color=COL['orig'], linewidth=1.6)
    axs1[0, 0].set_title('Original Signal (with Noise)')
    axs1[0, 0].set_ylabel('Amplitude')

    axs1[0, 1].plot(t, Va_pmu_interp, color=COL['pmu'], linewidth=2.6)
    axs1[0, 1].scatter(t_pmu, Va_pmu, color='#1e8449', s=45, zorder=3,
                        label='PMU Samples', edgecolor='white', linewidth=0.8)
    axs1[0, 1].set_title('PMU Signal (Filtered + Sampled)')
    axs1[0, 1].legend(loc='upper right', framealpha=0.9)

    axs1[1, 0].plot(t, Va_cs, color=COL['cs'], linewidth=2.6)
    axs1[1, 0].scatter(t[cs_idx], cs_y, color='#c0392b', alpha=0.45, s=22,
                        zorder=2, label='Random CS Samples')
    axs1[1, 0].set_title('Compressed Sensing Reconstruction')
    axs1[1, 0].set_xlabel('Time (s)')
    axs1[1, 0].set_ylabel('Amplitude')
    axs1[1, 0].legend(loc='upper right', framealpha=0.9)

    axs1[1, 1].plot(t, Va_ai, color=COL['ai'], linewidth=2.6)
    axs1[1, 1].set_title('AI Reconstructed Signal (MLP on Raw PMU-rate Samples)')
    axs1[1, 1].set_xlabel('Time (s)')

    for ax in axs1.flat:
        ax.tick_params(width=1.2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('Figure_1.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("Saved Figure_1.png (time-domain dashboard)")


    # ======================================================================
    # FIGURE 2: overlay comparison
    # ======================================================================
    plt.figure(figsize=(13, 7))
    plt.plot(t, Va_noisy, color='#95a5a6', alpha=0.75, linewidth=1.8, label='Original (Noisy)')
    plt.plot(t, Va_cs, color=COL['cs'], linewidth=2.8, linestyle='--', label='Compressed Sensing (CS)')
    plt.plot(t, Va_ai, color=COL['ai'], linewidth=2.8, linestyle='-.', label='AI Model')
    plt.plot(t, Va_pmu_interp, color=COL['pmu'], linewidth=2.8, linestyle=':', label='PMU Interp')
    plt.title('All Signals Overlay Comparison', pad=14)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper right')
    plt.xlim([0, T])
    plt.tick_params(width=1.2)
    plt.tight_layout()
    plt.savefig('Figure_2.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("Saved Figure_2.png (overlay comparison)")


    # ======================================================================
    # FIGURE 3: frequency spectrum comparison (2D overlay)
    # ======================================================================
    f_axis = np.arange(N) * (Fs / N)
    mask = f_axis <= 60
    fN = f_axis[mask]
    X_orig = np.abs(np.fft.fft(Va_noisy))[mask]
    X_cs = np.abs(np.fft.fft(Va_cs))[mask]
    X_ai = np.abs(np.fft.fft(Va_ai))[mask]

    N_pmu = len(Va_pmu)
    f_pmu_full = np.arange(N_pmu) * (pmu_rate / N_pmu)
    X_pmu_full = np.abs(np.fft.fft(Va_pmu))
    mask_pmu = f_pmu_full <= 60
    X_pmu_interp = np.interp(fN, f_pmu_full[mask_pmu], X_pmu_full[mask_pmu], right=0)


    def norm(x):
        return x / x.max()


    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(fN, norm(X_orig), color=COL['orig'], linewidth=1.8, label='Original (Noisy)')
    ax.plot(fN, norm(X_pmu_interp), color=COL['pmu'], linewidth=2.6, linestyle=':',
            label='PMU (30 Hz)')
    ax.plot(fN, norm(X_cs), color=COL['cs'], linewidth=2.6, linestyle='--',
            label='Compressed Sensing')
    ax.plot(fN, norm(X_ai), color=COL['ai'], linewidth=2.6, linestyle='-.',
            label='AI (MLP)')
    ax.set_title('Frequency Spectrum Comparison (0\u201360 Hz)', pad=14)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized Magnitude')
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper right')
    ax.set_xlim(0, 60)
    ax.tick_params(width=1.2)
    plt.tight_layout()
    plt.savefig('Figure_3.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("Saved Figure_3.png (frequency spectrum comparison)")


    # ======================================================================
    # FIGURE 4: CS RMSE sensitivity to measurement ratio M/N
    # ======================================================================
    print("\nRunning sensitivity sweep: measurement ratio M/N ...")
    ratios = np.array([0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30])
    rmse_vs_M = []
    for r in ratios:
        Va_cs_r, *_ = cs_reconstruct(Va_noisy, Psi, M_ratio=r, K=K)
        rmse_vs_M.append(rmse(Va, Va_cs_r))
        print(f"  M/N={r*100:5.1f}%  ->  RMSE={rmse_vs_M[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ratios * 100, rmse_vs_M, marker='o', markersize=9, color=COL['cs'], linewidth=2.8)
    ax.set_xlabel('CS measurement ratio $M/N$ (%)')
    ax.set_ylabel('RMSE')
    ax.set_title('CS Reconstruction RMSE vs. Measurement Ratio\n($K=5$, $\\sigma=0.3$)')
    ax.axhline(rmse_pmu, color=COL['pmu'], linestyle='--', linewidth=2.2, label='PMU baseline (RMSE)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.tick_params(width=1.2)
    plt.tight_layout()
    plt.savefig('Figure_4.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("Saved Figure_4.png (measurement-ratio sensitivity)")


    # ======================================================================
    # FIGURE 5: RMSE sensitivity to noise level
    # ======================================================================
    print("\nRunning sensitivity sweep: noise level sigma ...")
    sigmas = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    rmse_cs_noise, rmse_pmu_noise = [], []
    for s in sigmas:
        Va_s, Va_noisy_s = generate_signal(s)
        Va_cs_s, *_ = cs_reconstruct(Va_noisy_s, Psi, M_ratio=cs_ratio, K=K)
        _, _, Va_pmu_interp_s = pmu_reconstruct(Va_noisy_s)
        # Both methods compared against the CLEAN ground truth Va_s for a fair
        # reconstruction-accuracy comparison (previously PMU used Va_s while CS
        # used Va_noisy_s, which artificially favoured PMU at higher noise).
        rmse_cs_noise.append(rmse(Va_s, Va_cs_s))
        rmse_pmu_noise.append(rmse(Va_s, Va_pmu_interp_s))
        print(f"  sigma={s:.1f}  ->  RMSE_CS={rmse_cs_noise[-1]:.4f}   RMSE_PMU={rmse_pmu_noise[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(sigmas, rmse_cs_noise, marker='o', markersize=9, color=COL['cs'],
            linewidth=2.8, label='Compressed Sensing')
    ax.plot(sigmas, rmse_pmu_noise, marker='s', markersize=9, color=COL['pmu'],
            linewidth=2.8, label='PMU Interpolation')
    ax.set_xlabel('Noise standard deviation $\\sigma$')
    ax.set_ylabel('RMSE')
    ax.set_title('Reconstruction RMSE vs. Noise Level')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.tick_params(width=1.2)
    plt.tight_layout()
    plt.savefig('Figure_5.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("Saved Figure_5.png (noise-level sensitivity)")

    print("\nAll 5 figures generated successfully.")


if __name__ == "__main__":
    main()
