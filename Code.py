import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# Set a professional, attractive plot style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# ==========================
# OMP FUNCTION
# ==========================
def omp(A, y, K):
    """Orthogonal Matching Pursuit.
    A   : (M x P) sensing/dictionary matrix
    y   : (M,)    measurements
    K   : sparsity level (number of atoms to select)
    """
    r = y.copy()
    idx_set = []
    theta = np.zeros(A.shape[1])
    theta_ls = np.array([])          # FIX: avoid NameError if K == 0
    for k in range(K):
        proj = np.abs(A.T @ r)
        idx = np.argmax(proj)
        if idx not in idx_set:
            idx_set.append(idx)
        else:
            # Same atom re-selected: residual is no longer shrinking in a
            # useful direction, so stop early instead of wasting iterations.
            break
        A_sel = A[:, idx_set]
        theta_ls = np.linalg.pinv(A_sel) @ y
        r = y - A_sel @ theta_ls
    if idx_set:
        theta[idx_set] = theta_ls
    return theta

# ==========================
# 1. PARAMETERS & CONFIG
# ==========================
Fs = 1000
T = 2
t = np.arange(0, T, 1 / Fs)
N = len(t)

f_true = np.array([1, 12, 18, 20, 30])   # Hz
A = np.array([1, 0.4, 0.3, 0.7, 0.5])    # amplitudes

pmu_rate = 30  # Hz  (PMU report rate)

# ==========================
# 2. SIGNAL GENERATION
# ==========================
Va = np.zeros(N)
for i in range(len(f_true)):
    Va += A[i] * np.cos(2 * np.pi * f_true[i] * t)

np.random.seed(42)                       # fixes the noise realization
Va += 0.3 * np.random.randn(N)

# ==========================
# 3. PMU ANTI-ALIASING FILTER + SAMPLING
# ==========================
# FIX: the anti-aliasing filter cutoff must be set relative to the PMU's own
# Nyquist frequency (pmu_rate / 2 = 15 Hz), not an arbitrary low value.
# A 5 Hz cutoff was throwing away the 12 Hz tone even though it is well
# below Nyquist and perfectly recoverable at a 30 Hz report rate.
# 14 Hz leaves a ~1 Hz guard band below the 15 Hz Nyquist so 18/20/30 Hz
# (which WOULD alias) are suppressed, while 12 Hz survives.
cutoff = 14
L = 101
n = np.arange(-(L - 1) / 2, (L - 1) / 2 + 1)
x = 2 * cutoff * n / Fs
h = np.zeros_like(x)
zero_idx = (x == 0)
h[~zero_idx] = (2 * cutoff / Fs) * (np.sin(np.pi * x[~zero_idx]) / (np.pi * x[~zero_idx]))
h[zero_idx] = 2 * cutoff / Fs
w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(L) / (L - 1))   # Hamming window
h = h * w
h = h / np.sum(h)
Va_filtered = signal.convolve(Va, h, mode='same')

step = int(np.round(Fs / pmu_rate))
Va_pmu = Va_filtered[::step]
t_pmu = t[::step]
Va_pmu_interp = np.interp(t, t_pmu, Va_pmu)

# ==========================
# 4. COMPRESSED SENSING RECONSTRUCTION
# ==========================
np.random.seed(42)     # separate, explicit seed so CS sampling is
                        # reproducible regardless of what happens earlier
M = int(np.round(0.1 * N))
idx = np.random.choice(N, M, replace=False)
y = Va[idx]

Phi = np.zeros((M, N))
for i in range(M):
    Phi[i, idx[i]] = 1

# FIX: the frequency dictionary must be coherent with the true tones, or
# OMP is forced to spread each tone's energy across neighboring off-grid
# atoms (basis mismatch), which was the single largest source of error.
# The record length is T = 2 s, so frequency resolution 1/T = 0.5 Hz.
# All true frequencies (1, 12, 18, 20, 30 Hz) are exact multiples of 0.5 Hz,
# so building the dictionary on that grid puts every true tone exactly on
# an atom.
freqs = np.arange(0, 60 + 0.5, 0.5)
Psi = np.zeros((N, len(freqs)))
for k in range(len(freqs)):
    Psi[:, k] = np.cos(2 * np.pi * freqs[k] * t)
A_cs = Phi @ Psi

# FIX: K should reflect the actual (or estimated) sparsity of the signal.
# There are 5 true tones, so K = 5 rather than an arbitrary 8; extra
# iterations beyond the true sparsity level just fit noise.
K = 5
theta = omp(A_cs, y, K)
Va_cs = Psi @ theta

# ==========================
# 5. "AI" RECONSTRUCTION (fixed: no data leakage)
# ==========================
# The original version did:
#   p = np.polyfit(Va_pmu_interp, Va, 3); Va_ai = np.polyval(p, Va_pmu_interp)
# That fits directly against Va, the full-resolution ground truth you are
# trying to reconstruct -- info you would never have in a real deployment.
# The fix below trains only on the actual PMU measurements (t_pmu, Va_pmu),
# i.e. only information a real system would have, and predicts at full
# time resolution. It is deliberately a *fair* baseline, not necessarily a
# winning one: since PMU sampling is uniform at 30 Hz, no estimator (linear,
# spline, or neural network) trained on those samples alone can recover
# information above the 15 Hz Nyquist limit -- that loss is fundamental to
# uniform sampling, not a modeling shortcoming. This is precisely why
# compressed sensing (irregular, sub-Nyquist sampling of a sparse signal)
# can outperform it.
t_mean, t_std = t_pmu.mean(), t_pmu.std()
t_pmu_scaled = (t_pmu - t_mean) / t_std
t_scaled = (t - t_mean) / t_std

mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='tanh',
                    max_iter=20000, random_state=0)
mlp.fit(t_pmu_scaled.reshape(-1, 1), Va_pmu)
Va_ai = mlp.predict(t_scaled.reshape(-1, 1))

# ==========================
# VISUALIZATION 1: INDIVIDUAL SIGNALS DASHBOARD
# ==========================
fig1, axs1 = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
fig1.suptitle('Signal Reconstruction Methods Breakdown', fontsize=16, fontweight='bold', color='#2c3e50')

axs1[0, 0].plot(t, Va, color='#2c3e50', linewidth=1.2)
axs1[0, 0].set_title('Original Signal (with Noise)', fontweight='bold')
axs1[0, 0].set_ylabel('Amplitude')

axs1[0, 1].plot(t, Va_pmu_interp, color='#27ae60', linewidth=2)
axs1[0, 1].scatter(t_pmu, Va_pmu, color='#1e8449', s=15, zorder=3, label='PMU Samples')
axs1[0, 1].set_title('PMU Signal (Filtered + Sampled)', fontweight='bold')
axs1[0, 1].legend(loc='upper right')

axs1[1, 0].plot(t, Va_cs, color='#e74c3c', linewidth=2)
axs1[1, 0].scatter(t[idx], y, color='#c0392b', alpha=0.4, s=10, zorder=2, label='Random CS Samples')
axs1[1, 0].set_title('Compressed Sensing Reconstruction', fontweight='bold')
axs1[1, 0].set_xlabel('Time (s)')
axs1[1, 0].set_ylabel('Amplitude')
axs1[1, 0].legend(loc='upper right')

axs1[1, 1].plot(t, Va_ai, color='#3498db', linewidth=2)
axs1[1, 1].set_title('AI Reconstructed Signal (MLP on PMU samples)', fontweight='bold')
axs1[1, 1].set_xlabel('Time (s)')

plt.tight_layout(rect=[0, 0, 1, 0.95])

# ==========================
# VISUALIZATION 2: THE MASTER COMPARISON
# ==========================
plt.figure(figsize=(12, 6))
plt.plot(t, Va, color='#bdc3c7', alpha=0.7, linewidth=1.5, label='Original (Noisy)')
plt.plot(t, Va_cs, color='#e74c3c', linewidth=2, linestyle='--', label='Compressed Sensing (CS)')
plt.plot(t, Va_ai, color='#3498db', linewidth=2, linestyle='-.', label='AI Model')
plt.plot(t, Va_pmu_interp, color='#27ae60', linewidth=2, linestyle=':', label='PMU Interp')

plt.title('All Signals Overlay Comparison', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=11, loc='upper right')
plt.xlim([0, T])

# ==========================
# VISUALIZATION 3: SPECTRUM ANALYSIS
# ==========================
f = np.arange(N) * (Fs / N)
X_orig = np.abs(np.fft.fft(Va))
X_cs = np.abs(np.fft.fft(Va_cs))
X_ai = np.abs(np.fft.fft(Va_ai))

N_pmu = len(Va_pmu)
f_pmu = np.arange(N_pmu) * (pmu_rate / N_pmu)
X_pmu = np.abs(np.fft.fft(Va_pmu))

fig3, axs3 = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
fig3.suptitle('Frequency Spectrum Comparison (0-60 Hz)', fontsize=15, fontweight='bold')

spectrums = [X_orig, X_cs, X_ai]
colors = ['#2c3e50', '#e74c3c', '#3498db']
titles = ['Original Spectrum', 'CS Spectrum', 'AI Spectrum']

for i in range(3):
    axs3[i].fill_between(f, spectrums[i], color=colors[i], alpha=0.3)
    axs3[i].plot(f, spectrums[i], color=colors[i], linewidth=1.5)
    axs3[i].set_xlim([0, 60])
    axs3[i].set_ylabel('Mag')
    axs3[i].set_title(titles[i], fontsize=11, fontweight='bold', loc='left', pad=4)

axs3[3].fill_between(f_pmu, X_pmu, color='#27ae60', alpha=0.3)
axs3[3].plot(f_pmu, X_pmu, color='#27ae60', linewidth=1.5)
axs3[3].set_xlim([0, 60])
axs3[3].set_ylabel('Mag')
axs3[3].set_title('PMU Spectrum (Nyquist-limited to 15 Hz)', fontsize=11, fontweight='bold', loc='left', pad=4)
axs3[3].set_xlabel('Frequency (Hz)', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# ==========================
# 6. ERROR METRICS
# ==========================
rmse_cs = np.sqrt(np.mean((Va - Va_cs) ** 2))
rmse_ai = np.sqrt(np.mean((Va - Va_ai) ** 2))
rmse_pmu = np.sqrt(np.mean((Va - Va_pmu_interp) ** 2))

print("\n" + "=" * 30)
print(f"  RMSE PMU: {rmse_pmu:.4f}")
print(f"  RMSE CS : {rmse_cs:.4f}")
print(f"  RMSE AI : {rmse_ai:.4f}")
print("=" * 30 + "\n")

plt.show()