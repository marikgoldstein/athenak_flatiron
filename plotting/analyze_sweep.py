#!/usr/bin/env python3
"""Analyze turbulence sustainability and domain gap for nu sweep."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cases = {
    '128 nu=3e-5': '/mnt/home/mgoldstein/ceph/athenak/mri128_nu3e-5/seed_0000/HB3.mhd.hst',
    '256 nu=3e-5': '/mnt/home/mgoldstein/ceph/athenak/mri256_nu3e-5/seed_0000/HB3.mhd.hst',
    '128 nu=5e-5': '/mnt/home/mgoldstein/ceph/athenak/mri128_nu5e-5/seed_0000/HB3.mhd.hst',
    '256 nu=5e-5': '/mnt/home/mgoldstein/ceph/athenak/mri256_nu5e-5/seed_0000/HB3.mhd.hst',
}

# Also load the 1e-4 check runs for comparison
cases_ref = {
    '128 nu=1e-4': '/mnt/home/mgoldstein/ceph/athenak/mri128_check/seed_0000/HB3.mhd.hst',
    '256 nu=1e-4': '/mnt/home/mgoldstein/ceph/athenak/mri256_check/seed_0000/HB3.mhd.hst',
}

T_orb = 2 * np.pi  # orbital period

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for label, path in {**cases, **cases_ref}.items():
    try:
        data = np.loadtxt(path)
    except Exception as e:
        print(f"SKIP {label}: {e}")
        continue

    time = data[:, 0]
    orbits = time / T_orb
    # Columns: [1]=time [2]=dt [3]=mass [4]=1-mom [5]=2-mom [6]=3-mom
    #          [7]=1-KE [8]=2-KE [9]=3-KE [10]=1-ME [11]=2-ME [12]=3-ME
    KE_x = data[:, 6]   # radial kinetic energy
    KE_y = data[:, 7]   # azimuthal kinetic energy
    KE_total = KE_x + KE_y + data[:, 8]
    ME_x = data[:, 9]   # radial magnetic energy
    ME_y = data[:, 10]  # azimuthal magnetic energy
    ME_total = ME_x + ME_y + data[:, 11]

    # Determine linestyle and color
    ls = '-' if '128' in label else '--'
    color_map = {'3e-5': 'C0', '5e-5': 'C1', '1e-4': 'C2'}
    color = 'C3'
    for k, v in color_map.items():
        if k in label:
            color = v
            break

    axes[0, 0].semilogy(orbits, KE_total, ls=ls, color=color, label=label, alpha=0.8)
    axes[0, 1].semilogy(orbits, ME_total, ls=ls, color=color, label=label, alpha=0.8)
    axes[1, 0].semilogy(orbits, KE_x, ls=ls, color=color, label=label, alpha=0.8)
    axes[1, 1].semilogy(orbits, ME_x, ls=ls, color=color, label=label, alpha=0.8)

    # Print summary stats
    early = (orbits > 5) & (orbits < 20)
    mid = (orbits > 30) & (orbits < 50)
    late = (orbits > 70) & (orbits < 100)

    print(f"\n=== {label} ===")
    print(f"  KE_total: early(5-20 orb)={KE_total[early].mean():.2e}, mid(30-50)={KE_total[mid].mean():.2e}, late(70-100)={KE_total[late].mean():.2e}")
    print(f"  ME_total: early(5-20 orb)={ME_total[early].mean():.2e}, mid(30-50)={ME_total[mid].mean():.2e}, late(70-100)={ME_total[late].mean():.2e}")
    print(f"  KE_x:     early(5-20 orb)={KE_x[early].mean():.2e}, mid(30-50)={KE_x[mid].mean():.2e}, late(70-100)={KE_x[late].mean():.2e}")

    # Check if turbulence is sustained: ratio of late to mid
    if KE_total[mid].mean() > 0:
        ratio = KE_total[late].mean() / KE_total[mid].mean()
        status = "SUSTAINED" if ratio > 0.3 else "DECAYING" if ratio > 0.01 else "DEAD"
        print(f"  Late/Mid KE ratio: {ratio:.3f} -> {status}")

axes[0, 0].set_title('Total Kinetic Energy')
axes[0, 1].set_title('Total Magnetic Energy')
axes[1, 0].set_title('Radial KE (MRI indicator)')
axes[1, 1].set_title('Radial ME (MRI indicator)')

for ax in axes.flat:
    ax.set_xlabel('Orbits')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('MRI Turbulence Sustainability: nu sweep comparison', fontsize=14)
plt.tight_layout()
outpath = '/mnt/home/mgoldstein/ceph/athenak/turbulence_sustainability.png'
plt.savefig(outpath, dpi=150)
print(f"\nSaved: {outpath}")
