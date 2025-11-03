#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 13:42:50 2025

@author: avnishsingh
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import product, permutations
from scipy.spatial import Voronoi

# Parameters for Silicon
a = 5.431  # Lattice constant in Angstroms
hbar2_2m = 3.81  # hbar^2 / 2m in eV Angstrom^2
ry_to_ev = 13.6057  # Conversion from Ry to eV

# Pseudopotential form factors for Si in Ry (from Cohen-Bergstresser)
V3_ry = -0.21
V8_ry = 0.04
V11_ry = 0.08

# Convert to eV
V3 = V3_ry * ry_to_ev
V8 = V8_ry * ry_to_ev
V11 = V11_ry * ry_to_ev

# Unit for reciprocal space
unit = 2 * np.pi / a

# Generate reciprocal lattice vectors G = unit * (h, k, l) where h,k,l all even or all odd, and |G_red|^2 <= 15
G_list = []
max_index = 6
cutoff_sq = 15
for h, k, l in product(range(-max_index, max_index + 1), repeat=3):
    par_h = abs(h) % 2
    par_k = abs(k) % 2
    par_l = abs(l) % 2
    if par_h == par_k == par_l:
        sq = h**2 + k**2 + l**2
        if sq <= cutoff_sq:
            G_list.append(unit * np.array([h, k, l]))

G_list = np.array(G_list)
num_G = len(G_list)
print(f"Number of plane waves in basis: {num_G}")

# High symmetry points in reduced coordinates (multiplied by 2pi/a later)
sym_points = {
    'Gamma': np.array([0.0, 0.0, 0.0]),
    'L': np.array([0.5, 0.5, 0.5]),
    'X': np.array([0.0, 1.0, 0.0]),
    'W': np.array([1.0, 0.5, 0.0]),
    'K': np.array([0.75, 0.75, 0.0]),
}

# Path in k-space
path_labels = ['L', 'Gamma', 'X', 'W', 'K', 'Gamma']
path = [sym_points[label] * unit for label in path_labels]

# Generate k-points along the path
num_segments = len(path) - 1
num_points_per_segment = 50  # Adjust for smoother plot (higher = smoother but slower)
k_list = []
x_vals = []
tick_pos = [0.0]
tick_labels = [path_labels[0]]
current_x = 0.0

for i in range(num_segments):
    k_start = path[i]
    k_end = path[i + 1]
    segment_length = np.linalg.norm(k_end - k_start)
    for j in range(1, num_points_per_segment + 1):
        frac = j / num_points_per_segment
        k = (1 - frac) * k_start + frac * k_end
        k_list.append(k)
        current_x += segment_length / num_points_per_segment
        x_vals.append(current_x)
    tick_pos.append(current_x)
    tick_labels.append(path_labels[i + 1])

# Add the starting point
k_list.insert(0, path[0])
x_vals.insert(0, 0.0)

k_list = np.array(k_list)
x_vals = np.array(x_vals)
num_k = len(k_list)

# Function to get pseudopotential V for a given |G|^2 in reduced units
def get_V(sq):
    if abs(sq - 3) < 1e-6:
        return V3
    elif abs(sq - 8) < 1e-6:
        return V8
    elif abs(sq - 11) < 1e-6:
        return V11
    else:
        return 0.0

# Compute bands
num_bands = 8  # Number of bands to plot
energies = np.zeros((num_k, num_bands))

for ik, k in enumerate(k_list):
    H = np.zeros((num_G, num_G))
    for i in range(num_G):
        G_i = G_list[i]
        kin_i = hbar2_2m * np.sum((k + G_i)**2)
        H[i, i] = kin_i
        for j in range(num_G):
            if i == j:
                continue
            dq = G_list[i] - G_list[j]
            dq_red = dq / unit
            sq_dq = np.dot(dq_red, dq_red)
            v = get_V(sq_dq)
            H[i, j] = v  # Since V is real and symmetric
    eigs = np.linalg.eigvalsh(H)
    energies[ik, :] = np.sort(eigs)[:num_bands]

# Plot the band structure
fig_band, ax_band = plt.subplots(figsize=(8, 6))
for band in range(num_bands):
    ax_band.plot(x_vals, energies[:, band], color='blue')

ax_band.set_xticks(tick_pos)
ax_band.set_xticklabels(tick_labels)
ax_band.set_xlabel('k-path')
ax_band.set_ylabel('Energy (eV)')
ax_band.set_title('Nearly Free Electron Band Structure for Si (1st Brillouin Zone)')
ax_band.grid(True)

# Function to get Brillouin zone
def get_brillouin_zone_3d(reciprocal_cell):
    reciprocal_cell = np.asarray(reciprocal_cell, dtype=float)
    assert reciprocal_cell.shape == (3, 3)

    px, py, pz = np.tensordot(reciprocal_cell, np.mgrid[-1:2, -1:2, -1:2], axes=[0, 0])
    points = np.c_[px.ravel(), py.ravel(), pz.ravel()]

    vor = Voronoi(points)

    bz_facets = []
    bz_ridges = []
    bz_vertices = []

    for pid, rid in zip(vor.ridge_points, vor.ridge_vertices):
        if (pid[0] == 13 or pid[1] == 13):
            bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
            bz_facets.append(vor.vertices[rid])
            bz_vertices += rid

    bz_vertices = list(set(bz_vertices))

    return vor.vertices[bz_vertices], bz_ridges, bz_facets

# Reciprocal primitive vectors for Si (diamond/FCC Bravais)
rec_cell = unit * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])

# Get BZ
bz_vertices, bz_ridges, bz_facets = get_brillouin_zone_3d(rec_cell)

# To show the irreducible zone, plot the high symmetry path in 3D k-space
fig_3d = plt.figure(figsize=(8, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# Plot the whole BZ faces
ax_3d.add_collection3d(Poly3DCollection(bz_facets, facecolors='gray', linewidths=1, edgecolors='k', alpha=0.1))

# Plot the path
path_k = np.array([sym_points[label] * unit for label in path_labels])
ax_3d.plot(path_k[:, 0], path_k[:, 1], path_k[:, 2], color='blue', linewidth=3, label='High symmetry path')

# Plot symmetry points
for label, pos in sym_points.items():
    p = pos * unit
    ax_3d.scatter(p[0], p[1], p[2], color='red', s=50)
    ax_3d.text(p[0], p[1], p[2] + 0.05 * unit, label, fontsize=12)

ax_3d.set_xlabel('kx (1/Å)')
ax_3d.set_ylabel('ky (1/Å)')
ax_3d.set_zlabel('kz (1/Å)')
ax_3d.set_title('1st Brillouin Zone and Irreducible Path for Si')
ax_3d.legend()

plt.show()