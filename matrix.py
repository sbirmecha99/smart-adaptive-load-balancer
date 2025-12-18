import numpy as np

param_labels = ["p1", "p2", "p3", "p4", "p5"]
xapp_labels  = ["x1", "x2", "x3", "x4"]

xapp_param_map = {
    "x1": ["p1", "p2", "p3"],
    "x2": ["p1", "p4"],
    "x3": ["p5"],
    "x4": ["p2", "p5"]
}

parameter_kpi_map = {
    "p1": ["k1", "k2"],
    "p2": ["k1", "k3"],
    "p3": ["k3"],
    "p4": ["k1", "k4"],
    "p5": ["k2", "k3"]
}

# 1) GLOBAL PARAMETER RANGES (toy example)
param_global_ranges = {
    "p1": (-6.0, 6.0),
    "p2": (0.0, 10.0),
    "p3": (0.0, 320.0),
    "p4": (-10.0, 10.0),
    "p5": (0.0, 40.0),
}

# 2) xAPP-SPECIFIC PARAMETER RANGES
xapp_param_ranges = {
    "x1": {
        "p1": (-3.0, 3.0),
        "p2": (2.0, 6.0),
        "p3": (40.0, 80.0)
    },
    "x2": {
        "p1": (1.0, 5.0),
        "p4": (-5.0, 5.0)
    },
    "x3": {
        "p5": (10.0, 20.0)
    },
    "x4": {
        "p2": (4.0, 9.0),
        "p5": (15.0, 30.0)
    },
}


num_xapps  = len(xapp_labels)
num_params = len(param_labels)

# BUILD I_xp (xApp vs Parameter)
I_xp = np.zeros((num_xapps, num_params), dtype=int)

for i, x in enumerate(xapp_labels):
    for j, p in enumerate(param_labels):
        if p in xapp_param_map.get(x, []):
            I_xp[i, j] = 1

# BUILD I_pk (Parameter vs KPI)
kpi_labels = []
for p in param_labels:
    for k in parameter_kpi_map[p]:
        if k not in kpi_labels:
            kpi_labels.append(k)

num_kpis = len(kpi_labels)

I_pk = np.zeros((num_params, num_kpis), dtype=int)
for i, p in enumerate(param_labels):
    for k in parameter_kpi_map[p]:
        I_pk[i, kpi_labels.index(k)] = 1

# PRINT MATRICES
print("\n=== I_xp (xApp vs Parameter) ===")
print("      " + "  ".join(param_labels))
for i, x in enumerate(xapp_labels):
    print(f"{x}:  " + "  ".join(map(str, I_xp[i])))

print("\n=== I_pk (Parameter vs KPI) ===")
print("KPI:", list(kpi_labels))
print("      " + "  ".join(kpi_labels))
for i, p in enumerate(param_labels):
    print(f"{p}:  " + "  ".join(map(str, I_pk[i])))

# SAVE MATRICES
np.save("I_xp.npy", I_xp)
np.save("I_pk.npy", I_pk)
np.save("param_labels.npy", param_labels)
np.save("xapp_labels.npy", xapp_labels)
np.save("kpi_labels.npy", kpi_labels)

print("\nMatrices saved successfully.")

I_xp = np.load("I_xp.npy")
I_pk = np.load("I_pk.npy")
param_labels = np.load("param_labels.npy", allow_pickle=True)
xapp_labels  = np.load("xapp_labels.npy", allow_pickle=True)
kpi_labels   = np.load("kpi_labels.npy", allow_pickle=True)

num_xapps  = len(xapp_labels)
num_params = len(param_labels)

print("\n=== Loaded I_xp ===")
print("      " + "  ".join(param_labels))
for i, x in enumerate(xapp_labels):
    print(f"{x}:  " + "  ".join(map(str, I_xp[i])))

print("\n=== Loaded I_pk ===")
print("KPI:", list(kpi_labels))
print("      " + "  ".join(kpi_labels))
for i, p in enumerate(param_labels):
    print(f"{p}:  " + "  ".join(map(str, I_pk[i])))

# AND operation + L1 norm
for k_idx, k in enumerate(kpi_labels):
    print(f"\n========== KPI = {k} ==========")

    v_k = I_pk[:, k_idx].astype(bool)

    for i, x in enumerate(xapp_labels):
        and_vec = np.logical_and(I_xp[i].astype(bool), v_k).astype(int)
        L1 = and_vec.sum()
        print(f"{x}: AND = {and_vec}, L1 = {L1}")

from itertools import combinations

I_xp = np.load("I_xp.npy")
I_pk = np.load("I_pk.npy")
param_labels = np.load("param_labels.npy", allow_pickle=True)
xapp_labels  = np.load("xapp_labels.npy", allow_pickle=True)
kpi_labels   = np.load("kpi_labels.npy", allow_pickle=True)

print("\n=== Loaded I_xp ===")
print("      " + "  ".join(param_labels))
for i, x in enumerate(xapp_labels):
    print(f"{x}:  " + "  ".join(map(str, I_xp[i])))

print("\n=== Loaded I_pk ===")
print("KPI:", list(kpi_labels))
print("      " + "  ".join(kpi_labels))
for i, p in enumerate(param_labels):
    print(f"{p}:  " + "  ".join(map(str, I_pk[i])))

# TRIANGULAR CONFLICT MATRIX PER KPI
for k_idx, k in enumerate(kpi_labels):
    print(f"\n===== Conflict Matrix for KPI {k} =====")

    v_k = I_pk[:, k_idx].astype(bool)

    active = {}
    for i, x in enumerate(xapp_labels):
        and_vec = np.logical_and(I_xp[i].astype(bool), v_k).astype(int)
        if and_vec.sum() > 0:
            active[x] = and_vec

    if len(active) == 0:
        print("No active xApps (L1=0).")
        continue

    xapps = list(active.keys())
    n = len(xapps)
    C = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            dot = active[xapps[i]].dot(active[xapps[j]])
            C[i, j] = C[j, i] = 1 if dot >= 1 else 0

    print("      " + "  ".join(xapps))
    for i, xi in enumerate(xapps):
        row = []
        for j in range(n):
            row.append("-" if i == j else str(C[i, j]))
        print(f"{xi}:  " + "  ".join(row))