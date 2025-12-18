import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools

# --- 1. DIRECTORY SETUP ---
output_dir = "dataset"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def plot_xapp_network(xapp_map, param_kpi_map):
    plt.figure(figsize=(12, 8))
    xapp_list = list(xapp_map.keys())
    param_list = sorted(list(param_kpi_map.keys()), key=lambda x: int(x[1:]))
    kpi_list = sorted([k for k in set(k for kpis in param_kpi_map.values() for k in kpis)])
    pos = {}
    for i, label in enumerate(xapp_list): pos[label] = (0, i / max(1, len(xapp_list) - 1))
    for i, label in enumerate(param_list): pos[label] = (1, i / max(1, len(param_list) - 1))
    for i, label in enumerate(kpi_list): pos[label] = (2, i / max(1, len(kpi_list) - 1))

    for x, params in xapp_map.items():
        for p in params:
            if p in pos:
                plt.plot([pos[x][0], pos[p][0]], [pos[x][1], pos[p][1]], color='blue', alpha=0.3, lw=1)
    for p, kpis in param_kpi_map.items():
        for k in kpis:
            if k in pos:
                plt.plot([pos[p][0], pos[k][0]], [pos[p][1], pos[k][1]], color='green', alpha=0.3, lw=1)

    for label, (x, y) in pos.items():
        color = 'skyblue' if x == 0 else ('lightgreen' if x == 1 else 'salmon')
        plt.scatter(x, y, s=500, color=color, zorder=5, edgecolors='black')
        plt.text(x, y, label, fontsize=10, ha='center', va='center', fontweight='bold')

    plt.title("Three-Layer xApp-Parameter-KPI Control Network")
    plt.xticks([0, 1, 2], ["xApps", "Parameters", "KPIs"])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def generate_param_domains(num_params):
    """Generate random domain ranges for parameters"""
    domains = {}
    for i in range(1, num_params + 1):
        lo = random.uniform(0, 100)
        hi = lo + random.uniform(50, 500)
        domains[f"p{i}"] = (lo, hi)
    return domains

def generate_kpi_domains(num_kpis):
    """Generate random domain ranges for KPIs"""
    domains = {}
    for i in range(1, num_kpis + 1):
        lo = random.uniform(0, 50)
        hi = lo + random.uniform(20, 200)
        domains[f"k{i}"] = (lo, hi)
    return domains

def generate_xapp_param_map(num_xapps, num_params, sparsity=0.5):
    """Randomly generate xApp-Parameter mappings"""
    xapp_map = {}
    for i in range(1, num_xapps + 1):
        # Each xApp controls 30-60% of parameters
        num_controls = max(1, int(num_params * random.uniform(0.3, 0.6)))
        params = random.sample([f"p{j}" for j in range(1, num_params + 1)], num_controls)
        xapp_map[f"x{i}"] = params
    return xapp_map

def generate_param_kpi_map(num_params, num_kpis, sparsity=0.5):
    """Randomly generate Parameter-KPI influence mappings"""
    param_map = {}
    for i in range(1, num_params + 1):
        # Each parameter influences 30-70% of KPIs
        num_influences = max(1, int(num_kpis * random.uniform(0.3, 0.7)))
        kpis = random.sample([f"k{j}" for j in range(1, num_kpis + 1)], num_influences)
        param_map[f"p{i}"] = kpis
    return param_map

def random_range(lo, hi, min_width=0.2):
    midpoint = (lo + hi) / 2
    a = random.uniform(lo, midpoint - (min_width / 2))
    b = random.uniform(midpoint + (min_width / 2), hi)
    return round(a, 2), round(b, 2)

def run_simulation(sim_num, num_xapps, num_params, num_kpis):
    """Run a single simulation and return results"""
    print(f"\n{'='*80}")
    print(f"SIMULATION {sim_num}: {num_xapps} xApps, {num_params} Parameters, {num_kpis} KPIs")
    print(f"{'='*80}")
    
    # Generate dynamic configuration
    xapp_labels = [f"x{i}" for i in range(1, num_xapps + 1)]
    param_labels = [f"p{i}" for i in range(1, num_params + 1)]
    kpi_labels = [f"k{i}" for i in range(1, num_kpis + 1)]
    
    param_domains = generate_param_domains(num_params)
    kpi_domains = generate_kpi_domains(num_kpis)
    xapp_param_map = generate_xapp_param_map(num_xapps, num_params)
    parameter_kpi_map = generate_param_kpi_map(num_params, num_kpis)
    
    print(f"Generated config: {num_xapps} xApps, {num_params} params, {num_kpis} KPIs")
    
    # Generate xApp Proposals
    xapp_param_data = []
    for x in xapp_labels:
        for p in xapp_param_map.get(x, []):
            lo, hi = param_domains[p]
            rmin, rmax = random_range(lo, hi)
            xapp_param_data.append([x, p, rmin, rmax])
    xapp_param_df = pd.DataFrame(xapp_param_data, columns=["xapp", "parameter", "range_min", "range_max"])
    
    # Generate KPI Influence Ranges
    param_kpi_data = []
    for p in param_labels:
        for k in parameter_kpi_map.get(p, []):
            plo, phi = param_domains[p]
            klo, khi = kpi_domains[k]
            pmin, pmax = random_range(plo, phi)
            param_kpi_data.append([p, pmin, pmax, k, klo, khi])
    param_kpi_df = pd.DataFrame(param_kpi_data, columns=["parameter", "p_range_min", "p_range_max", "kpi", "k_min", "k_max"])
    
    # Structural Incidence Matrices
    I_xp = pd.DataFrame(0, index=xapp_labels, columns=param_labels)
    for x, params in xapp_param_map.items():
        for p in params: 
            I_xp.loc[x, p] = 1
    
    I_pk = pd.DataFrame(0, index=param_labels, columns=kpi_labels)
    for p, k_list in parameter_kpi_map.items():
        for k in k_list: 
            I_pk.loc[p, k] = 1
    
    # Convert to numpy arrays for matrix operations
    I_xp_np = I_xp.values.astype(int)
    I_pk_np = I_pk.values.astype(int)
    
    Nx = len(xapp_labels)
    Nk = len(kpi_labels)
    
    # Calculate C_ik (xApp × KPI interaction matrix)
    print("\nCalculating dot product of xApp and KPI")
    C_ik = I_xp_np @ I_pk_np  # Nx × Nk matrix
    
    Cik_details = []
    for i in range(Nx):
        row_i_of_Ixp = I_xp_np[i, :]
        for k in range(Nk):
            column_k_of_Ipk = I_pk_np[:, k]
            
            C_ik_value = np.dot(row_i_of_Ixp, column_k_of_Ipk)
            vector = row_i_of_Ixp * column_k_of_Ipk
            
            if C_ik_value > 0:
                Cik_details.append({
                    'xApp_i': i,
                    'xApp_label': xapp_labels[i],
                    'KPI_k': k,
                    'KPI_label': kpi_labels[k],
                    'Cik_sum': C_ik_value,
                    'vector': vector
                })
    
    print(f"Found {len(Cik_details)} xApp-KPI interactions")
    
    print("\n" + "-"*80)
    print("Generating Conflict Matrices")
    
    # Conflict Matrix Generation for each KPI
    kpi_conflict_matrices = []
    
    for k_index in range(Nk):
        kpi_id = kpi_labels[k_index]
        
        # Create dictionary to store xapp:vector
        kpi_vectors = {
            detail['xApp_label']: detail['vector']
            for detail in Cik_details if detail['KPI_k'] == k_index
        }
        
        # List of xApp labels that are relevant for this KPI
        relevant_xapps = sorted(kpi_vectors.keys())
        
        N_prime = len(relevant_xapps)
        
        if N_prime < 2:
            print(f"KPI {kpi_id} skipped: Not enough xApps (N'={N_prime}) for conflict analysis.")
            continue
        
        # Initialize conflict matrix
        Ik_matrix = np.zeros((N_prime, N_prime), dtype=int)
        
        print(f"\nCalculating Conflict Matrix for KPI {kpi_id}")
        print(f"Includes xApps: {relevant_xapps}")
        
        # Fill diagonal with -1 (no conflict with itself)
        np.fill_diagonal(Ik_matrix, -1)
        
        # All unique xApp pairs
        index_pairs = list(itertools.combinations(range(N_prime), 2))
        
        conflict_count = 0
        for i_prime, j_prime in index_pairs:
            xapp_i_label = relevant_xapps[i_prime]
            xapp_j_label = relevant_xapps[j_prime]
            
            V_ik = kpi_vectors[xapp_i_label]
            V_jk = kpi_vectors[xapp_j_label]
            
            conflict_strength = int(V_ik @ V_jk)
            
            Ik_matrix[i_prime, j_prime] = conflict_strength
            Ik_matrix[j_prime, i_prime] = conflict_strength
            
            if conflict_strength > 0:
                conflict_count += 1
        
        print(f"Found {conflict_count} direct conflicts in KPI {kpi_id}")
        
        kpi_conflict_matrices.append({
            'KPI_ID': kpi_id,
            'XApp_IDs': relevant_xapps,
            'Matrix': Ik_matrix
        })
    
    # Return results
    return {
        'sim_num': sim_num,
        'num_xapps': num_xapps,
        'num_params': num_params,
        'num_kpis': num_kpis,
        'xapp_labels': xapp_labels,
        'param_labels': param_labels,
        'kpi_labels': kpi_labels,
        'I_xp': I_xp,
        'I_pk': I_pk,
        'C_ik': C_ik,
        'xapp_param_df': xapp_param_df,
        'param_kpi_df': param_kpi_df,
        'kpi_conflict_matrices': kpi_conflict_matrices
    }

# --- 2. SIMULATION CONFIGURATION ---
# Define simulation scenarios: (num_xapps, num_params, num_kpis)
simulations = [
    (6, 10, 4),      # Baseline
    (10, 15, 6),     # Medium scale
    (15, 20, 8),     # Large scale
    (20, 25, 10),    # Extra large scale
    (30, 40, 15),
    (50,75,20),
    (100,150,25),    # Very large scale
]

all_results = []

# --- 3. RUN ALL SIMULATIONS ---
for idx, (num_x, num_p, num_k) in enumerate(simulations, 1):
    result = run_simulation(idx, num_x, num_p, num_k)
    all_results.append(result)
    
    # Save each simulation to a single file: Sim1.csv, Sim2.csv, etc.
    sim_file_path = os.path.join(output_dir, f"Sim{idx}.csv")
    with open(sim_file_path, 'w') as f:
        f.write(f"SIMULATION {idx}: {num_x} xApps, {num_p} Parameters, {num_k} KPIs\n\n")
        f.write("XAPP-PARAMETER CONTROL RANGES\n")
        result['xapp_param_df'].to_csv(f, index=False)
        f.write("\nPARAMETER-KPI INFLUENCE RANGES\n")
        result['param_kpi_df'].to_csv(f, index=False)
        f.write("\nI_XP INCIDENCE MATRIX\n")
        result['I_xp'].to_csv(f)
        f.write("\nI_PK INCIDENCE MATRIX\n")
        result['I_pk'].to_csv(f)
        f.write("\nC_IK INTERACTION MATRIX (xApp-KPI)\n")
        df_C_ik = pd.DataFrame(result['C_ik'], index=result['xapp_labels'], columns=result['kpi_labels'])
        df_C_ik.to_csv(f)
        
        for conflict_data in result['kpi_conflict_matrices']:
            f.write(f"\nCONFLICT MATRIX FOR KPI {conflict_data['KPI_ID']}\n")
            df_conflict = pd.DataFrame(
                conflict_data['Matrix'],
                index=conflict_data['XApp_IDs'],
                columns=conflict_data['XApp_IDs']
            )
            df_conflict.to_csv(f)
    
    print(f"\nSimulation {idx} saved to {sim_file_path}")

# --- 4. CREATE FINALSIM.CSV WITH ALL RESULTS APPENDED ---
print(f"\n{'='*80}")
print("Creating FinalSim.csv with all simulations")
print(f"{'='*80}\n")

final_file_path = os.path.join(output_dir, "FinalSim.csv")

with open(final_file_path, 'w') as f:
    for result in all_results:
        sim_num = result['sim_num']
        num_x = result['num_xapps']
        num_p = result['num_params']
        num_k = result['num_kpis']
        
        f.write(f"\n{'='*80}\n")
        f.write(f"SIMULATION {sim_num}: {num_x} xApps, {num_p} Parameters, {num_k} KPIs\n")
        f.write(f"{'='*80}\n\n")
        f.write("XAPP-PARAMETER CONTROL RANGES\n")
        result['xapp_param_df'].to_csv(f, index=False)
        f.write("\nPARAMETER-KPI INFLUENCE RANGES\n")
        result['param_kpi_df'].to_csv(f, index=False)
        f.write("\nI_XP INCIDENCE MATRIX\n")
        result['I_xp'].to_csv(f)
        f.write("\nI_PK INCIDENCE MATRIX\n")
        result['I_pk'].to_csv(f)
        f.write("\nC_IK INTERACTION MATRIX (xApp-KPI)\n")
        df_C_ik = pd.DataFrame(result['C_ik'], index=result['xapp_labels'], columns=result['kpi_labels'])
        df_C_ik.to_csv(f)
        
        for conflict_data in result['kpi_conflict_matrices']:
            f.write(f"\nCONFLICT MATRIX FOR KPI {conflict_data['KPI_ID']}\n")
            df_conflict = pd.DataFrame(
                conflict_data['Matrix'],
                index=conflict_data['XApp_IDs'],
                columns=conflict_data['XApp_IDs']
            )
            df_conflict.to_csv(f)

print(f"All simulations saved to {final_file_path}")
print("\nGenerated files:")
for i in range(1, len(simulations) + 1):
    print(f"  - Sim{i}.csv")
print(f"  - FinalSim.csv (comprehensive)")

print(f"\n{'='*80}")
print("ALL SIMULATIONS COMPLETE")
print(f"{'='*80}\n")