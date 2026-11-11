import os 
import json 
import glob 
import matplotlib.pyplot as plt 
import numpy as np 

sacred_dir = os.path.expanduser("~/marl_experiments/pymarl/results/sacred")

def get_latest_run_info(config_keyword): 
    """Finds the latest Sacred run directory matching the specified config keyword.""" 
    run_dirs = sorted( [d for d in glob.glob(os.path.join(sacred_dir, "*")) if os.path.basename(d).isdigit()], key=lambda x: int(os.path.basename(x)) ) 
    for r_dir in reversed(run_dirs): 
        cfg_path = os.path.join(r_dir, "config.json") 
        info_path = os.path.join(r_dir, "info.json") 
        
        if os.path.exists(cfg_path) and os.path.exists(info_path): 
            with open(cfg_path, "r") as f: 
                cfg = json.load(f) 
            
            # Check if keyword exists anywhere in configuration 
            cfg_str = json.dumps(cfg).lower() 
            if config_keyword.lower() in cfg_str:
                with open(info_path, "r") as f: 
                    return json.load(f) 
    return None
qmix_info = get_latest_run_info("qmix") 
vdn_info = get_latest_run_info("vdn")

plt.figure(figsize=(12, 5), dpi=150)

def load_sacred_run(run_dir): 
    """Loads metrics.json and info.json from a Sacred run directory.""" 
    metrics_path = os.path.join(run_dir, "metrics.json")
    info_path = os.path.join(run_dir, "info.json") 
    data = {}
    
    if os.path.exists(metrics_path): 
        with open(metrics_path, "r") as f: 
            data["metrics"] = json.load(f) 
    if os.path.exists(info_path): 
        with open(info_path, "r") as f: 
            data["info"] = json.load(f) 
    return data 
def parse_latest_experiment(sacred_base_dir, config_name): 
    """Parses the latest Sacred run for a given configuration (qmix or vdn).""" 
    
    run_dirs = sorted(glob.glob(os.path.join(sacred_base_dir, "[6-14]*")), key=lambda x: int(os.path.basename(x))) 
    
    for r_dir in reversed(run_dirs): 
        config_file = os.path.join(r_dir, "config.json") 
        
        if os.path.exists(config_file): 
            with open(config_file, "r") as f: 
                cfg = json.load(f) 
                if cfg.get("name") == config_name or cfg.get("config") == config_name: 
                    return load_sacred_run(r_dir) 
    return None 

sacred_dir = os.path.expanduser("~/marl_experiments/pymarl/results/sacred") 

# Extract QMIX and VDN metrics 
# qmix_data = parse_latest_experiment(sacred_dir, "qmix") 
# vdn_data = parse_latest_experiment(sacred_dir, "vdn") 
plt.figure(figsize=(12, 5), dpi=150) 

# 1. Plot Test Win Rate 
plt.subplot(1, 2, 1) 
plt.axhline(y=82, color="gray", linestyle="--", label="Heuristic Baseline (82%)")

if qmix_info and "test_battle_won_mean" in qmix_info: 
    t_qmix = np.array(qmix_info["test_battle_won_mean_T"]) / 1e6 
    win_qmix = np.array(qmix_info["test_battle_won_mean"]) * 100 
    plt.plot(t_qmix, win_qmix, label="QMIX", color="#1f77b4", linewidth=2)

if vdn_info and "test_battle_won_mean" in vdn_info: 
    t_vdn = np.array(vdn_info["test_battle_won_mean_T"]) / 1e6 
    win_vdn = np.array(vdn_info["test_battle_won_mean"]) * 100 
    plt.plot(t_vdn, win_vdn, label="VDN", color="#ff7f0e", linewidth=2)

plt.title("Test Battle Win Rate on 2s3z", fontsize=12, fontweight="bold") 
plt.xlabel("Environment Timesteps (Millions)", fontsize=10) 
plt.ylabel("Test Win Rate (%)", fontsize=10) 
plt.ylim(-5, 105) 
plt.grid(True, alpha=0.3) 
plt.legend(loc="lower right")


# if qmix_data and "test_battle_won_mean" in qmix_data.get("info", {}): 
#     t_qmix = qmix_data["info"]["test_battle_won_mean_T"] 
#     win_qmix = np.array(qmix_data["info"]["test_battle_won_mean"]) * 100 
#     plt.plot(np.array(t_qmix) / 1e6, win_qmix, label="QMIX (Monotonic)", color="#1f77b4", linewidth=2) 
    
# if vdn_data and "test_battle_won_mean" in vdn_data.get("info", {}): 
#     t_vdn = vdn_data["info"]["test_battle_won_mean_T"] 
#     win_vdn = np.array(vdn_data["info"]["test_battle_won_mean"]) * 100 
#     plt.plot(np.array(t_vdn) / 1e6, win_vdn, label="VDN (Linear)", color="#ff7f0e", linewidth=2) 
    
# plt.axhline(y=82, color="gray", linestyle="--", label="Heuristic Baseline (82%)") 
# plt.title("Test Battle Win Rate on 2s3z", fontsize=12, fontweight="bold") 
# plt.xlabel("Environment Timesteps (Millions)", fontsize=10) 
# plt.ylabel("Test Win Rate (%)", fontsize=10) 
# plt.ylim(-5, 105) 
# plt.grid(True, alpha=0.3) 
# plt.legend(loc="lower right") 

# 2. Plot Test Return Mean 
plt.subplot(1, 2, 2)
plt.axhline(y=20, color="green", linestyle=":", label="Max Reward (20.0)")

if qmix_info and "test_return_mean" in qmix_info: 
    t_qmix = np.array(qmix_info["test_return_mean_T"]) / 1e6 
    ret_qmix = qmix_info["test_return_mean"] 
    plt.plot(t_qmix, ret_qmix, label="QMIX", color="#1f77b4", linewidth=2)

if vdn_info and "test_return_mean" in vdn_info: 
    t_vdn = np.array(vdn_info["test_return_mean_T"]) / 1e6 
    ret_vdn = vdn_info["test_return_mean"] 
    plt.plot(t_vdn, ret_vdn, label="VDN", color="#ff7f0e", linewidth=2)    

plt.title("Test Return Mean on 2s3z", fontsize=12, fontweight="bold") 
plt.xlabel("Environment Timesteps (Millions)", fontsize=10) 
plt.ylabel("Mean Episode Return", fontsize=10) 
plt.grid(True, alpha=0.3) 
plt.legend(loc="lower right") 
plt.tight_layout() 
plt.savefig("qmix_vs_vdn_2s3z_results.png") 
plt.show()
    
# if qmix_data and "test_return_mean" in qmix_data.get("info", {}): 
#     t_qmix = qmix_data["info"]["test_return_mean_T"] 
#     ret_qmix = qmix_data["info"]["test_return_mean"] 
#     plt.plot(np.array(t_qmix) / 1e6, ret_qmix, label="QMIX", color="#1f77b4", linewidth=2) 

# if vdn_data and "test_return_mean" in vdn_data.get("info", {}): 
#     t_vdn = vdn_data["info"]["test_return_mean_T"] 
#     ret_vdn = vdn_data["info"]["test_return_mean"] 
#     plt.plot(np.array(t_vdn) / 1e6, ret_vdn, label="VDN", color="#ff7f0e", linewidth=2) 
# plt.axhline(y=20, color="green", linestyle=":", label="Max Reward (20.0)") 
# plt.title("Test Return Mean on 2s3z", fontsize=12, fontweight="bold") 
# plt.xlabel("Environment Timesteps (Millions)", fontsize=10) 
# plt.ylabel("Mean Episode Return", fontsize=10) 
# plt.grid(True, alpha=0.3) 
# plt.legend(loc="lower right") 
# plt.tight_layout() 
# plt.savefig("qmix_vs_vdn_2s3z_results.png") 
# plt.show() 
# print("Saved comparison plot to qmix_vs_vdn_2s3z_results.png")