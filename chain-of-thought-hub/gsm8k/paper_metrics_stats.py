import orjson
import numpy as np
import matplotlib.pyplot as plt
import mmap
import time

# Script to calculate paper metrics: block efficiency (token/step) and decoding speed (token/second)
# Based on compute_speculative_stats.py paradigm

models = "Qwen_72B_0.5B_"
gamma = 10

start = time.time()

# Load all JSON files using memory mapping for efficiency
with open(f"/u/qwu4/ssd/code/Fast-HSD/chain-of-thought-hub/gsm8k/exp/72b/Qwen_72B_0.5B_naive_gamma_10_topp_1.0_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_naive = orjson.loads(mm[:])
    mm.close()

with open(f"/u/qwu4/ssd/code/Fast-HSD/chain-of-thought-hub/gsm8k/exp/72b/Qwen_72B_0.5B_blockwise_gamma_10_topp_1.0_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_blockwise = orjson.loads(mm[:])
    mm.close()

with open(f"/u/qwu4/ssd/code/Fast-HSD/chain-of-thought-hub/gsm8k/exp/72b/Qwen_72B_0.5B_backward_gamma_10_topp_1.0_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_backward = orjson.loads(mm[:])
    mm.close()

with open(f"/u/qwu4/ssd/code/Fast-HSD/chain-of-thought-hub/gsm8k/exp/72b/Qwen_72B_0.5B_backward_clever_approxi_gamma_10_topp_1.0_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_backward_clever_approxi = orjson.loads(mm[:])
    mm.close()

end = time.time()
print(f"Data loading time: {end-start:.2f} seconds")

# Method configurations
counts = [
    counts_naive,
    counts_blockwise,
    counts_backward,
    counts_backward_clever_approxi,
]

method_labels = [
    'Naive',
    'Blockwise', 
    'NaiveHSD',
    'FastHSD'
]

# Initialize metric storage
block_efficiency = []  # token/step
decoding_speed = []    # token/second
sample_counts = []     # number of valid samples for each method

print("Calculating metrics for each method...")

for i, count in enumerate(counts):
    total_tokens = 0    # Total tokens for samples where draft_list==gamma
    total_steps = 0     # Total steps for samples where draft_list==gamma  
    total_time = 0      # Total time across all batches
    valid_samples = 0   # Number of samples where draft_list==gamma
    
    for n in range(len(count["draft_eval"])):
        # Extract arrays for current batch
        draft_list = np.array(count["draft_eval"][n])
        target_list = np.array(count["target_eval"][n])
        step_list = np.array(count["total_step"][n])
        sample_list = np.array(count["sample_length"][n])
        
        # Filter samples where draft_list equals gamma (following compute_speculative_stats.py paradigm)
        mask = draft_list == gamma
        
        # Accumulate metrics only for valid samples
        total_tokens += sample_list[mask].sum()
        total_steps += step_list[mask].sum()
        total_time += float(count["time"][n])
        valid_samples += mask.sum()
    
    # Calculate the two paper metrics
    block_eff = total_tokens / total_steps if total_steps > 0 else 0
    decode_speed = total_tokens / total_time if total_time > 0 else 0
    
    block_efficiency.append(block_eff)
    decoding_speed.append(decode_speed)
    sample_counts.append(valid_samples)
    
    print(f"{method_labels[i]}:")
    print(f"  Valid samples (draft_list==gamma): {valid_samples}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total steps: {total_steps}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Block efficiency (token/step): {block_eff:.4f}")
    print(f"  Decoding speed (token/second): {decode_speed:.2f}")
    print()

# Convert to numpy arrays for easier manipulation
block_efficiency = np.array(block_efficiency)
decoding_speed = np.array(decoding_speed)
sample_counts = np.array(sample_counts)

# Print summary
print("="*60)
print("PAPER METRICS SUMMARY")
print("="*60)
print(f"{'Method':<20} {'Block Efficiency':<18} {'Decoding Speed':<15} {'Valid Samples':<12}")
print(f"{'(token/step)':<20} {'(token/second)':<18}")
print("-"*60)
for i, label in enumerate(method_labels):
    print(f"{label:<20} {block_efficiency[i]:<18.4f} {decoding_speed[i]:<15.2f} {sample_counts[i]:<12}")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

x = np.arange(len(method_labels))
width = 0.6

# Plot 1: Block Efficiency (token/step)
bars1 = ax1.bar(x, block_efficiency, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax1.set_ylabel('Block Efficiency (token/step)')
ax1.set_title('Block Efficiency by Method')
ax1.set_xticks(x)
ax1.set_xticklabels(method_labels, rotation=45, ha='right')
ax1.bar_label(bars1, fmt='%.3f', padding=3)

# Plot 2: Decoding Speed (token/second)
bars2 = ax2.bar(x, decoding_speed, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax2.set_ylabel('Decoding Speed (token/second)')
ax2.set_title('Decoding Speed by Method')
ax2.set_xticks(x)
ax2.set_xticklabels(method_labels, rotation=45, ha='right')
ax2.bar_label(bars2, fmt='%.1f', padding=3)

plt.tight_layout()
plt.savefig("BL_comp_72B.png", dpi=300, bbox_inches='tight')
print(f"\nVisualization saved as 'paper_metrics_comparison.png'")

# Save results to a clean format for paper
results = {
    'methods': method_labels,
    'block_efficiency_token_per_step': block_efficiency.tolist(),
    'decoding_speed_token_per_second': decoding_speed.tolist(),
    'valid_sample_counts': sample_counts.tolist(),
    'gamma_filter': gamma
}

with open('BL_comp_72B.json', 'w') as f:
    import json
    json.dump(results, f, indent=2)
print("Results saved to 'paper_metrics_results.json'")
