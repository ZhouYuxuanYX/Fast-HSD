import orjson
import numpy as np
import matplotlib.pyplot as plt
import mmap
import time
# too large to fit into the local machine ram
# orjson is a super-fast alternative to the standard json module.
# It’s written in Rust and much faster for both reading and writing.
# this is faster than multiprocessing, 820s vs 1729.9 (multiprocessing) seconds!!!
# memoryview is also slower than mm[:]

models = "Qwen_72B_0.5B_"

start =time.time()
with open(f"{models}naive_gamma_10_topp_1.0_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_naive = orjson.loads(mm[:])   # mm[:] gives you a bytes object
    mm.close()

with open(f"{models}blockwise_gamma_10_topp_1.0_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_blockwise = orjson.loads(mm[:])   # mm[:] gives you a bytes object
    mm.close()
#
# with open(f"{models}backward_gamma_10_total_counts.json", "rb") as f:
#     mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
#     counts_backward =orjson.loads(mm[:])   # mm[:] gives you a bytes object
#     mm.close()
#
# with open(f"{models}backward_recursive_gamma_10_total_counts.json", "rb") as f:
#     mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
#     counts_backward_recursive = orjson.loads(mm[:])   # mm[:] gives you a bytes object
#     mm.close()

with open(f"{models}backward_gamma_10_topp_1.0_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_backward =orjson.loads(mm[:])   # mm[:] gives you a bytes object
    mm.close()
#
with open(f"{models}backward_clever_approxi_gamma_10_topp_1.0_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_backward_clever_approxi = orjson.loads(mm[:])   # mm[:] gives you a bytes object
    mm.close()



end = time.time()

print(end-start)


counts = [
    counts_naive,
    counts_blockwise,
    counts_backward, 
    counts_backward_clever_approxi,
]

# Labels for the methods
method_labels = [
    'Naive',
    'Blockwise', 
    'Backward',
    'Backward Clever Approxi'
]


draft_eval = []

target_eval = []

total_step = []

sample_length = []

times = []

for count in counts:
    draft = 0
    target = 0
    step = 0
    sample = 0
    time_ = 0
    valid_samples = 0  # Count of samples that have draft_length == 10
    
    for n in range(len(count["draft_eval"])):
        # exclude the draft lengths<10 cases for a fair comparison
        draft_list = np.array(count["draft_eval"][n])
        target_list = np.array(count["target_eval"][n])
        step_list = np.array(count["total_step"][n])
        sample_list = np.array(count["sample_length"][n])

        # Only compute averages if there are elements equal to 10 to avoid division by zero
        mask = draft_list == 10
        if mask.sum() > 0:  # Check if there are any elements equal to 10
            draft += draft_list[mask].sum() / mask.sum()
            target += target_list[mask].sum() / mask.sum()
            step += step_list[mask].sum() / mask.sum()
            sample += sample_list[mask].sum() / mask.sum()
            valid_samples += 1
        
        time_ += float(count["time"][n])/len(sample_list)

    # Use valid_samples for averaging instead of total samples to avoid bias
    if valid_samples > 0:
        draft_eval.append(draft/valid_samples)
        target_eval.append(target/valid_samples)
        total_step.append(step/valid_samples)
        sample_length.append(sample/valid_samples)
    else:
        # If no valid samples, append zeros or NaN
        draft_eval.append(0)
        target_eval.append(0)
        total_step.append(0)
        sample_length.append(0)
    
    times.append(-time_/len(count["draft_eval"]))

draft_eval = np.array(draft_eval)
target_eval = np.array(target_eval)
total_step = np.array(total_step)
sample_length = np.array(sample_length)
times = np.array(times)




# Print results for inspection
print(f"\nResults summary:")
print(f"{'Method':<25} {'Draft Eval':<12} {'Target Eval':<12} {'Total Steps':<12} {'Sample Length':<14} {'Time':<10}")
print("-" * 85)
for i, label in enumerate(method_labels):
    print(f"{label:<25} {draft_eval[i]:<12.4f} {target_eval[i]:<12.4f} {total_step[i]:<12.4f} {sample_length[i]:<14.4f} {abs(times[i]):<10.4f}")

# Create subplot for better visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

x = np.arange(len(counts))  # [0, 1, 2, 3]
width = 0.6  # Width of the bars

# Plot 1: Target Evaluations (main performance metric)
bars1 = ax1.bar(x, target_eval, width, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
ax1.set_ylabel('Target Evaluations')
ax1.set_title('Target Evaluations by Method')
ax1.set_xticks(x)
ax1.set_xticklabels(method_labels, rotation=45, ha='right')
ax1.bar_label(bars1, padding=3, fmt='%.3f')

# Plot 2: Total Steps (efficiency metric)
bars2 = ax2.bar(x, total_step, width, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
ax2.set_ylabel('Total Steps')
ax2.set_title('Total Steps by Method')
ax2.set_xticks(x)
ax2.set_xticklabels(method_labels, rotation=45, ha='right')
ax2.bar_label(bars2, padding=3, fmt='%.3f')

# Plot 3: Sample Length
bars3 = ax3.bar(x, sample_length, width, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
ax3.set_ylabel('Sample Length')
ax3.set_title('Sample Length by Method')
ax3.set_xticks(x)
ax3.set_xticklabels(method_labels, rotation=45, ha='right')
ax3.bar_label(bars3, padding=3, fmt='%.3f')

# Plot 4: Time (performance metric) - using absolute values since times are negative
bars4 = ax4.bar(x, np.abs(times), width, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
ax4.set_ylabel('Time (seconds)')
ax4.set_title('Execution Time by Method')
ax4.set_xticks(x)
ax4.set_xticklabels(method_labels, rotation=45, ha='right')
ax4.bar_label(bars4, padding=3, fmt='%.3f')

plt.tight_layout()
plt.savefig(f"{models}compare_efficiency_gamma_10.png", dpi=300, bbox_inches='tight')
plt.show()
