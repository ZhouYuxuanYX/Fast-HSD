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
with open(f"{models}naive_gamma_10_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_naive = orjson.loads(mm[:])   # mm[:] gives you a bytes object
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

with open(f"{models}backward_clever_gamma_10_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_backward_clever =orjson.loads(mm[:])   # mm[:] gives you a bytes object
    mm.close()
#
# with open(f"{models}backward_clever_approxi_gamma_10_total_counts.json", "rb") as f:
#     mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
#     counts_backward_clever_approxi = orjson.loads(mm[:])   # mm[:] gives you a bytes object
#     mm.close()

with open(f"{models}blockwise_gamma_10_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_blockwise = orjson.loads(mm[:])   # mm[:] gives you a bytes object
    mm.close()

end = time.time()

print(end-start)


counts = [
    counts_naive,
    # counts_backward, counts_backward_recursive,
          counts_backward_clever,
          # counts_backward_clever_approxi,
          counts_blockwise,
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
    for n in range(len(count["draft_eval"])):
        # exclude the draft lengths<10 cases for a fair comparison
        # count["draft_eval"][n][count["draft_eval"][n]==10]
        draft_list = np.array(count["draft_eval"][n])
        target_list = np.array(count["target_eval"][n])
        step_list = np.array(count["total_step"][n])
        sample_list = np.array(count["sample_length"][n])


        draft += draft_list[draft_list==10].sum()/len(draft_list[draft_list==10])
        target += target_list[draft_list==10].sum()/len(target_list[draft_list==10])
        step += step_list[draft_list==10].sum()/len(step_list[draft_list==10])
        sample += sample_list[draft_list==10].sum()/len(sample_list[draft_list==10])
        time_ += float(count["time"][n])/len(sample_list)


    draft_eval.append(draft/len(count["draft_eval"]))
    target_eval.append(target/len(count["draft_eval"]))
    total_step.append(step/len(count["draft_eval"]))
    sample_length.append(sample/len(count["draft_eval"]))
    times.append(-time_/len(count["draft_eval"]))

draft_eval = np.array(draft_eval)
target_eval = np.array(target_eval)
total_step = np.array(total_step)
sample_length = np.array(sample_length)
times = np.array(times)




x = np.arange(len(counts))  # [0, 1, 2, 3]

width = 0.3  # Width of the bars

# Create plot
fig, ax = plt.subplots()
bar1 = ax.bar(x - width, [a.mean() for a in times], width, label='List 1')
bar2 = ax.bar(x, [a.mean() for a in target_eval], width, label='List 2')
bar3 = ax.bar(x + width, [a.mean() for a in sample_length], width, label='List 3')

# Add labels, title, and legend
ax.set_ylabel('Scores')
ax.set_title('Comparison of Two Lists')
ax.set_xticks(x)
# ax.set_xticklabels(labels)
ax.legend()

# Optional: Add bar labels
ax.bar_label(bar1, padding=3)
ax.bar_label(bar2, padding=3)
ax.bar_label(bar3, padding=3)

plt.tight_layout()
plt.savefig(f"{models}compare_efficiency_gamma_10.png")
