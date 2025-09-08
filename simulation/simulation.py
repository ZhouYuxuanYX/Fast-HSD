import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from utils import tv_distance, make_pair_with_tv
from algorithm import *
import json


# Simulation runner
def simulate_spec_sampling(method_func, vocab_size, trials, draft_length, Ms_probs, Mb_probs, lenience=None, save_history_path=None):
    match_lengths = []
    count = 0
    history = {}
    for i in range(trials):
        count += 1
        if count % 1000 == 0:
            print(f"Trial {count}/{trials} for {method_func.__name__}")
            # print('[DEBUG] candidate_tokens:', candidate_tokens)
            # print('[DEBUG] candidate_probs:', candidate_probs)
            # print('[DEBUG] new_probs:', new_probs)
            # print('[DEBUG] n_matches:', n_matches)
            if save_history_path is not None:
                with open(save_history_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(history) + "\n")

        candidate_tokens = np.random.choice(vocab_size, size=draft_length, p=Ms_probs)
        candidate_tokens = torch.tensor(candidate_tokens).unsqueeze(0)
        

        candidate_probs = np.tile(Ms_probs, (draft_length, 1))
        candidate_probs = torch.tensor(candidate_probs).unsqueeze(0)
        

        new_probs = np.tile(Mb_probs, (draft_length+1, 1))
        new_probs = torch.tensor(new_probs).unsqueeze(0)
        

        _, n_matches, p, q, h = method_func(candidate_tokens, candidate_probs, draft_length, new_probs, lenience=lenience)

        match_lengths.append(n_matches)
        history['trail_{}'.format(i)] = {'p': p.squeeze().tolist(), 'q': q.squeeze().tolist(), 'h': h.squeeze().tolist(), 'tau': n_matches.squeeze().tolist() if isinstance(n_matches, torch.Tensor) else n_matches}

    return match_lengths


def plot(matches_1, matches_2, matches_3, matches_4):

    import seaborn as sns
    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 4.5))
    bins = np.arange(draft_length + 2) - 0.5

    plt.hist(matches_1, bins=bins, alpha=0.6, label="Method 1: Backward", color="skyblue", edgecolor="black", linewidth=1.2)
    plt.hist(matches_2, bins=bins, alpha=0.6, label="Method 2: Blockwise", color="salmon", edgecolor="black", linewidth=1.2)
    plt.hist(matches_3, bins=bins, alpha=0.6, label="Method 3: Tokenwise", color="grey", edgecolor="black", linewidth=1.2)
    plt.hist(matches_4, bins=bins, alpha=0.6, label="Method 4: Backward Old", color="orange", edgecolor="black", linewidth=1.2)

    plt.xlabel("Accepted Token Length", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Speculative Sampling Comparison (1000 Trials)", fontsize=14)
    plt.xticks(np.arange(0, draft_length + 2), fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)

    # Clean up spines
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig("refined_histogram.png", dpi=300)

    # plt.show()

    plt.figure(figsize=(8, 4.5))

    def plot_ccdf(data, label, color):
        sorted_data = np.sort(data)
        ccdf = 1.0 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.step(sorted_data, ccdf, where='post', label=label, color=color, linewidth=2)

    plot_ccdf(matches_1, "Method 1: Hierarchical (ours)", "skyblue")
    plot_ccdf(matches_2, "Method 2: Blockwise", "salmon")
    plot_ccdf(matches_3, "Method 3: Tokenwise", "grey")
    # plot_ccdf(matches_4, "Method 4: Backward old", "orange")

    plt.xlabel("Accepted Length τ", fontsize=12)
    plt.ylabel("P(t > τ)", fontsize=12)
    plt.title("Empirical CCDF of Accepted Lengths", fontsize=14)
    plt.xticks(np.arange(0, draft_length + 2), fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)

    # Clean up spines
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("refined_ccdf.png", dpi=300)



    # Data
    x_labels = ["tokenwise", "blockwise", "ours", "ideal"]
    y_values = [       # ideal (r_upper)
        5.137820725489648e-07,       # tokenwise
        1.1438590845320614e-05,      # blockwise
        1.657084951613326e-05 ,
        0.36611327081148815# ours
    ]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(x_labels, y_values, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])

    # Log scale for better visual contrast
    plt.yscale("log")
    plt.ylabel(r"$h(\mathbf{X}_{1:\gamma})$", fontsize="x-large")
    # plt.title("Acceptance Probability Comparison")
    # Annotate each bar
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    for bar, val in zip(bars, y_values):
        plt.text(bar.get_x() + bar.get_width() / 2, val * 1.2, f"{val:.1e}",
                ha='center', va='bottom', fontsize=16)

    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.savefig("average_probs.png")



import os
data_dir = "outputs/simulation1"

def load_jsons(data_dir):
    data = []
    for filename in os.listdir(data_dir):
        json_path = os.path.join(data_dir, filename)
        if filename.endswith(".json"):
            with open(json_path, "r", encoding="utf-8") as f:
                history = json.load(f)
                data.append(history)
    return data

def loss(input_data):
    losses = []
    length = []
    for item in input_data.values():
        p = np.array(item['p'])
        q = np.array(item['q'])
        h = np.array(item['h'])
        tau = int(item['tau'])

        loss = (p.prod() - q.prod()) * (1 - (1-h)[:tau-1].prod())
        losses.append(np.abs(loss))
        length.append(tau)
    return np.mean(losses), np.mean(length)


def main():
    # Run simulations
    # Settings
    
    trials = 3000
    task = 'simulation1'  # 'simulation1' or 'simulation2'

    # ==================================== Find distributions with desired TV ====================================
    # Option A (small TV = 0.02)
    # vocab_size = 4
    # Ms_probs = np.array([0.26, 0.25, 0.24, 0.25])
    # Mb_probs = np.array([0.24, 0.25, 0.26, 0.25])

    # Option B (medium TV ≈ 0.1666667)
    # vocab_size = 4
    # Ms_probs = np.array([1/6, 2/6, 1/3, 1/6])
    # Mb_probs = np.array([0.25, 0.25, 0.25, 0.25])

    # Option C (large TV = 0.60)
    vocab_size = 4
    Ms_probs = np.array([0.55, 0.25, 0.15, 0.05])
    Mb_probs = np.array([0.05, 0.15, 0.25, 0.55])

    # Find p, q with desired TV
    # tv_target = 0.8
    # Ms_probs, Mb_probs, tv = make_pair_with_tv(vocab_size, target_tv=tv_target, base='dirichlet', seed=42)
    # print(f"\nTarget TV={tv_target}")
    # print("Ms:", np.round(Ms_probs, 6))
    # print("Mb:", np.round(Mb_probs, 6))
    # print("Achieved TV:", tv, "Check (1 - sum min):", tv_distance(Ms_probs, Mb_probs))
    # ==================================== Find distributions with desired TV ====================================


    # ==================================== Start Simulation====================================
    if task == 'simulation1':
        lenience = [1.0, 0.8, 0.6, 0.4, 0.2]
        draft_length = 10  # changing gamma here
        for l in lenience:
            _ = simulate_spec_sampling(FastHSD, vocab_size, trials, draft_length, Ms_probs, Mb_probs, lenience=l, save_history_path=f"outputs/simulation1/lenience_{l}.json")
            # clever_old_match = simulate_spec_sampling(clever_old, vocab_size, trials, draft_length, Ms_probs, Mb_probs, lenience=l, save_history_path=f"outputs/simulation1/lenience_{l}.json")

    elif task == 'simulation2':
        lenience = None
        draft_length = 10  # changing gamma here
        naive_match = simulate_spec_sampling(naive, vocab_size, trials, draft_length, Ms_probs, Mb_probs, lenience=lenience, save_history_path="outputs/simulation2/naive_history.json")
        FastHSD_match = simulate_spec_sampling(FastHSD, vocab_size, trials, draft_length, Ms_probs, Mb_probs, lenience=lenience, save_history_path="outputs/simulation2/FastHSD_history.json")
        mentored_decoding_match = simulate_spec_sampling(mentored_decoding, vocab_size, trials, draft_length, Ms_probs, Mb_probs, lenience=lenience, save_history_path="outputs/simulation2/mentored_decoding_history.json")
    else:
        raise ValueError("Unknown task")
    # ==================================== End Simulation====================================

    # plot(matches_1, matches_2, matches_3, matches_4)
    json_data = load_jsons(data_dir)
    lenience = [0.2, 0.4, 0.6, 0.8, 1.0]
    for i, history in enumerate(json_data):
        simulated_loss, avg_length = loss(history)
        print(f"Lenience: {lenience[i]}, Simulated Loss: {simulated_loss}, Average Length: {avg_length}")




if __name__ == "__main__":
    main()
    






