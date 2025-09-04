import json
import numpy as np
import matplotlib.pyplot as plt

with open("qwen25_05B_3B/naive_total_counts.json", "r") as f:
    naive = json.load(f)

with open("qwen25_05B_3B/backward_total_counts.json", "r") as f:
    backward = json.load(f)

print(np.array([
    x
    for xs in naive["draft_eval"]
    for x in xs]
).sum())
# print(naive["draft_eval"])

print(np.array([
    x
    for xs in naive["target_eval"]
    for x in xs]
).sum())
# print(naive["target_eval"])

print(np.array([
    x
    for xs in naive["draft_length"]
               for x in xs]
).sum())
# print(naive["draft_length"])

print(np.array([
    x
    for xs in naive["sample_length"]
    for x in xs]
).sum())
# print(naive["sample_length"])

print(np.array([
    x
    for xs in backward["draft_eval"]
               for x in xs]
).sum())
# print(backward["draft_eval"])

print(np.array([
    x
    for xs in backward["target_eval"]
    for x in xs]
).sum())
# print(backward["target_eval"])

print(np.array([
    x
    for xs in backward["draft_length"]
               for x in xs]
).sum())
# print(backward["draft_length"])

print(np.array([
    x
    for xs in backward["sample_length"]
    for x in xs]
).sum())
# print(backward["sample_length"])

plt.figure()
# normalize to have a clear comparison, because the total number of evaluation counts are different
weights = np.ones_like(np.array([
    x
    for xs in backward["sample_length"]
    for x in xs]
))/float(len(np.array([
    x
    for xs in backward["sample_length"]
    for x in xs]
)))
plt.hist(np.array([
    x
    for xs in backward["sample_length"]
    for x in xs]
), weights=weights)

weights = np.ones_like(np.array([
    x
    for xs in naive["sample_length"]
    for x in xs]
))/float(len(np.array([
    x
    for xs in naive["sample_length"]
    for x in xs]
)))
plt.hist(np.array([
    x
    for xs in naive["sample_length"]
    for x in xs]
),
weights=weights)
plt.legend(["backward", "naive"])
plt.show()