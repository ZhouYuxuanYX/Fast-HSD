import orjson
import numpy as np
import matplotlib.pyplot as plt
import mmap
import time
import pandas as pd
import os
import glob

data = glob.glob("/storage/home/hcoda1/8/twang730/p-rg1002-0/UW/spd_icml/Fast-HSD/chain-of-thought-hub/GPQA/results/eval_BE/new_select/*.json")

# data = [data[4]]

start = time.time()
gamma = 10

counts = []
for file in data:
    with open(file, "rb") as f:

        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        counts.append(orjson.loads(mm[:]))   # mm[:] gives you a bytes object
        mm.close()

end = time.time()

print(end-start)


draft_eval = []

target_eval = []

total_step = []

sample_length = []

DS = []
lens = []
len_gammas = []

for count in counts:
    draft = 0
    target = 0
    step = 0
    sample = 0
    time_ = 0
    len_ = 0
    len_gamma = 0
    for n in range(len(count["draft_eval"])):
        # exclude the draft lengths<10 cases for a fair comparison
        # count["draft_eval"][n][count["draft_eval"][n]==10]
        draft_list = np.array(count["draft_eval"][n])
        target_list = np.array(count["target_eval"][n])
        step_list = np.array(count["total_step"][n])
        sample_list = np.array(count["sample_length"][n])
        # print("check length")
        # print(len(sample_list))
        # print(len(step_list))
        # print(count["time"][n])

        draft += draft_list[draft_list==gamma].sum()
        target += target_list[draft_list==gamma].sum()
        step += step_list[draft_list==gamma].sum()
        sample += sample_list[draft_list==gamma].sum()
        time_ += float(count["time"][n])
        len_ += len(sample_list[draft_list==gamma])
        # len_ += len(sample_list)
        len_gamma += len(sample_list)

    lens.append(len_) #total steps
    draft_eval.append(draft/len_)
    target_eval.append(target/len_)
    total_step.append(step/len_)
    sample_length.append(sample/len_) # block efficiency
    DS.append(len_/ time_ * gamma)

print("Total decoding times:", DS)
draft_eval = np.array(draft_eval)
target_eval = np.array(target_eval)
total_step = np.array(total_step)
# print("total_step", total_step)
sample_length = np.array(sample_length)
DS = np.array(DS)
# speed = []
# for l in range(len(times)):
#     speed.append(sample_length[l].sum() / times[l].sum())

# speed = np.array(speed)
# print("sample_length", sample_length)
# print("speed", speed)


for i, file in enumerate(data):
    print(file.split('/')[-1])
    print("BE:", sample_length[i])
    # print("ACC:", target_eval[i])
    print("Speed:", DS[i])
    print('---')
