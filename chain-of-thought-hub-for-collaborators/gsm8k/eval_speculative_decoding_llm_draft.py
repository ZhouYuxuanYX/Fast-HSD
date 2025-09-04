import re
import os
# !!!!!!!!!!!!must set the environment variable before importing transformers, otherwise it won't work!!!!!!!!
######### use the local cache on haicore
# os.environ['HF_HOME'] = '/p/scratch/hai_llm_diversity/cache/transformers'
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
from tqdm import tqdm
from torch import LongTensor, FloatTensor, eq, device
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, MaxLengthCriteria, StoppingCriteria, StoppingCriteriaList
import datasets
import numpy as np
import torch
import random

class BSD():
    def __init__(self, target_model, draft_model, draft_length, prefix):
        self.target_model = target_model
        self.draft_model = draft_model
        self.draft_length = draft_length
        self.prefix = prefix
        # a sequence of draft token IDs at each position
        self.draft_tokens = []
        # squeeze the batch dimension, so that all the following tensors are 1d
        # all possible tokens' associate probabilities given a subsequence of draft as prefix
        self.draft_dists = None
        self.target_dists = None
        self.prob_ratios = None
        self.draft_probs = None
        self.target_probs = None
        self.target_tokens = None
        self.generated_tokens = []
        self.accepted_draft_length = []
        self.evaluation_counts = []
        # probabilities of accepted + resampled tokens
        self.gen_p = None
        self.gen_q = None
        self.gen_r = None
        self.total_gen = []


    def sample_draft(self):
        for i in range(self.draft_length):
            draft_tokens, draft_dists = self.draft_model(self.prefix + self.draft_tokens)
            self.draft_tokens.append(draft_tokens[0, -1])
            self.draft_dists.append(draft_dists[0, -1])

    def evaluate_draft(self):
        self.target_tokens, self.target_dists = self.target_model(self.prefix + self.draft_tokens)
        draft_inds = (torch.arange(self.draft_length),self.draft_tokens)
        self.draft_probs = self.draft_dists[draft_inds]
        self.target_probs = self.target_dists[draft_inds]
        self.prob_ratio = self.target_probs/self.draft_probs

    def step(self):
        self.sample_draft()
        self.evaluate_draft()
        self.verify()

    def verify(self):
        step = self.draft_length - 1
        while True:
            r = torch.cumprod(self.prob_ratios[:step+1])[-1]
            q = torch.cumprod(self.draft_probs[:step+1])[-1]
            p = torch.cumprod(self.target_probs[:step+1])[-1]

            # only evaluate this at the gamma step
            if step == self.draft_length - 1:
                if r>=1:
                    self.generated_tokens.extend(self.draft_tokens)
                    self.generated_tokens.append(self.target_tokens[-1])
                    self.accepted_draft_length.append(step + 1)
                    self.evaluation_counts.append(1)
                    break
                else:
                    h = random.uniform(1, 0)
                    if h <= r:
                        self.generated_tokens.extend(self.draft_tokens)
                        self.generated_tokens.append(self.target_tokens[-1])
                        self.accepted_draft_length.append(step + 1)
                        self.evaluation_counts.append(1)
                        break
            # otherwise the sequence is rejected, then we either resample or go back in an iterative manner
            r_previous = torch.cumprod(self.prob_ratios[:step])[-1] if step>0 else 1
            r_prime = r_previous*self.target_dists[step]/self.draft_dists[step]

            p_previous = torch.cumprod(self.target_probs[:step])[-1] if step>0 else 1
            p_prime = p_previous*self.target_dists[step]

            q_previous = torch.cumprod(self.draft_probs[:step])[-1] if step>0 else 1
            q_prime = q_previous*self.draft_dists[step]

            r_plus = r_prime[r_prime>=1]
            r_minus = r_prime[r_prime<1]

            r_plus_sum = r_plus.sum()
            r_minus_sum = r_minus.sum()

            q_plus = q_prime[r_prime>=1] if len(q_prime[r_prime>=1])!=0 else 0
            q_minus = q_prime[r_prime<1] if len(q_prime[r_prime<1])!=0 else 0

            p_res = torch.maximum(p_prime - q_prime, 0)/max(((r_plus-1)*q_plus).sum(),((1-r_minus)*q_minus).sum())
            p_res_sum = p_res.sum()

            p_back = 1 - p_res_sum

            sample = np.random.choice(torch.arange(p_res.shape[-1]).tolist() + [-1], 1, p=p_res.tolist() + [p_back])

            if sample[0] != -1:
                # the token at the current step is not included by using :step
                self.generated_tokens.extend(self.draft_tokens[:step])
                self.generated_tokens.append(sample[0])
                self.accepted_draft_length.append(step)
                self.evaluation_counts.append(self.draft_length-step)

                self.gen_p = self.target_probs[:step].tolist() + [self.target_dists[step][sample[0]]]
                self.gen_q = self.draft_probs[:step].tolist() + [self.draft_dists[step][sample[0]]]
                self.gen_r = self.prob_ratios[:step].tolist() + [self.target_dists[step][sample[0]]/self.draft_dists[step][sample[0]]]

                while step < self.draft_length-1:
                # original sequence till one step before the current step, and the token at the current step is the resampled token
                    q_current = torch.cumprod(self.gen_q)[-1]
                    p_current = torch.cumprod(self.gen_p)[-1]
                    r_current = torch.cumprod(self.gen_r)[-1]

                    target_tokens, target_dists = self.target_model(self.prefix+self.generated_tokens)
                    draft_tokens, draft_dists = self.draft_model(self.prefix+self.generated_tokens)
                    q_next = q_current * draft_dists[:, -1]
                    p_next = p_current * target_dists[:, -1]
                    r_next = r_current * target_dists[:, -1]/draft_dists[:, -1]

                    p_res_next =  torch.maximum(p_next - q_next, 0)
                    p_next_minus = np.maximum(q_next - p_next, 0)

                    p_res_next = p_res_next/max(p_res_next.sum(), p_next_minus.sum())

                    sample = np.random.choice(torch.arange(p_res_next.shape[-1]).tolist(), 1, p=p_res_next.tolist())
                    self.generated_tokens.append(sample)
                    step+=1
                break
            step-=step

        self.total_gen.extend(self.generated_tokens)


########### it is super slow to use cpu ##########
########## takes 4.5 h to use Llama-3-7B on the full test set of 1319 test questions#########

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if (input_ids[0][-len(stop_ids[0]):] == stop_ids[0]).all():
                print("input tail ids:", input_ids[0][-len(stop_ids[0]):])
                print("stop ids:", stop_ids[0])
                return True
        return False


######### use only a fraction to speed up the development process #########

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# for mac
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
gsm8k = load_dataset('gsm8k', 'main')

gsm8k_test = gsm8k['test']

# num_samples = len(gsm8k_test['question'])
num_samples = 10

validation_index = np.load('lib_prompt/validation_index.npy')
validation_data = gsm8k['train'].select(validation_index)

tokenizer1 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model1 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", device_map='auto')
tokenizer2 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
model2 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", device_map='auto')

# check the default configuration of the pretained model
# print(model.generation_config)
"""
GenerationConfig {
  "bos_token_id": 128000,
  "do_sample": true,
  "eos_token_id": 128001,
  "max_length": 4096,
  "temperature": 0.6,
  "top_p": 0.9
}
"""

# exit()

q_lens = [len(d['question']) for d in gsm8k['train']]
print(np.percentile(q_lens, [50, 80, 90, 95, 99, 100]))

print(np.argmax(q_lens))

input_text = gsm8k['train'][3331]['question']

print(gsm8k['train'][3331]['answer'])

# verified this by checking gpt3.5turbo evaluation script, prompt_complex is named as prompt_hardest.txt
prompt_complex = open('lib_prompt/prompt_hardest.txt').read()
prompt_q = prompt_complex + '\nQuestion: ' + input_text + '\n'
print(prompt_q)

input_ids = tokenizer1(prompt_q, return_tensors="pt").input_ids.to(device)
print(input_ids.size())
# exit()
bsd = BSD(model2, model1, draft_length=3, prefix=input_ids)

print(input_ids.device)
print(model1.device)
# exit()
# Llama3 has a context window of 8k tokens
outputs = model1.generate(input_ids, max_new_tokens=256)

# while len(bsd.total_gen)

# print("##############check#################")
# print(tokenizer.decode(input_ids[0]))
# print("###################")
# print(tokenizer.decode(outputs[0]))
# print(outputs[0].size())

# exit()

# for n, m in model.named_parameters():
#     print(n, m.device)

with torch.no_grad():
    outputs = model1.generate(input_ids,
                             do_sample=True,
                             max_new_tokens=256,
                             output_scores=True,
                             return_dict_in_generate=True,
                             num_return_sequences=2
                            )

print(outputs.keys())

print(tokenizer1.decode(outputs['sequences'][0]))

print(tokenizer1.decode(outputs['sequences'][1]))

# print(tokenizer.decode(outputs['sequences'][2]))

print(len(outputs['scores']))

print(outputs['scores'][0].size())

print(torch.softmax(outputs['scores'][-1], dim=-1).topk(3))

print(tokenizer1.convert_ids_to_tokens([1]))

print(torch.softmax(outputs['scores'][1], dim=-1).topk(3))

print(tokenizer1.convert_ids_to_tokens([31]))

def test_answer(pred_str, ans_str):
    pattern = '\d*\.?\d+'
    pred = re.findall(pattern, pred_str)
    if (len(pred) >= 1):
        # print(pred_str)
        pred = pred[-1]
        print(pred)
        gold = re.findall(pattern, ans_str)
        # print(ans_str)
        gold = gold[-1]
        print(gold)
        return pred == gold
    else:
        return False

def parse_pred_ans(filename):
    with open(filename) as fd:
        lines = fd.readlines()
    am, a = None, None
    num_q, acc = 0, 0
    current_mode = 'none'
    questions = []
    ans_pred = []
    ans_gold = []
    for l in lines:
        # print("check l")
        # print(l)
        # print("#######################")

        if (l.startswith('Q: ')):
            if (am is not None and a is not None):

                ## after adding the stopping criteria, it is correct now for LLama-3
                # print("check saved q, am and a")
                # print(q)
                # print("####################")
                # print(am)
                # print("###################")
                # print(a)
                # exit()

                questions.append(q)
                ans_pred.append(am)
                ans_gold.append(a)
                if (test_answer(am, a)):
                    acc += 1
            current_mode = 'q'
            q = l
            num_q += 1
        elif (l.startswith('A_model:')):
            current_mode = 'am'
            am = l
        elif (l.startswith('A:')):
            current_mode = 'a'
            a = l
        else:
            if (current_mode == 'q'):
                q += l
            elif (current_mode == 'am'):
                am += l
            elif (current_mode == 'a'):
                a += l
            else:
                raise ValueError(current_mode)

    questions.append(q)
    ans_pred.append(am)
    ans_gold.append(a)
    if (test_answer(am, a)):
        acc += 1
    print('num_q %d correct %d ratio %.4f' % (num_q, acc, float(acc / num_q)))
    return questions, ans_pred, ans_gold

i = 0

stop_list = [" \n\nQuestion:", " \n\n", "\n\n"]
stop_token_ids = [tokenizer1(x, return_tensors='pt', add_special_tokens=False)['input_ids'] for x in stop_list]
stop_token_ids = [LongTensor(x).to(device) for x in stop_token_ids]

with open('outputs/test_Llama3-8B_complex.txt', 'w') as fd:
    for q, a in tqdm(zip(gsm8k_test['question'][:num_samples], gsm8k_test['answer'][:num_samples]),
                     total=num_samples):
        prompt_q = prompt_complex + '\nQuestion: ' + q + '\n'

        input_ids = tokenizer1(prompt_q, return_tensors="pt").input_ids.to(device)
        # the model will continue creating new question and answer patterns after answering the real question
        # add "\n\n" as stopping criterion to avoid such a problem

        stopping_criteria = StoppingCriteriaList([StopOnTokens()])
        outputs = model1.generate(input_ids, max_new_tokens=256, stopping_criteria=stopping_criteria)
        ans_ = tokenizer1.decode(outputs[0])
        # print("check q, ans_ and a")
        # print(q)
        # print("###########################")
        # print(ans_)
        # # the model will continue create new question and answer patterns after answering the real question
        # print("###########################")
        # print(a)
        # exit()
        fd.write('Q: %s\nA_model:\n%s\nA:\n%s\n\n' % (q, ans_, a))

_, _, _ = parse_pred_ans('outputs/test_Llama3-8B_complex.txt')

# prompt_original = open('lib_prompt/prompt_original.txt').read()

# i = 0
# with open('outputs/test_Llama3-8B_original.txt', 'w') as fd:
#     for q, a in tqdm(zip(gsm8k_test['question'], gsm8k_test['answer']),
#                      total=len(gsm8k_test['question'])):
#         prompt_q = prompt_original + '\nQuestion: ' + q + '\n'
#         input_ids = tokenizer(prompt_q, return_tensors="pt").input_ids.to("cuda:0")
#         outputs = model.generate(input_ids, max_new_tokens=256)
#         ans_ = tokenizer.decode(outputs[0])
#         fd.write('Q: %s\nA_model:\n%s\nA:\n%s\n\n' % (q, ans_, a))
# _, _, _ = parse_pred_ans('outputs/test_Llama3-8B_original.txt')
#[ prompt_simple = open('lib_prompt/prompt_simple.txt').read()
