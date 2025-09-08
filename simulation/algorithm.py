import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F



def clever_old(candidate_input_ids, candidate_logits, candidate_length, new_logits, lenience=None):

    q = candidate_logits
    p = new_logits

    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.

    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

    # # print("check distribution")
    # print(p[:, torch.arange(hist_length, candidate_length)].topk(10, dim=-1)[0])
    # print(q[:, torch.arange(hist_length, candidate_length)].topk(10, dim=-1)[0])

    q_previous = torch.roll(q_i, 1, 1)
    q_previous[:, 0] = 1
    log_q_previous = torch.exp(torch.log(q_previous).cumsum(1).unsqueeze(-1))
    q_next = log_q_previous * q[:, :candidate_length]

    if True:
        # do not cap the previous ratio
        if False:
            # p_i corresponds to marginal probability
            p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

            p_previous = torch.roll(p_i, 1, 1)
            # # print(p_previous.shape)
            # exit()
            p_previous[:, 0] = 1
            p_next = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1) * p[:, :candidate_length]

        # cap the entire prefix
        else:
            # p_i corresponds to marginal probability
            p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

            p_previous = torch.roll(p_i, 1, 1)
            # in this case, we compensate the subbranch X^i with prefix r^i-1>1 cleverly,
            # then there's no need to do forward sampling
            p_previous[:, 0] = 1

            # i want to control the joint prob ratio of the preceding draft tokens to be <=1,
            log_p_previous = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1)
            p_next = torch.minimum(log_p_previous, log_q_previous) * p[:, :candidate_length]

    else:
        # p_i corresponds to marginal probability
        p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

        p_previous = torch.roll(p_i, 1, 1)
        # # print(p_previous.shape)
        # exit()
        p_previous[:, 0] = 1
        p_next = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1) * p[:, :candidate_length]

    diffs = p_next - q_next

    p_plus, p_minus = torch.clamp(diffs, min=0), torch.clamp(-diffs, min=0)

    # to avoid underflow, try to formulate using r_previous
    # for extremely small probabilities of any draft in p_i or q_i, it could be 0 simply due to default truncation sampling
    # so i have to avoid the case of devided by 0! but shall i turn off truncation sampling for the models or not? maybe i just keep
    # their default parameters

    denominator = torch.maximum(p_plus.sum(dim=-1, keepdim=True), p_minus.sum(dim=-1, keepdim=True))
    # divide by zero will cause nan value, i can simply replace nan with 0
    # however, for clever sampling
    # if there differences are 0, it can also mean that  the joint distribution is exactly the same and could simply accept the current token

    # for tokenwise verification, it will never happen when both p_next and q_next=0, because the draft token is sampled from
    # q, thus never has probability >0
    # for my backward joint verfication with forward sampling, the previous token probability could be 0 (accumulated candidates contain resampled token in the forward sampling process) and then the joint probability q_i will always be
    # zero after this token, and a new token sampled from q could also be zero for p, if truncation sampling are applied for p,
    # in this case, we have two zero joint probabilities, causing the problem to happen
    p_primes = torch.nan_to_num(p_plus / denominator)

    # for recursive backward speculative, i actually have to reset the ratio of already accepted tokens to 1,

    # compute the residual probabilities for stepping back
    # for clever backward, we could accept the token at an intermediate step if p = q,
    # sine we assume p_previous always <=1 for clever backward, in this case p_primes.sum()=0, but we should not step back

    step_back_probs = 1 - p_primes.sum(dim=-1)


    # this is needed for clever algorithm, to avoied when p_prime=0, when i forced p_previous=q_previous when p_previous>q_previsou
    # and p_current = q_current
    step_back_probs[(p_i / q_i).cumprod(1) >= 1] = 0

    # randomly sample if stepping back, i.e., neither accepted, nor resampled
    uniform_rand = torch.rand_like(step_back_probs)

    step_back = uniform_rand < step_back_probs

    stop_positions = candidate_length - 1 - torch.flip(~step_back, [-1]).max(-1, keepdim=True)[1]

    select = torch.zeros_like(step_back).to(step_back.device)

    # apply cumprod on the ratio instead of the raw probabilities to avoid underflow
    probability_ratio = (p_i / q_i).cumprod(1).unsqueeze(-1)
    # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
    # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
    # (= keep with p = probability_ratio). Keep all the tokens until the first rejection

    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio


    # only decide to accept or not at the last position based on the joint probability ratio
    # assign 0 to all positions when the full draft is rejected, otherwise assign 1 to the rest of the positions
    select[torch.arange(p_primes.shape[0]), stop_positions] = ~is_accepted[:, -1:]
    is_accepted = 1 - torch.cumsum(select, dim=-1)

    #### assume batch_size=1 for the current implementation
    n_matches = is_accepted.sum().item()



    # for clever, p_prime (regard previous ratio as 1) could be 0, it means p=q, and we don't need to resample
    if n_matches < candidate_length and p_primes[:, n_matches].sum() == 0:
        n_matches += 1
        valid_tokens = new_candidate_input_ids[:, : n_matches]

    else:
        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        gamma = candidate_logits.shape[1]
        # # print("check gamma")
        # print(gamma)
        # # print("check p")
        # print(p.shape)
        # print(candidate_length)
        p_n_plus_1 = p[:, candidate_length, :]
        if n_matches < gamma:
            # then we don't have a bonus token, and start the resample from the step where we don't step back, i.e., n_matches+1
            # # print("check p_prime")
            # print(p_primes)
            # don't use in_place operation! because slicing is not creating a new tensor
            p_prime = p_primes[:, n_matches]

            p_prime = p_prime.div(p_prime.sum())

        else:
            p_prime = p_n_plus_1
            # # # print("else")
            # # print(p_prime.shape)

        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

        # The selected tokens include the matches (if any) plus the next sampled tokens
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t
    return valid_tokens, n_matches, p_i, q_i, probability_ratio
# Method 1:
def clever(candidate_input_ids, candidate_logits, candidate_length, new_logits, lenience=None):
    q = candidate_logits
    p = new_logits

    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.

    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

    # # print("check distribution")
    # print(p[:, torch.arange(hist_length, candidate_length)].topk(10, dim=-1)[0])
    # print(q[:, torch.arange(hist_length, candidate_length)].topk(10, dim=-1)[0])

    q_previous = torch.roll(q_i, 1, 1)
    q_previous[:, 0] = 1
    log_q_previous = torch.exp(torch.log(q_previous).cumsum(1).unsqueeze(-1))
    q_next = log_q_previous * q[:, :candidate_length]

    # ========================== do not cap the previous ratio ========================== 

    if True:
        # do not cap the previous ratio
        if True:
            # p_i corresponds to marginal probability
            p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

            p_previous = torch.roll(p_i, 1, 1)
            # # print(p_previous.shape)
            # exit()
            p_previous[:, 0] = 1
            p_next = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1) * p[:, :candidate_length]

        # cap the entire prefix
        elif False:
            # p_i corresponds to marginal probability
            p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

            p_previous = torch.roll(p_i, 1, 1)
            # in this case, we compensate the subbranch X^i with prefix r^i-1>1 cleverly,
            # then there's no need to do forward sampling
            p_previous[:, 0] = 1

            # i want to control the joint prob ratio of the preceding draft tokens to be <=1,
            log_p_previous = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1)
            p_next = torch.minimum(log_p_previous, log_q_previous) * p[:, :candidate_length]

        # cap the maximum prefix ratio
        else:
            # p_i corresponds to marginal probability
            p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

            p_previous = torch.roll(p_i, 1, 1)
            # in this case, we compensate the subbranch X^i with prefix r^i-1>1 cleverly,
            # then there's no need to do forward sampling
            p_previous[:, 0] = 1

            # i want to control the joint prob ratio of the preceding draft tokens to be <=1,
            log_p_previous = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1)

            # p_next = torch.minimum(log_p_previous, log_q_previous) * p[:, :candidate_length]
            ratio = log_p_previous / log_q_previous

            previous_max = 1
            new_p_previous = torch.ones_like(log_p_previous).to(log_p_previous.device)
            for k in range(candidate_length):
                if ratio[:, k] > previous_max:
                    previous_max = ratio[:, k]

                new_p_previous[:, k] = log_p_previous[:, k] / previous_max

            p_next =  new_p_previous * p[:, :candidate_length]

    else:
        # p_i corresponds to marginal probability
        p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

        p_previous = torch.roll(p_i, 1, 1)
        # # print(p_previous.shape)
        # exit()
        p_previous[:, 0] = 1
        p_next = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1) * p[:, :candidate_length]

    diffs = p_next - q_next
    p_plus, p_minus = torch.clamp(diffs, min=0), torch.clamp(-diffs, min=0)

    denominator = torch.maximum(p_plus.sum(dim=-1, keepdim=True), p_minus.sum(dim=-1, keepdim=True))
    p_primes = torch.nan_to_num(p_plus / denominator)

    step_back_probs = 1 - p_primes.sum(dim=-1)
    step_back_probs[(p_i / q_i).cumprod(1) >= 1] = 0

    uniform_rand = torch.rand_like(step_back_probs)

    step_back = uniform_rand < step_back_probs

    stop_positions = candidate_length - 1 - torch.flip(~step_back, [-1]).max(-1, keepdim=True)[1]

    select = torch.zeros_like(step_back).to(step_back.device)
    probability_ratio = (p_i / q_i).cumprod(1).unsqueeze(-1)

    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio
    select[torch.arange(p_primes.shape[0]), stop_positions] = ~is_accepted[:, -1:]
    is_accepted = 1 - torch.cumsum(select, dim=-1)
    n_matches = is_accepted.sum().item()
    if n_matches < candidate_length and p_primes[:, n_matches].sum() == 0:
        n_matches += 1
        valid_tokens = new_candidate_input_ids[:, : n_matches]

    else:
        gamma = candidate_logits.shape[1]
        p_n_plus_1 = p[:, candidate_length, :]
        if n_matches < gamma:
            p_prime = p_primes[:, n_matches]

            p_prime = p_prime.div(p_prime.sum())

        else:
            p_prime = p_n_plus_1

        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t


    return valid_tokens, n_matches

# Method 2: Mock a "lenient" acceptance (e.g., 80% chance to keep going on mismatch)
def block(candidate_input_ids, candidate_logits, candidate_length, new_logits, lenience=None):
    q = candidate_logits
    q = F.pad(q, pad=(0, 0, 0, 1), mode='constant', value=0)

    # print("debug")
    # print(q[:, -1].sum())

    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    token_sequence = []  # Will include the token sequence we return

    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p = new_logits
    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    probability_ratio = p_i / q_i
    accept_probability = 1

    vocab_size = q.shape[-1]

    reject_probs = []
    for token_index in range(candidate_length + 1):

        # Unnormalized residual probability
        sampling_weights = torch.maximum(torch.zeros_like(p[:, token_index]),
                                         p[:, token_index] * accept_probability - q[:, token_index])
        # unnormalized reject probability
        reject = torch.tensor([1 - accept_probability])[None, :].to(sampling_weights.device)

        # print(weights)
        # print(weights.sum())

        # if could happen that when p exactly equals to q at every position, especially for temperature=0
        # the sampling_weights will sum to zero
        if token_index < candidate_length:
            weights = torch.cat([sampling_weights, reject], dim=-1)
            if weights.sum().item() == 0:
                # this means always accept the previous token, same effect as if chosen_token.item() < vocab_size in the other case
                valid_tokens = new_candidate_input_ids[:, :token_index + 1]
                n_matches = token_index + 1
            else:
                weights = weights / weights.sum()

                chosen_token = torch.multinomial(weights, num_samples=1).squeeze(1)[None, :]

                if chosen_token.item() < vocab_size:
                    valid_tokens = torch.cat([new_candidate_input_ids[:, :token_index], chosen_token], dim=-1)
                    n_matches = token_index

            reject_probs.append(weights[0, -1].cpu().item())
        else:
            # h_gamma = p_gamma
            u_gamma = torch.rand(1)
            is_accepted = u_gamma >= reject.cpu()

            if is_accepted:
                chosen_token = torch.multinomial(p[:, token_index], num_samples=1).squeeze(1)[None, :]

                # if the last token is eos token, then the probability will be all zero for the next token
                # however, the torch.multinomial function will ignore the error and generate random tokens, thus causing the issue

                valid_tokens = torch.cat([new_candidate_input_ids[:, :token_index], chosen_token], dim=-1)
                n_matches = token_index
            reject_probs.append(reject[0, 0].cpu().item())

        # no probability ratio for the bonus token
        if token_index < candidate_length:
            accept_probability = min(1, probability_ratio[token_index] * accept_probability)

    return valid_tokens, n_matches


def naive(candidate_input_ids, candidate_logits, candidate_length, new_logits, lenience=None):
    if lenience is None:
        lenience = 1.0
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.
    q = candidate_logits

    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p = new_logits

    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    probability_ratio = p_i / (q_i * lenience)

    # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
    # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
    # (= keep with p = probability_ratio). Keep all the tokens until the first rejection
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio

    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1

    # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
    gamma = candidate_logits.shape[1]
    p_n_plus_1 = p[:, n_matches, :]
    if n_matches < gamma:
        q_n_plus_1 = q[:, n_matches, :]
        p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
        p_prime.div_(p_prime.sum())
    else:
        p_prime = p_n_plus_1
    t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

    # The selected tokens include the matches (if any) plus the next sampled tokens
    if n_matches > 0:
        valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
    else:
        valid_tokens = t


    return valid_tokens, n_matches, p_i, q_i, probability_ratio

def FastHSD(candidate_input_ids, candidate_logits, candidate_length, new_logits, lenience=None):
    if lenience is None:
        lenience = 1.0
    q = candidate_logits
    p = new_logits

    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]

    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1) * lenience

    q_previous = torch.roll(q_i, 1, 1)
    q_previous[:, 0] = 1
    log_q_previous = torch.exp(torch.log(q_previous).cumsum(1).unsqueeze(-1))
    q_next = log_q_previous * q[:, :candidate_length]

    # ========================== do not cap the previous ratio ========================== 
    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

    p_previous = torch.roll(p_i, 1, 1)
    p_previous[:, 0] = 1
    p_next = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1) * p[:, :candidate_length]


    diffs = p_next - q_next
    p_plus, p_minus = torch.clamp(diffs, min=0), torch.clamp(-diffs, min=0)

    denominator = torch.maximum(p_plus.sum(dim=-1, keepdim=True), p_minus.sum(dim=-1, keepdim=True))
    p_primes = torch.nan_to_num(p_plus / denominator)

    step_back_probs = 1 - p_primes.sum(dim=-1)
    step_back_probs[(p_i / q_i).cumprod(1) >= 1] = 0

    uniform_rand = torch.rand_like(step_back_probs)

    step_back = uniform_rand < step_back_probs

    stop_positions = candidate_length - 1 - torch.flip(~step_back, [-1]).max(-1, keepdim=True)[1]

    select = torch.zeros_like(step_back).to(step_back.device)
    probability_ratio = (p_i / q_i).cumprod(1).unsqueeze(-1)

    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio
    
    select[torch.arange(p_primes.shape[0]), stop_positions] = ~is_accepted[:, -1:]
    is_accepted = 1 - torch.cumsum(select, dim=-1)
    n_matches = is_accepted.sum().item()
    if n_matches < candidate_length and p_primes[:, n_matches].sum() == 0:
        n_matches += 1
        valid_tokens = new_candidate_input_ids[:, : n_matches]

    else:
        gamma = candidate_logits.shape[1]
        p_n_plus_1 = p[:, candidate_length, :]
        if n_matches < gamma:
            p_prime = p_primes[:, n_matches]

            p_prime = p_prime.div(p_prime.sum())

        else:
            p_prime = p_n_plus_1

        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t


    return valid_tokens, n_matches, p_i, q_i, probability_ratio

def FastHSD_new(candidate_input_ids, candidate_logits, candidate_length, new_logits, lenience=None):
    if lenience is None:
        lenience = 1.0
    if 1:
        q = candidate_logits.softmax(dim=-1).double() if "mps" not in str(
            candidate_logits.device) else candidate_logits.softmax(dim=-1)

        p = new_logits.softmax(dim=-1).double() if "mps" not in str(new_logits.device) else new_logits.softmax(dim=-1)
        hist_length=0
        for length in [0]:
            hist_length+=length
            new_candidate_input_ids = candidate_input_ids[:, -(candidate_length-hist_length):]

            q_i = q[:, torch.arange(hist_length, candidate_length), new_candidate_input_ids].squeeze(1)
            q_previous = torch.roll(q_i, 1, 1)
            q_previous[:, 0] = 1
            log_q_previous = torch.exp(torch.log(q_previous).cumsum(1).unsqueeze(-1))
            q_next = log_q_previous *q[:, hist_length:candidate_length]

            if hist_length > 0:

                def zero_after_first_zero(x):
                    zero_mask = (x == 0)
                    first_zero_idx = zero_mask.float().cumsum(dim=1).clamp(max=1)
                    keep_mask = (first_zero_idx == 0).float().cumsum(dim=1).clamp(max=1)
                    return x * keep_mask
                p_new = p_primes[:, length:candidate_length - hist_length + length].clone()

                p_new_sum = p_new.sum(-1, keepdim=True)
                p_new_sum[p_new_sum==0] = 1
                p_new = p_new.div(p_new_sum)
                p_i = p_new[:, torch.arange(candidate_length-hist_length), new_candidate_input_ids].squeeze(1)
                p_i = zero_after_first_zero(p_i)
                p_previous = torch.roll(p_i, 1, 1)
                p_previous[:, 0] = 1
                p_next = p_previous.cumprod(-1).unsqueeze(-1) * p_new
            else:
                p_i = p[:, torch.arange(hist_length, candidate_length), new_candidate_input_ids].squeeze(1)
                p_previous = torch.roll(p_i, 1, 1)
                p_previous[:, 0] = 1
                p_next = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1)*p[:, :candidate_length]
            diffs = p_next - q_next
            p_plus, p_minus = torch.clamp(diffs, min=0), torch.clamp(-diffs, min=0)
            denominator = torch.maximum(p_plus.sum(dim=-1, keepdim=True), p_minus.sum(dim=-1, keepdim=True))
            p_primes = torch.nan_to_num(p_plus / denominator)
        step_back_probs = 1 - p_primes.sum(dim=-1)
        uniform_rand = torch.rand_like(step_back_probs)
        step_back = uniform_rand < step_back_probs
        stop_positions = candidate_length-hist_length - 1 - torch.flip(~step_back, [-1]).max(-1, keepdim=True)[1]
        select = torch.zeros_like(step_back).to(step_back.device)
        probability_ratio = (p_i / q_i).cumprod(1).unsqueeze(-1)
        r_i = torch.rand_like(probability_ratio)
        is_accepted = r_i <= probability_ratio
        select[torch.arange(p_primes.shape[0]), stop_positions] = ~is_accepted[:, -1:]
        is_accepted = 1 - torch.cumsum(select, dim=-1)
        n_matches = is_accepted.sum().item()
        if n_matches == candidate_length-hist_length:
            n_matches -= 1
            valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
        else:
            gamma = candidate_logits.shape[1]
            p_n_plus_1 = p[:, candidate_length, :]
            if n_matches < gamma - hist_length:
                p_prime = p_primes[:, n_matches]
                p_prime = p_prime.div(p_prime.sum())
            else:
                p_prime = p_n_plus_1
            t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]
            if n_matches > 0:
                valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
            else:
                valid_tokens = t

        # return valid_tokens, n_matches

        return valid_tokens, n_matches, p_i, q_i, probability_ratio








def mentored_decoding(candidate_input_ids, candidate_logits, candidate_length, new_logits, target_dkl=0.1, tol=0.2, binary_search_steps=10, lenience=None):
    #@Qw: p q notation is aligned with our paper, q is draft model, p is target model
    """
    @Qw:According to the blog:
    • ri, probability to accept the draft token i as the next token;
    • si, probability to select i as the next token if the draft token is rejected;
    • πi, the distribution resulting from the adapted rejection sampling scheme.
    """

    q_draft = candidate_logits 
    p_target = new_logits

    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]

    q_i = q_draft[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)
    p_i = p_target[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

    probability_ratio = p_i / q_i

    for i in range(candidate_length):
        # We process token by token, assuming a batch size of 1.
        q = q_draft[0, i, :]
        p = p_target[0, i, :]
        x = new_candidate_input_ids[0, i].item()

        # Generate a single random number for this token's verification step.
        random_number = torch.rand(1).item()

        # Pre acceptance check1: High-probability acceptance check (from standard SpD) 
        # r_i >= min(1, p[x]/q[x]).
        if random_number * q[x] < p[x]:
            continue # Accept and check the next token.

        # Add a small epsilon for numerical stability.
        q_stable = q + 1e-9
        p_stable = p + 1e-9
        
        p_over_q = p_stable / q_stable
        
        # Pre acceptance check2: Check if the initial KL divergence is already within the tolerated range.
        # D_KL(P || Q) = sum(p * ln(p/q))
        dkl = (p_stable * torch.log(p_over_q)).sum()
        if dkl <= target_dkl * (1 + tol):
            continue # Accept and check the next token.

        #  Binary search for alpha and beta to meet the D_KL constraint 
        sorted_p_over_q, argsort_p_over_q = torch.sort(p_over_q)
        sorted_p = p_stable[argsort_p_over_q]
        sorted_q = q_stable[argsort_p_over_q]
        
        min_alpha, max_alpha = 0.0, 1.0
        best_alpha, best_beta = 1.0, 1.0
        
        # Precompute cumulative sums for the binary search.
        cumsum_p = sorted_p.cumsum(-1)
        cumsum_q = sorted_q.cumsum(-1)
        cumsum_last_p = torch.flip(torch.flip(sorted_p, [0]).cumsum(-1), [0])
        cumsum_last_q = torch.flip(torch.flip(sorted_q, [0]).cumsum(-1), [0])
        right_side_eq = cumsum_last_p / sorted_p_over_q - cumsum_last_q

        

        for _ in range(binary_search_steps):
            alpha = (min_alpha + max_alpha) / 2
            
            # Find n1 (index where sorted_p_over_q crosses alpha)
            try:
                n1 = torch.nonzero(sorted_p_over_q <= alpha)[-1].item()
            except IndexError:
                n1 = -1

            # `one_minus_R` is the probability of rejection.
            one_minus_R = (cumsum_p[n1] if n1 > -1 else 0) - (cumsum_q[n1] / alpha if n1 > -1 and alpha > 0 else 0)

            # Find n2 (index where right_side_eq crosses one_minus_R)
            try:
                n2 = torch.nonzero(right_side_eq <= one_minus_R)[0].item()
            except IndexError:
                n2 = -1
            
            # Calculate beta based on n2 and one_minus_R
            beta_denom = one_minus_R + (cumsum_last_q[n2 - 1] if n2 > 0 else 0)
            beta = (cumsum_last_p[n2-1] if n2 > 0 else 0) / beta_denom if beta_denom > 0 else 1.0
            
            # Calculate the estimated KL divergence with the current alpha and beta
            dkl_estimate = 0
            if n1 > -1 and alpha > 0:
                dkl_estimate += torch.log(torch.tensor(alpha)) * cumsum_p[n1]
            if n2 > -1 and n2 > n1 + 1:
                # D_KL between alpha and beta region
                dkl_estimate += ((sorted_p * torch.log(p_over_q))[argsort_p_over_q][n1+1:n2]).sum()
            if n2 > -1 and beta > 0:
                dkl_estimate += torch.log(torch.tensor(beta)) * cumsum_last_p[n2]

            if dkl_estimate < target_dkl * (1 - tol):
                max_alpha = alpha
                best_alpha, best_beta = alpha, beta
            elif dkl_estimate > target_dkl * (1 + tol):
                min_alpha = alpha
            else:
                best_alpha, best_beta = alpha, beta
                break
        
        alpha, beta = best_alpha, best_beta

        # --- Final acceptance and resampling step ---
        # r_i: probability to accept the draft token `x`
        acceptance_prob = min(p[x] / (alpha * q[x]), 1.0) if q[x] > 0 and alpha > 0 else 0.0
        probability_ratio[i] = acceptance_prob

        if random_number < acceptance_prob:
            continue # Token accepted, check next one.
        else:
            # Rejection: Resample a new token and terminate this step.
            n_matches = i
            
            # s_i: probability distribution for resampling.
            one_minus_R = (cumsum_p[n1] if n1 > -1 else 0) - (cumsum_q[n1] / alpha if n1 > -1 and alpha > 0 else 0)
            s_sorted = torch.zeros_like(sorted_p)
            if one_minus_R > 0:
                s_sorted = torch.clamp(sorted_p / beta - sorted_q, min=0) / one_minus_R
            
            s = torch.zeros_like(p)
            s[argsort_p_over_q] = s_sorted
            
            # Normalize s to be a valid probability distribution
            s_sum = s.sum()
            if s_sum > 0:
                s /= s_sum
            else: # Fallback to target distribution if s is all zero
                s = p

            t = torch.multinomial(s, num_samples=1).unsqueeze(0)
            
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
            return valid_tokens, n_matches, p_i, q_i, probability_ratio

    # If the loop completes, all draft tokens were accepted.
    n_matches = candidate_length
    
    # Sample one more token from the target model's distribution.
    p_final = p_target[:, n_matches, :]
    t = torch.multinomial(p_final.squeeze(0), num_samples=1).unsqueeze(0)
    
    valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
    #return valid_tokens, n_matches
    return valid_tokens, n_matches, p_i, q_i, probability_ratio