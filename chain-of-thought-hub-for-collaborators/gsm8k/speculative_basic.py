import torch

def spec_sampling_method_3(candidate_input_ids, candidate_logits, candidate_length, new_logits):
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.
    q = candidate_logits.softmax(dim=-1)

    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p = new_logits.softmax(dim=-1)

    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    probability_ratio = p_i / q_i

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


    return valid_tokens, n_matches