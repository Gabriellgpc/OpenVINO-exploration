import numpy as np



def softmax(x):
    """
      A softmax function is used to convert top-k logits into a probability distribution.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    summation = e_x.sum(axis=-1, keepdims=True)
    return e_x / summation

def process_logits(cur_length, scores, eos_token_id, min_length=0):
    """
    If the minimum sequence length is not reached,
    the following code will reduce the probability of the `eos` token occurring.
    This continues the process of generating the next words.

    reduce probability for padded indicies.
    Parameters:
      cur_length - current length of input sequence
      scores - model output logits
      eos_token_id - index of end of string token in model vocab
      min_length - minimum length for appling postprocessing
    """
    if cur_length < min_length:
        scores[:, eos_token_id] = -float("inf")
    return scores

def get_top_k_logits(scores, top_k):
    """
    In Top-K sampling, we filter the K most likely next words and redistribute
    the probability mass among only those K next words.

    perform top-k sampling

    Parameters:
      scores - model output logits
      top_k - number of elements with highest probability to select
    """
    filter_value = -float("inf")
    top_k = min(max(top_k, 1), scores.shape[-1])
    top_k_scores = -np.sort(-scores)[:, :top_k]
    indices_to_remove = scores < np.min(top_k_scores)
    filtred_scores = np.ma.array(scores, mask=indices_to_remove,
                                 fill_value=filter_value).filled()
    return filtred_scores