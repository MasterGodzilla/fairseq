from pickle import TRUE
from typing import List
import sacrebleu
import torch
from time import time


def min_bayes_risk1(hypos_i, sample_size, reference_func="BLEU"):
    """
    This function changes the score of the input hypos to the expected utility function, 
    so the outside ranking function can arrange it accordingly. See

        Sampling-Based Minimum Bayes Risk Decoding for Neural Machine Translation
        https://arxiv.org/pdf/2108.04718.pdf
    
    Args:

        hypos_i (list): list of dictionary with attributes
            "tokens": torch.Tensor with len sentence_length and dtype int (token id)
            "score": torch.Tensor of size 1 float
            "attention": empty torch.Tensor
            "alignment":
            "positional_scores": torch.Tensor() len sentence_length dtype float)
        and added attributes
            "str": str
            "detok_str": str
        
        sample_size (int): the size of sample/cfg.beam_size
        
        reference_func (str, optional): choices of 
            "BLEU"
            "BEER"
            "METEOR"
            "ChrF"
    
    return:
        hypos with score as the expected utility
    """
    #Optimization: since bleu(j,j) is 100 and bleu (j,k) = bleu (k,j), we save half calculations by symmetry
    bleu_time = 0
    for j in range(sample_size):
        hypos_i[j]["expected_utility"] = 0
        for k in range(sample_size):
            if reference_func == "BLEU":
                tic = time()
                utility = sacrebleu.corpus_bleu([hypos_i[j]["detok_str"]], [[hypos_i[k]["detok_str"]]]).score
                bleu_time += time()-tic
            hypos_i[j]["expected_utility"] += (utility * torch.exp(hypos_i[k]["score"]).item())
    
    #sort expected utility in descending order
    hypos_i.sort(key = lambda hypo: hypo.get("expected_utility"),reverse=True)
    print ("bleu time:", bleu_time)
    return hypos_i