from pickle import TRUE
from typing import List
import sacrebleu
import torch
from time import time


def min_bayes_risk(hypos_i, sample_size, reference_func="BLEU"):
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
    #Optimization idea 1: since bleu(j,j) is 100 and bleu (j,k) = bleu (k,j), we save half calculations by symmetry
    #Optimization idea 2: since most hypos are equivalent, only check bleu on distinct hypos

    e = [j for j in range(sample_size)] #e[j] = k means hypothesis j is the same as hypothesis k 
    for j in range(1,sample_size):
        if hypos_i[j]["detok_str"] == hypos_i[j-1]["detok_str"]: #same as previous sentence
            e[j] = e[j-1] 
    
    utility_dict = {}
    
    bleu_time = 0
    for j in range(sample_size):
        hypos_i[j]["expected_utility"] = 0
        for k in range(sample_size):
            if not utility_dict.__contains__((e[j],e[k])):
                if reference_func == "BLEU":
                    if e[j] == e[k]:
                        utility_dict[(e[j],e[k])] = 100.0 #the same sentence gives 100.0 in BLEU score
                    else: 
                        tic = time()
                        utility_dict[(e[j],e[k])] = sacrebleu.corpus_bleu([hypos_i[j]["detok_str"]], [[hypos_i[k]["detok_str"]]]).score
                        bleu_time += time()-tic
                utility_dict[(e[k],e[j])] = utility_dict[(e[j],e[k])]
            hypos_i[j]["expected_utility"] += utility_dict[(e[j],e[k])] / sample_size
    
    #sort expected utility in descending order
    hypos_i.sort(key = lambda hypo: hypo.get("expected_utility"),reverse=True)

    return hypos_i