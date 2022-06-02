from typing import List
import sacrebleu
import torch


def min_bayes_risk(hypos, sample_size, utility="BLEU"):
    """
    This function changes the score of the input hypos to the expected utility function, 
    so the outside ranking function can arrange it accordingly. See

        Sampling-Based Minimum Bayes Risk Decoding for Neural Machine Translation
        https://arxiv.org/pdf/2108.04718.pdf
    
    Args:
        hypos (list(size batch_size) of list (size beam_size) of Dict with attributes
            "tokens": torch.Tensor with len sentence_length and dtype int (token id)
            "score": torch.Tensor of size 1 float
            "attention": empty torch.Tensor
            "alignment":
            "positional_scores": torch.Tensor() len sentence_length dtype float)
        
        sample_size (int): the size of sample/cfg.beam_size
        utility (str, optional): choices of 
            "BLEU"
            "BEER"
            "METEOR"
            "ChrF"
    
    return:
        hypos with score as the expected utility
    """
    #print (type(hypos))
    #print ("hypos.size()",hypos.size())
    #print ("hypos[0]", hypos[0])

    for i in range(len(hypos)):
        e_utility = [0.0 for _ in range(sample_size)]
        for j in range(sample_size):
            for k in range(sample_size):
                e_utility[j] += (sacrebleu.corpus_bleu(hypos[i][j]["tokens"], hypos[i][k]["tokens"], tokenize = "none") 
                * torch.exp(hypos[i][k]["score"]))
            
        for j in range(sample_size):
            hypos[i][j]["score"] = e_utility[j]
    
    return hypos