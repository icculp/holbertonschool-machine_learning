#!/usr/bin/env python3
"""
    NLP Evaluation Metrics
"""
import numpy as np
from collections import Counter
import math
from nltk.translate.bleu_score import corpus_bleu



def n_gram_generator(sentence, n=4, n_gram=False):
    '''
    N-Gram generator with parameters sentence
    n is for number of n_grams
    The n_gram parameter removes repeating n_grams 
    '''
    # sentence = sentence.lower() # converting to lower case
    print('sent1', sentence)
    sent_arr = np.array(sentence) # .split()) # split to string arrays
    print('sent2', sent_arr)
    length = len(sent_arr)

    word_list = []
    for i in range(length+1):
        if i < n:
            continue
        word_range = list(range(i-n,i))
        s_list = sent_arr[word_range]
        string = ' '.join(s_list) # converting list to strings
        word_list.append(string) # append to word_list
        if n_gram:
            word_list = list(set(word_list))
    return word_list

def bleu_score(original, machine_translated):
    '''
    Bleu score function given a orginal and a machine translated sentences
    '''
    mt_length = len(machine_translated)
    o_length = len(original)

    # Brevity Penalty 
    if mt_length>o_length:
        BP=1
    else:
        penality=1-(mt_length/o_length)
        BP=np.exp(penality)

    # Clipped precision
    clipped_precision_score = []
    for i in range(1, 5):
        original_n_gram = Counter(n_gram_generator(original, i))
        machine_n_gram = Counter(n_gram_generator(machine_translated, i))

        c = sum(machine_n_gram.values())
        for j in machine_n_gram:
            if j in original_n_gram:
                if machine_n_gram[j] > original_n_gram[j]:
                    machine_n_gram[j] = original_n_gram[j]
            else:
                machine_n_gram[j] = 0

        #print (sum(machine_n_gram.values()), c)
        clipped_precision_score.append(sum(machine_n_gram.values()) / c)
    print(clipped_precision_score)
    print('LENGTH CLIPPED', len(clipped_precision_score))
    #print (clipped_precision_score)

    weights = [.25,.25,.25,.25]
    #weights = [1, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16]
    #weights = [.5, .5, 0, 0]
    s = (w_i * np.log(p_i) for w_i, p_i in zip(weights, clipped_precision_score))
    print(type(s))
    #for si in s:
    #    print(si)
    s = ( BP) * np.exp(math.fsum(s))
    print('sss', s)
    return s


def uni_bleu(references, sentence):
    """ calculates the unigram BLEU score for a sentence:
        references is a list of reference translations
        each reference translation is a list of the words in the translation
        sentence is a list containing the model proposed sentence
        Returns: the unigram BLEU score
    """
    l = len(references)
    print('l', l)
    score = 0
    for i in range(l):
        score += bleu_score(references[i], sentence)
        print('score', score)
    return score / 2#/ 4#bleu_score([references], [sentence]) #, weights=1)



original = "It is a guide to action which ensures that the military alwasy obeys the command of the party"
machine_translated = "It is the guiding principle which guarantees the military forces alwasy being under the command of the party"

#print (bleu_score(original, machine_translated))
#print (sentence_bleu([original.split()], machine_translated.split()))