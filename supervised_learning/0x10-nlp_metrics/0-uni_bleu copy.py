#!/usr/bin/env python3
"""
    NLP Evaluation Metrics
"""
import numpy as np
from collections import Counter
import math
from nltk.translate.bleu_score import corpus_bleu
from fractions import Fraction

def brevity_penalty(closest_ref_len, hyp_len):
    if hyp_len > closest_ref_len:
        return 1
    # If hypothesis is empty, brevity penalty = 0 should result in BLEU = 0.0
    elif hyp_len == 0:
        return 0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)


def closest_ref_length(references, hyp_len):
    """
    This function finds the reference that is the closest length to the
    hypothesis. The closest reference length is referred to as *r* variable
    from the brevity penalty formula in Papineni et. al. (2002)

    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hyp_len: The length of the hypothesis.
    :type hyp_len: int
    :return: The length of the reference that's closest to the hypothesis.
    :rtype: int
    """
    ref_lens = (len(reference) for reference in references)
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len)
    )
    return closest_ref_len


def ngrams(sentence, n=2, n_gram=True):
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
    #weights = [1, 0, 0, 0]
    s = (w_i * np.log(p_i) for w_i, p_i in zip(weights, clipped_precision_score))
    print(type(s))
    #for si in s:
    #    print(si)
    s = (BP) * np.exp(math.fsum(s))
    print('sss', s)
    return s






def modified_precision(references, hypothesis, n):
    """
    Calculate modified ngram precision.

    The normal precision method may lead to some wrong translations with
    high-precision, e.g., the translation, in which a word of reference
    repeats several times, has very high precision.

    This function only returns the Fraction object that contains the numerator
    and denominator necessary to calculate the corpus-level precision.
    To calculate the modified precision for a single pair of hypothesis and
    references, cast the Fraction object into a float.

    The famous "the the the ... " example shows that you can get BLEU precision
    by duplicating high frequency words.

        >>> reference1 = 'the cat is on the mat'.split()
        >>> reference2 = 'there is a cat on the mat'.split()
        >>> hypothesis1 = 'the the the the the the the'.split()
        >>> references = [reference1, reference2]
        >>> float(modified_precision(references, hypothesis1, n=1)) # doctest: +ELLIPSIS
        0.2857...

    In the modified n-gram precision, a reference word will be considered
    exhausted after a matching hypothesis word is identified, e.g.

        >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
        ...               'ensures', 'that', 'the', 'military', 'will',
        ...               'forever', 'heed', 'Party', 'commands']
        >>> reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
        ...               'guarantees', 'the', 'military', 'forces', 'always',
        ...               'being', 'under', 'the', 'command', 'of', 'the',
        ...               'Party']
        >>> reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
        ...               'army', 'always', 'to', 'heed', 'the', 'directions',
        ...               'of', 'the', 'party']
        >>> hypothesis = 'of the'.split()
        >>> references = [reference1, reference2, reference3]
        >>> float(modified_precision(references, hypothesis, n=1))
        1.0
        >>> float(modified_precision(references, hypothesis, n=2))
        1.0

    An example of a normal machine translation hypothesis:

        >>> hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
        ...               'ensures', 'that', 'the', 'military', 'always',
        ...               'obeys', 'the', 'commands', 'of', 'the', 'party']

        >>> hypothesis2 = ['It', 'is', 'to', 'insure', 'the', 'troops',
        ...               'forever', 'hearing', 'the', 'activity', 'guidebook',
        ...               'that', 'party', 'direct']

        >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
        ...               'ensures', 'that', 'the', 'military', 'will',
        ...               'forever', 'heed', 'Party', 'commands']

        >>> reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
        ...               'guarantees', 'the', 'military', 'forces', 'always',
        ...               'being', 'under', 'the', 'command', 'of', 'the',
        ...               'Party']

        >>> reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
        ...               'army', 'always', 'to', 'heed', 'the', 'directions',
        ...               'of', 'the', 'party']
        >>> references = [reference1, reference2, reference3]
        >>> float(modified_precision(references, hypothesis1, n=1)) # doctest: +ELLIPSIS
        0.9444...
        >>> float(modified_precision(references, hypothesis2, n=1)) # doctest: +ELLIPSIS
        0.5714...
        >>> float(modified_precision(references, hypothesis1, n=2)) # doctest: +ELLIPSIS
        0.5882352941176471
        >>> float(modified_precision(references, hypothesis2, n=2)) # doctest: +ELLIPSIS
        0.07692...


    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hypothesis: A hypothesis translation.
    :type hypothesis: list(str)
    :param n: The ngram order.
    :type n: int
    :return: BLEU's modified precision for the nth order ngram.
    :rtype: Fraction
    """
    # Extracts all ngrams in hypothesis
    # Set an empty Counter if hypothesis is empty.
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    # Extract a union of references' counts.
    # max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
    max_counts = {}
    for reference in references:
        reference_counts = (
            Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        )
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

    # Assigns the intersection between hypothesis and references' counts.
    clipped_counts = {
        ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()
    }

    numerator = sum(clipped_counts.values())
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))

    return Fraction(numerator, denominator, _normalize=False)




def uni_bleu(references, sentence):
    """ calculates the unigram BLEU score for a sentence:
        references is a list of reference translations
        each reference translation is a list of the words in the translation
        sentence is a list containing the model proposed sentence
        Returns: the unigram BLEU score
    """
    hypothesis = sentence
    weights=(0.25, 0.25, 0.25, 0.25)
    weights=(1, 0, 0, 0)
    weights=(0, 1, 0, 0)
    # compute modified precision for 1-4 ngrams
    p_numerators = Counter()  
    p_denominators = Counter()  
    hyp_lengths, ref_lengths = 0, 0

    for i, _ in enumerate(weights, start=1):
        p_i = modified_precision(references, hypothesis, i)
        p_numerators[i] += p_i.numerator
        p_denominators[i] += p_i.denominator

    # compute brevity penalty    
    hyp_len = len(hypothesis)
    ref_len = closest_ref_length(references, hyp_len)
    bp = brevity_penalty(ref_len, hyp_len)

    # compute final score
    p_n = [
        Fraction(p_numerators[i], p_denominators[i], 
        _normalize=False)
        for i, _ in enumerate(weights, start=1)
        if p_numerators[i] > 0
    ]
    s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, p_n))
    s = bp * math.exp(math.fsum(s))

    return s





    l = len(references)
    print('l', l)
    score = 0
    for i in range(l):
        score += bleu_score(references[i], sentence)
        print('score', score)
    return score / 10#/ 4#bleu_score([references], [sentence]) #, weights=1)



original = "It is a guide to action which ensures that the military alwasy obeys the command of the party"
machine_translated = "It is the guiding principle which guarantees the military forces alwasy being under the command of the party"

#print (bleu_score(original, machine_translated))
#print (sentence_bleu([original.split()], machine_translated.split()))