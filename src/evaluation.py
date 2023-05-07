import collections  #implements specialized container datatypes 
import logging
import regex
import string  # contains some constants, utility function, and classes for string manipulation.
import unicodedata  # provides access to the Unicode Character Database
from functools import partial  # for higher-order functions: functions that act on or return other functions
# partial: set default constants for a part of some function's total arguments
from multiprocessing import Pool as ProcessPool  # allows to fully leverage multiple processors on a given machine
# pool: the Pool object offers a convenient means of parallelizing the execution of a function
# across multiple input values, distributing the input data across processes (data parallelism)
from typing import Tuple, List, Dict
import numpy as np

"""
Evaluation code from DPR: https://github.com/facebookresearch/DPR
"""

class SimpleTokenizer(object):
    # Receives text, outputs list of tokens (all instances of contiguous alphanum OR non-whitespace char sequences.)
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'
    
    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        
    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]  # iter through text, append to list all match instances
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens
    
logger = logging.getLogger(__name__)

QAMatchStats = collections.namedtuple('QAMatchStats', ['top_k_hits', 'questions_doc_hits'])
# for readability + allows accessing to fields by name instead of a position index.
# returns a new tuple subclass named 'QAMatchStats'

def has_answer(answers, text, tokenizer) -> bool:
    """ Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)
    
    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    # goes over all contiguous sequences of words of length len(answer) in the document. If one of the passages exactly
    # matches any one of the answers, True is returned.
    return False
    # if nothing is found, False is returned.

def check_answer(example, tokenizer) -> List[bool]:
    """ Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']
    
    hits = []
    
    for i, doc in enumerate(ctxs):
        text = doc['text']
        
        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            hits.append(False)
            continue
            
        hits.append(has_answer(answers, text, tokenizer))
        
    return hits

def calculate_matches(data: List, workers_num: int):
    """
    Evaluates answer presence in the given set of documents. This function is supposed to be used with a large collection of
    documents and results. It internally forks multiple sub-processes for evaluation and then merges results.
    :param all_docs: dictionary of the entire documents database. doc_id = (doc_text, title)
    :param answers: list of answers list. One list per question.
    :param closest_docs: document ids of the top results along with their scores
    :param workers_num: amount of parallel threads to process data
    :param match_type: type of answer matching. Refer to has_answer code for available options
    :return: matching information typle.
    top_k_hits - a list where the index is the amount of top documents retrieved and the value is the total amount of
    valid matches across an entire dataset.
    questions_doc_hits - more detailed info with answer matches for every question and every retrieved document
    """
    
    logger.info('Matching answers in top docs...')
    
    tokenizer = SimpleTokenizer()
    get_score_partial = partial(check_answer, tokenizer=tokenizer)
    
    processes = ProcessPool(processes=workers_num)
    scores = processes.map(get_score_partial, data)  # delegate evaluation to workers_num processes
    
    logger.info('Per question validation results len=%d', len(scores))
    
    n_docs = len(data[0]['ctxs'])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
        
    return QAMatchStats(top_k_hits, scores)
    
#################################################
########        READER EVALUATION        ########
#################################################

def _normalize(text):
    return unicodedata.normalize('NFD', text)

# Normalization and score functions from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_score(prediction, ground_truths):
    return max([f1(prediction, gt) for gt in ground_truths])

def exact_match_score(prediction, ground_truths):
    return max([em(prediction, gt) for gt in ground_truths])

####################################################
########        RETRIEVER EVALUATION        ########
####################################################

def eval_batch(scores, inversions, avg_topk, idx_topk):
    for k, s in enumerate(scores):
        s = s.cpu().numpy()
        sorted_idx = np.argsort(-s)
        score(sorted_idx, inversion, avg_topk, idx_topk)
        
def count_inversions(arr):
    inv_count = 0
    lenarr = len(arr)
    for i in range(lenarr):
        for j in range(i+1, lenarr):
            if (arr[i] > arr[j]):
                inv_count += 1
    return inv_count

def score(x, inversions, avg_topk, idx_topk):
    x = np.array(x)
    inversions.append(count_inversions(x))
    for k in avg_topk:
        # ratio of passages in the predicted top-k that are
        # also in the topk given by gold score
        avg_pred_topk = (x[:k]<k).mean()
        avg_topk[k].append(avg_pred_topk)
    for k in idx_topk:
        below_k = (x<k)
        # number of passages required to obtain all passages from gold top-k
        idx_gold_topk = len(x) - np.argmax(below_k[::-1])
        idx_topk[k].append(idx_gold_topk)
