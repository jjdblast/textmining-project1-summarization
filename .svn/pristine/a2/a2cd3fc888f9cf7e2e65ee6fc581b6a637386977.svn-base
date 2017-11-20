from os.path import normpath, basename, join, exists, expanduser
import gzip
import json
from glob import glob
from termcolor import colored, cprint
from pythonrouge.pythonrouge import Pythonrouge

ROUGE = expanduser("/root/sharefolder/playgrounds/pku/text-mining/project1-summrazation/rouge/pythonrouge/pythonrouge/RELEASE-1.5.5")
ROUGE_PATH = join(ROUGE, "ROUGE-1.5.5.pl")
ROUGE_DATA = join(ROUGE, "data")

"""
Computes ROUGE scores for models and datasets that have outputs available
"""

def remove_prefix_and_suffix(text, prefix, suffix):
    if text.startswith(prefix):
        text = text[len(prefix):]
    if text.endswith(suffix):
        text = text[:-len(suffix)]
    return text

def remove_tags(sentence):
    return remove_prefix_and_suffix(sentence, "<SOS>", "<EOS>").strip()

def evaluate_rouge_scores(evaluation_file):
    summaries = [] # model-generated
    references = [] # human-generated
    # articles = {}
    with open(evaluation_file, encoding='utf8') as file:
        evaluation_lines = file.read().strip().split('\n')
        print("%d entries..." % len(evaluation_lines))
        for line in evaluation_lines:
            sum_line = line.split('\t')[0]
            ref_line = line.split('\t')[1:]
            summaries.append( remove_tags(sum_line).encode('utf-8').split())
            references.append([ remove_tags(example).encode('utf-8').split() for example in ref_line])
    print("%d entries are used for evaluation." % len(summaries))
    
    rouge = Pythonrouge(summary_file_exist=False,
                    summary=summaries, reference=references,
                    n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                    recall_only=True, stemming=True, stopwords=True,
                    word_level=True, length_limit=True, length=50,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()
    return score

if __name__ == '__main__':
    evaluation_file = './data/abstractive_output/test_result.txt'
    print(evaluate_rouge_scores(evaluation_file))
