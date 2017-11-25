import csv
import re

def duc_to_lang(duc, sentence, summary):
    '''
    Input:

        duc: the path of the important sentence extract by stage1, 
        which contains sentences and reference summaries in a row, 
        each records is one row

        sentence: the path of the sentence output,
        which contains extracted sentence from duc file

        summary: the path of the summary output,
        which contains extracted summary from duc file
        (the first one of referenced summary) 

    Output:

        sentence and summary in 2 txt file
    '''
    with open(duc, encoding='utf8') as duc_file:
        lines = duc_file.read().strip().split('\n') 
        ## save the extracted sentences
        with open(sentence, 'w+') as file:
            for line in lines:
                ## remove "</s><s> " with ""         
                line = re.sub(r"</s><s> ", "", line.split('\t')[0])
                file.write(line)
                file.write('\n')
        ## save the extracted sentences
        with open(summary, 'w+') as file:
            for line in lines:
                file.write(line.split('\t')[1])
                file.write('\n')
                
if __name__ == '__main__':
    duc_file = './data/extractive_ouput/train_output.txt'
    st_file = './data/abstractive_input/sentence.txt'
    sm_file = './data/abstractive_input/summary.txt'
    duc_to_lang(duc_file,st_file,sm_file)