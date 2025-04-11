import os
import re
import time
import argparse

from tqdm import tqdm

import sys
sys.path.append('../')
from utilities import *


def verify_extraction(extraction):
    if extraction == "" or extraction == None:
        return False
    return True


def extract_answer(response, problem, quick_extract=True):
    if response == "":
        return ""
    
    print("Quickly extracting answer...")
    try:
    ## Extended version for multiple 'Answer:'
        tmp_text = response
        
        while re.search(r'Answer\s*(.*)', tmp_text):
            result = re.search(r'Answer\s*(.*)', tmp_text)
            idx = result.span()
            tmp_text = tmp_text[idx[0] + 6 : ]

        print("")
        if tmp_text:
            extraction = tmp_text
        
            matches = re.findall(r'<CONCLUSION>(.*?)</CONCLUSION>', extraction)
            extraction = matches[-1].strip() if matches else extraction
            print(extraction)

            while "**" in extraction:
                extraction = extraction.replace("**", "").strip()
            print(extraction)

            if extraction[0] == ":":
                extraction = extraction[1:].strip()
            print(extraction)
            
            pattern = r"\\boxed\{([A-E])\}"
            matches = re.findall(pattern, extraction)
            extraction = matches[-1].upper().strip() if matches else extraction
            print(extraction)

            matches = re.findall(r'(?:^|\s)([A-E])(?=[\W_]|$)', extraction)
            extraction = matches[-1].upper().strip() if matches else extraction
            print(extraction)

            return extraction
        else:
            return response
    except:
        return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--output_dir', type=str, default='../')
    parser.add_argument('--output_file', type=str, default='results/test_temp/output_bard.json')
    parser.add_argument('--response_label', type=str, default='response', help='response label for the input file')
    # model
    parser.add_argument('--llm_engine', type=str, default='gpt-4-0613', help='llm engine',
                        choices = ['gpt-3.5-turbo', 'gpt-3.5', 'gpt-4', 'gpt-4-0314', 'gpt-4-0613'])
    parser.add_argument('--number', type=int, default=-1, help='number of problems to run')
    parser.add_argument('--quick_extract', action='store_true', help='use rules to extract answer for some problems')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer extraction')
    # output
    parser.add_argument('--save_every', type=int, default=10, help='save every n problems')
    parser.add_argument('--output_label', type=str, default='', help='label for the output file')
    args = parser.parse_args()

    # args
    label = args.response_label
    result_file = os.path.join(args.output_dir, args.output_file)

    if args.output_label != '':
        output_file = result_file.replace('.json', f'_{args.output_label}.json')
    else:
        output_file = result_file

    # read results
    print(f"Reading {result_file}...")
    results = read_json(result_file)

    # full pids
    full_pids = list(results.keys())
    if args.number > 0:
        full_pids = full_pids[:min(args.number, len(full_pids))]
    print("Number of testing problems:", len(full_pids))

    # test pids
    if args.rerun:
        test_pids = full_pids
    else:
        test_pids = []
        for pid in full_pids:
            if 'extraction' not in results[pid] or not verify_extraction(results[pid]['extraction']):
                test_pids.append(pid)
    
    test_num = len(test_pids)
    print("Number of problems to run:", test_num)

    # tqdm, enumerate results
    for i, pid in enumerate(tqdm(test_pids)):
        problem = results[pid]

        # only evaluate 'Plane Geometry'
        # if (problem['metadata']['subject'] == 'Plane Geometry') and (problem["question_type"] == "multi-choice"):
        if problem["question_type"] == "multi-choice":

            assert label in problem
            response = problem[label]       

            
            extraction  = extract_answer(response, problem, args.quick_extract).strip()
            results[pid]['extraction'] = extraction

            if i % args.save_every == 0 or i == test_num - 1:
                print(f"Saving results to {output_file}...")
                save_json(results, output_file)
                print(f"Results saved.")
        else:
            del results[pid]
    
    print("Number of problems run:", test_num)
