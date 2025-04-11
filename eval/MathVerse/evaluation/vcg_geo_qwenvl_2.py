import os
import io
import time
import argparse
import sys
import types

from tqdm import tqdm
import torch
import sys
sys.path.append('../')
from utilities import *

from build_query import create_query_data
from transformers import AutoTokenizer

# import torch_npu
# from torch_npu.contrib import transfer_to_npu
# torch_npu.npu.set_device("npu:0")
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

import base64
from io import BytesIO

from vllm_qwenvl import init_qwenvl_2_5, chat_qwenvl_2_5


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio


def dynamic_preprocess2(image, min_num=1, max_num=12, prior_aspect_ratio=None, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    new_target_ratios = []
    for i in target_ratios:
        if prior_aspect_ratio[0]%i[0] or prior_aspect_ratio[1]%i[1]:
            new_target_ratios.append(i)
        else:
            continue
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, new_target_ratios, orig_width, orig_height, image_size)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, min_num=1, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images, target_aspect_ratio = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, min_num=2, max_num=4)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values, target_aspect_ratio

def load_image2(image_file, input_size=448, min_num=1, max_num=12, target_aspect_ratio=None):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess2(image, image_size=input_size, use_thumbnail=True, min_num=1, max_num=2, prior_aspect_ratio=target_aspect_ratio)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def verify_response(response):
    if isinstance(response, str):
        response = response.strip() 
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        return False
    return True

def evaluate_code(code_string):
    # execute_code_and_capture_output
    # Backup the original stdout
    old_stdout = sys.stdout
    
    # Redirect stdout to capture the output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    # Try executing the code and capture any exception
    error = None
    try:
        exec(code_string)
    except Exception as e:
        error = e
    
    # Restore the original stdout
    sys.stdout = old_stdout
    
    # Get the captured output
    captured_output = new_stdout.getvalue()
    if isinstance(captured_output, str):
        captured_output = captured_output.strip()
    
    # Return the captured output or error
    return captured_output, error
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    # output
    parser.add_argument('--output_dir', type=str, default='../results/minimonkey_chat_baseline_hmx4221')
    parser.add_argument('--output_file', type=str, default='output_bard.json')
    # model
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='llm engine',
                        choices = ['gpt-3.5-turbo', 'claude-2', 'gpt4', 'gpt-4-0613', 'bard'])
    parser.add_argument('--key', type=str, default='', help='key for llm api')
    # query
    parser.add_argument('--query_file', type=str, default=None)  
    parser.add_argument('--caption_file', type=str, default='../data/texts/captions_bard.json') 
    parser.add_argument('--ocr_file', type=str, default='../data/texts/ocrs_easyocr.json')   
    parser.add_argument('--shot_type', type=str, default='solution', help='shot type', 
                        choices = ['solution', 'code'])
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    parser.add_argument('--use_caption', action='store_true', help='use caption data')
    parser.add_argument('--use_ocr', action='store_true', help='use ocr data')
    # other settings
    parser.add_argument('--rerun', action='store_true', help='rerun answer extraction for all problems')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument("--checkpoint", type=str, default="")

    # for MathVerse
    parser.add_argument('--query_format', type=str, default='query_cot')   # query_wo
    parser.add_argument('--use_familiar_hint', action='store_true', help='?')

    # for new test setting
    parser.add_argument('--use_sample', action='store_true', help='use_sample or greedy')
    parser.add_argument('--test_group', nargs='+', help='eval groups')

    args = parser.parse_args()

    # load data
    input_file = os.path.join(args.data_dir, args.input_file)
    print(f"Reading {input_file}...")
    data = read_json(input_file)
    if isinstance(data, list):
        data = {item["sample_index"]: item for item in data}
    if isinstance(data, dict): pass
    
    # load or create query data
    if args.query_file:
        query_file = os.path.join(args.data_dir, args.query_file)
        if os.path.exists(query_file):
            print(f"Loading existing {query_file}...")
            query_data = read_json(query_file)
    else:
        print("\nCreating new query...")
        # load caption
        caption_data = {}
        if args.use_caption:
            caption_file = args.caption_file
            if os.path.exists(caption_file):
                print(f"Reading {caption_file}...")
                try:
                    caption_data = read_json(caption_file)["texts"]
                    print("Caption data loaded.")
                except:
                    print("Caption data not found!! Please Check.")                    
        # load ocr
        ocr_data = {}
        if args.use_ocr:
            ocr_file = args.ocr_file
            if os.path.exists(ocr_file):
                print(f"Reading {ocr_file}...")
                try:
                    ocr_data = read_json(ocr_file)["texts"]
                    print("OCR data loaded.")
                except:
                    print("OCR data not found!! Please Check.")
        # create query
        query_data = create_query_data(data, caption_data, ocr_data, args)

    # output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file)
    
    # load results
    if os.path.exists(output_file):
        print("\nResults already exist.")
        print(f"Reading {output_file}...")
        results = read_json(output_file)
    else:
        results = {}

    path = args.checkpoint
    init_qwenvl_2_5(path, multi_image={"image": 2})

    if args.use_sample:
        generation_config = dict(do_sample=True, max_new_tokens=512, temperature=0.5, top_k=50)

    # build final test pid list
    test_pids = list(data.keys())
    print("\nNumber of test problems in total:", len(test_pids))

    skip_pids = []
    if not args.rerun:
        print("\nRemoving problems with existing valid response...")
        for pid in test_pids:
            # print(f"Checking {pid}...")
            if pid in results and 'response' in results[pid]:
                response = results[pid]['response']
                if verify_response(response):
                    # print(f"Valid response found for {pid}.")
                    skip_pids.append(pid)
    else:
        print("\nRerun answer extraction for all problems...")

    test_pids = [pid for pid in test_pids if pid not in skip_pids]
    print("Number of test problems to run:", len(test_pids))
    # print(test_pids)

    if args.test_group:
        print('** test groups:', args.test_group)
    else:
        print('** test all versions')


    def make_msg_wimage(contents):
        msg = []
        for i in range(len(contents)):
            if isinstance(contents[i], str):
                msg.append({"type": "text", "text": contents[i]})
            elif isinstance(contents[i], Image.Image):
                buffered = BytesIO()
                contents[i].save(buffered, format="PNG")
                base64_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                msg.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_str}"}}
                )
            else:
                raise NotImplementedError()
        return msg

    query_temp_VD_zs = """I will provide you with a geometry math problem with an image and the question. You don't need to answer the question, but you need to summarize all the known conditions and the question to be answered (include multiple-choice options if available) based on the provided information and the image.

Problem image:
<image>
Question:
{}
Summarization:"""


    query_temp_VO_zs = """I will provide you with a geometry math problem with an image and the question. You don't need to answer the question, but you need to summarize all the known conditions and the question to be answered (include multiple-choice options if available) based on the provided information and the image.

Problem image:
<image>
Question: Please answer the question shown in the image.
Summarization:"""


    # tqdm, enumerate results
    for _, pid in enumerate(tqdm(test_pids)):
        problem = data[pid]
        query = query_data[pid]

        if args.test_group:
            if problem['problem_version'] not in args.test_group:
                continue
        
        q = problem["question"]
        image = problem['image']
        image_path = os.path.join(args.data_dir, image)

        if problem['problem_version'] == "Vision Only":
            assert q == ""

            msgs = query_temp_VO_zs.split("<image>")
            query = [
                ["text", msgs[0]],
                ["image", image_path],
                ["text", msgs[1]]
            ]
        else:
            assert q != ""
            
            msgs = query_temp_VD_zs.split("<image>")
            query = [
                ["text", msgs[0]],
                ["image", image_path],
                ["text", msgs[1].format(q)],
            ]
        
        text_image_input = []
        for item in query:
            if item[0] == "image":
                text_image_input.append(Image.open(item[1]).convert('RGB'))
            elif item[0] == "text":
                text_image_input.append(item[1])
            else:
                raise NotImplementedError("Unsupported input type for qwenvl_2_5")

        text_image_input = make_msg_wimage(text_image_input)

        print(query)
        print(f"\nGenerating vcg for {pid}...")
        response = chat_qwenvl_2_5(text_image_input)
        response = response[0].outputs[0].text

        results[pid] = problem
        results[pid]['question_summary_query'] = query
        results[pid]['question_summary'] = response

        if args.debug:
            print(f"\n#Query: \n{query}")
            print(f"\n#Response: \n{response}")
    
        try:
            print(f"Saving results to {output_file}...")
            save_json(results, output_file)
            print(f"Results saved.")
        except Exception as e:
            print(e)
            print(f"Error in saving {output_file}")
