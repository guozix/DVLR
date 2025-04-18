# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference with
multi-image input on vision language models for text generation,
using the chat template defined by the model.
"""
from argparse import Namespace
from typing import List, NamedTuple, Optional

from PIL.Image import Image
from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser

QUESTION = "What is the content of each image?"
IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
]


class ModelRequestData(NamedTuple):
    llm: LLM
    prompt: str
    stop_token_ids: Optional[List[int]]
    image_data: List[Image]
    chat_template: Optional[str]


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.

def load_qwen_vl_chat(question: str,
                      image_urls: List[str]) -> ModelRequestData:
    model_name = "Qwen/Qwen-VL-Chat"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=1024,
        max_num_seqs=2,
        hf_overrides={"architectures": ["QwenVLForConditionalGeneration"]},
        limit_mm_per_prompt={"image": len(image_urls)},
    )
    placeholders = "".join(f"Picture {i}: <img></img>\n"
                           for i, _ in enumerate(image_urls, start=1))

    # This model does not have a chat_template attribute on its tokenizer,
    # so we need to explicitly pass it. We use ChatML since it's used in the
    # generation utils of the model:
    # https://huggingface.co/Qwen/Qwen-VL-Chat/blob/main/qwen_generation_utils.py#L265
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)

    # Copied from: https://huggingface.co/docs/transformers/main/en/chat_templating
    chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"  # noqa: E501

    messages = [{'role': 'user', 'content': f"{placeholders}\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True,
                                           chat_template=chat_template)

    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=[fetch_image(url) for url in image_urls],
        chat_template=chat_template,
    )


def load_qwen2_vl(question, image_urls: List[str]) -> ModelRequestData:
    try:
        from qwen_vl_utils import process_vision_info
    except ModuleNotFoundError:
        print('WARNING: `qwen-vl-utils` not installed, input images will not '
              'be automatically resized. You can enable this functionality by '
              '`pip install qwen-vl-utils`.')
        process_vision_info = None

    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    # Tested on L40
    llm = LLM(
        model=model_name,
        max_model_len=32768 if process_vision_info is None else 4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role":
        "user",
        "content": [
            *placeholders,
            {
                "type": "text",
                "text": question
            },
        ],
    }]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    stop_token_ids = None

    if process_vision_info is None:
        image_data = [fetch_image(url) for url in image_urls]
    else:
        image_data, _ = process_vision_info(messages)

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=image_data,
        chat_template=None,
    )


def load_qwen2_5_vl(question, image_urls: List[str]) -> ModelRequestData:
    try:
        from qwen_vl_utils import process_vision_info
    except ModuleNotFoundError:
        print('WARNING: `qwen-vl-utils` not installed, input images will not '
              'be automatically resized. You can enable this functionality by '
              '`pip install qwen-vl-utils`.')
        process_vision_info = None

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    llm = LLM(
        model=model_name,
        max_model_len=32768 if process_vision_info is None else 4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role":
        "user",
        "content": [
            *placeholders,
            {
                "type": "text",
                "text": question
            },
        ],
    }]

    processor = AutoProcessor.from_pretrained(model_name)

    prompt = processor.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    stop_token_ids = None

    if process_vision_info is None:
        image_data = [fetch_image(url) for url in image_urls]
    else:
        image_data, _ = process_vision_info(messages,
                                            return_video_sample_fps=False)

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
        image_data=image_data,
        chat_template=None,
    )


model_example_map = {
    "qwen_vl_chat": load_qwen_vl_chat,
    "qwen2_vl": load_qwen2_vl,
    "qwen2_5_vl": load_qwen2_5_vl,
}


def run_generate(model, question: str, image_urls: List[str]):
    req_data = model_example_map[model](question, image_urls)

    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=128,
                                     stop_token_ids=req_data.stop_token_ids)

    outputs = req_data.llm.generate(
        {
            "prompt": req_data.prompt,
            "multi_modal_data": {
                "image": req_data.image_data
            },
        },
        sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def run_chat(model: str, question: str, image_urls: List[str]):
    req_data = model_example_map[model](question, image_urls)

    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=128,
                                     stop_token_ids=req_data.stop_token_ids)
    outputs = req_data.llm.chat(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": question,
                },
                *({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    },
                } for image_url in image_urls),
            ],
        }],
        sampling_params=sampling_params,
        chat_template=req_data.chat_template,
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def main(args: Namespace):
    model = args.model_type
    method = args.method

    if method == "generate":
        run_generate(model, QUESTION, IMAGE_URLS)
    elif method == "chat":
        run_chat(model, QUESTION, IMAGE_URLS)
    else:
        raise ValueError(f"Invalid method: {method}")

global_model = None
global_processer = None

try:
    from qwen_vl_utils import process_vision_info
except ModuleNotFoundError:
    print('WARNING: `qwen-vl-utils` not installed, input images will not '
            'be automatically resized. You can enable this functionality by '
            '`pip install qwen-vl-utils`.')
    import sys
    sys.exit()

def init_qwenvl_2_5(model_name, multi_image=None):
    global global_model
    if "72" in model_name:
        global_model = LLM(
            model=model_name,
            max_model_len=8000,
            max_num_seqs=5,
            # Note - mm_processor_kwargs can also be passed to generate/chat calls
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            tensor_parallel_size=4,
            limit_mm_per_prompt=multi_image
            # disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
            # limit_mm_per_prompt={"image": len(image_urls)},
        )
    else:
        global_model = LLM(
            model=model_name,
            max_model_len=8000,
            max_num_seqs=5,
            # Note - mm_processor_kwargs can also be passed to generate/chat calls
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            limit_mm_per_prompt=multi_image
            # disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
            # limit_mm_per_prompt={"image": len(image_urls)},
        )

    global global_processer
    global_processer = AutoProcessor.from_pretrained(model_name)


def inference_qwenvl_2_5(texts_images):
    global global_model
    global global_processer
    
    inputs = []
    for item in texts_images:
        text = item['text']
        image = item['image']

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }, {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }]

        prompt = global_processer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)

        image_data, _ = process_vision_info(messages)

        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_data
            },
        },)
    
    stop_token_ids = None
    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=1200,
                                     stop_token_ids=stop_token_ids)

    outputs = global_model.generate(
        inputs,
        sampling_params=sampling_params)

    return outputs


def chat_qwenvl_2_5(content_list):
    global global_model

    messages = [
        {
            "role": "user",
            "content": content_list,
        }]

    stop_token_ids = None
    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=1200,
                                     stop_token_ids=stop_token_ids)

    outputs = global_model.chat(
        messages,
        sampling_params=sampling_params,
    )

    return outputs


def init_qwen_2_5(model_name):
    global global_model
    if "72" in model_name:
        global_model = LLM(
            model=model_name,
            max_model_len=4096,
            max_num_seqs=5,
            tensor_parallel_size=4
        )
    else:
        global_model = LLM(
            model=model_name,
            max_model_len=4096,
            max_num_seqs=5
        )

    global global_processer
    global_processer = AutoProcessor.from_pretrained(model_name)


def chat_qwen_2_5(content_list):
    global global_model

    messages = [
        {
            "role": "user",
            "content": content_list,
        }
    ]

    stop_token_ids = None
    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=1200,
                                     stop_token_ids=stop_token_ids)

    outputs = global_model.chat(
        messages,
        sampling_params=sampling_params,
    )

    return outputs



if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models that support multi-image input for text '
        'generation')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="phi3_v",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument("--method",
                        type=str,
                        default="generate",
                        choices=["generate", "chat"],
                        help="The method to run in `vllm.LLM`.")

    args = parser.parse_args()
    main(args)