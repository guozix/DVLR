

# pids: 799, 681, 615
shot_examples = [
{
"question": "How much money does Ruth need to buy a baking dish, a casserole dish, and an ice cream scoop? (Unit: $)",
"caption": "The image shows a table with a variety of items on it, including a baking dish, ice cream scoop, casserole dish, and rolling pin. The text in the image says:\n\n```\nbaking dish\n$4.00\nice cream scoop\n$6.00\ncasserole dish\n$3.00\nrolling pin\n$4.00\n```",
"ocr": "[([5, 3], 'baking dish'), ([177, 5], '$4.00'), ([7, 41], 'ice cream scoop'), ([177, 37], '$6.00'), ([9, 69], 'casserole dish'), ([177, 69], '$3.00'), ([5, 98], 'rolling pin'), ([177, 101], '$4.00')]",
"solution": """
Find the total cost of a baking dish, a casserole dish, and an ice cream scoop.\n\n$4.00 + $3.00 + $6.00 = $13.00\n\nRuth needs $13.00.
""",
"code": """
baking_dish_price = 4.00
casserole_dish_price = 3.00
ice_cream_scoop_price = 6.00

ans = baking_dish_price + casserole_dish_price + ice_cream_scoop_price
print(ans)
"""
},

{
"question": "What is the largest city in the nation where this plane is headquartered?",
"choices": ['hong kong', 'osaka', 'shanghai', 'tokyo'],
"caption": "The image shows a large passenger jet parked on a tarmac at an airport. The jet is white with red trim and has a red tail. It is sitting on top of a tarmac next to a building. The jet is being loaded with passengers and cargo. The text on the image says \"Japan. Endless Discovery\".",
"solution": """
The caption mentions that the text on the image says "Japan. Endless Discovery". This indicates that the plane is headquartered in Japan. 

Among the Japanese cities, Tokyo is the largest city.

Thus, the answer is D (tokyo).
""",
"code": """
def largest_city(caption, choices):
    countries_largest_cities = {
        'Japan': 'tokyo',
        'China': 'shanghai'
    }

    if "Japan" in caption:
        country = 'Japan'
    elif "China" in caption:
        country = 'China'

    for choice in choices:
        if choice == countries_largest_cities[country]:
            return choice
    return ""

choices = ['hong kong', 'osaka', 'shanghai', 'tokyo']
caption = "The image shows a large passenger jet parked on a tarmac at an airport. The jet is white with red trim and has a red tail. It is sitting on top of a tarmac next to a building. The jet is being loaded with passengers and cargo. The text on the image says 'Japan. Endless Discovery'."

print(largest_city(caption, choices))
"""
},

{
"question": "If two sides of a triangle measure 12 and 7, which of the following cannot be the perimeter of the triangle?",
"choices": ['29', '34', '37', '38'],
"caption": "The image shows a triangle with two sides labeled 7 and 12. The triangle is drawn on a white background. There is no text other than the labels.",
"ocr": "[([70, 74], '7'), ([324, 74], '12')]",
"solution": """
To determine which of the given perimeters cannot be possible for the triangle, we apply the triangle inequality theorem. The sum of any two sides of a triangle must be greater than the third side.

For the maximum possible value of the third side:
12 + 7 = 19

The minimum possible value for the third side:
12 - 7 = 5

The third side for each option:
(A) 29 - 12 - 7 = 10 (valid)
(B) 34 - 12 - 7 = 15 (valid)
(C) 37 - 12 - 7 = 18 (valid)
(D) 38 - 12 - 7 = 19 (invalid because it should be less than 19)

Thus, the answer is D.
""",
"code": """
def is_valid_triangle(a, b, perimeter):
    # Given a and b, find the third side
    third_side = perimeter - a - b
    
    # Check triangle inequality
    if (a + b > third_side) and (a + third_side > b) and (b + third_side > a):
        return True
    return False

# Given sides
a = 12
b = 7

# Given perimeters
perimeters = [29, 34, 37, 38]

# Check which perimeter is not valid
for p in perimeters:
    if not is_valid_triangle(a, b, p):
        print(p)
""",

}
]


from few_shot_prompt import fs_examples
shot_examples = fs_examples


def refine_caption(caption):
    if isinstance(caption, str):
        nonsense = ["Sure. ", 
                    "Sure, I can do that.",
                    "Sorry, I can't help with images of people yet.",
                    "I can't process this file.",
                    "I'm unable to help you with that, as I'm only a language model and don't have the necessary information or abilities.",
                    "I'm not programmed to assist with that.",
                    "Please let me know if you have any other questions.",
                    "I hope this is helpful!",
                    "I hope this helps!"]
        for non in nonsense:
            caption = caption.replace(non, "").strip()
        caption = caption.replace("  ", " ").strip()
    else:
        caption = ""
    return caption


def refine_ocr(ocr):
    """
    [   (
        [161, 39], [766, 39], [766, 120], [161, 120]], 
        'The spring force does', 
        0.912845069753024
        ), 
    ]
    -->
    [   (
        [161, 39], 
        'The spring force does', 
        ), 
    ]
    """
    try:
        ocr = eval(ocr)
        if len(ocr) > 0:
            ocr = [([int(e[0][0][0]), int(e[0][0][1])], e[1]) for e in ocr]
            ocr = str(ocr)
        else:
            ocr = ""
    except:
        ocr = ""
    return ocr


def create_one_query(problem, examples, shot_num, shot_type, use_caption, use_ocr, query_format):


    ### [1] Demo prompt
    if shot_num == 0:
        demo_prompt = ""
    else:
        demos = []
        shot_num = min(shot_num, len(examples))
        for example in examples[:shot_num]:
            prompt = ""

            if "as_example" in example:
                prompt += example["as_example"]

            demos.append(prompt)

        demo_prompt = "\n\n".join(demos)

    ### [2] Test query
    # problem info
    # question = problem['question']
    question = problem[query_format]
    caption = problem['caption']
    ocr = problem['ocr']
    question_type = problem['question_type']

    # hint
    if shot_type == 'solution':
        if question_type == "multi_choice":
            hint_text = f"First perform reasoning, then finally select the answer from the choices in the following format: Answer: xxx."
        else:
            hint_text = f"First perform reasoning, then finally answer the question requiring a number or value in the following format: Answer: xxx."
    else:
        assert shot_type == 'code'
        hint_text = "Hint: Please generate a python code to solve the problem"

    # question
    # question_text = f"Question: {question}"
    question_text = question
    
    # elements = [question_text, choices_text, caption_text, ocr_text, hint_text, prompt]
    elements = [question_text]
    test_query = "\n".join([e for e in elements if e != ""])

    ### [3] Final query
    if demo_prompt:
        query = demo_prompt + "\n\n" + test_query
    else:
        query = test_query
    query = query.strip()
    return query


def create_query_data(data, caption_data, ocr_data, args):
    query_data = {}

    for pid, problem in data.items():
        if pid in caption_data:
            caption = caption_data[pid]
            caption = refine_caption(caption)
            problem['caption'] = caption
        else:
            problem['caption'] = ""

        if pid in ocr_data:
            ocr = ocr_data[pid]
            ocr = refine_ocr(ocr)
            problem['ocr'] = ocr
        else:
            problem['ocr'] = []

        query = create_one_query(
            problem = problem, 
            examples = shot_examples,
            shot_num = args.shot_num,
            shot_type = args.shot_type,
            use_caption = args.use_caption,
            use_ocr = args.use_ocr,
            query_format = args.query_format
            )
        query_data[pid] = query

    return query_data


if __name__ == "__main__":
    for example in shot_examples:
        print("----------------------------------------")
        print("\nQuestion:", example['question'])
        if "choices" in example:
            print("\nChoices:", example['choices'])
        print("\nCaption:", example['caption'])
        if "ocr" in example:
            print("\nOCR:", example['ocr'])
        print("\nSolution:", example['solution'])
        print("\nCode:", example['code'])
        
        # execute code
        exec(example['code'])
        

