fs_examples = [
    {
        "sample_index": "9",
        "problem_index": "2",
        "problem_version": "Vision Dominant",
        "question": "As shown in the figure, AB parallel CD, EG bisects angle BEF, then angle 2 is equal to ()\nChoices:\nA:50\u00b0\nB:60\u00b0\nC:65\u00b0\nD:90\u00b0",
        "image": "images_version_5/image_2.png",
        "answer": "C",
        "question_type": "multi-choice",
        "metadata": {
            "split": "testmini",
            "source": "GeoQA",
            "subject": "Plane Geometry",
            "subfield": "Angle"
        },
        "query_wo": "Please directly answer the question and provide the correct option letter, e.g., A, B, C, D.\nQuestion: As shown in the figure, AB parallel CD, EG bisects angle BEF, then angle 2 is equal to ()\nChoices:\nA:50\u00b0\nB:60\u00b0\nC:65\u00b0\nD:90\u00b0",
        "query_cot": "Please first conduct reasoning, and then answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: As shown in the figure, AB parallel CD, EG bisects angle BEF, then angle 2 is equal to ()\nChoices:\nA:50\u00b0\nB:60\u00b0\nC:65\u00b0\nD:90\u00b0",
        "question_for_eval": "As shown in the figure, AB parallel CD, EG bisects angle BEF, then angle 2 is equal to ()\nChoices:\nA:50\u00b0\nB:60\u00b0\nC:65\u00b0\nD:90\u00b0",
        "as_example": "First perform reasoning, then finally select the correct option letter from the choices, e.g., A, B, C, D, in the following format: Answer: xxx.\nQuestion: As shown in the figure, AB parallel CD, EG bisects angle BEF, then angle 2 is equal to ()\nChoices:\nA:50\u00b0\nB:60\u00b0\nC:65\u00b0\nD:90\u00b0\nFigure analysis: AB parallel CD, straight line EF intersects AB at point E, intersects CD at point F, EG bisects angle BEF, and it intersects CD at point G, angle 1 = 50.0. Question analysis: Since AB is parallel to CD, we have the following relationships: angle 1 + angle BEF = 180\u00b0, angle 1 = 50\u00b0, angle BEF = 130\u00b0. Also, since EG bisects angle BEF, we have angle BEG = 1/2 angle BEF = 65\u00b0. Therefore, angle 2 = angle BEG = 65\u00b0. Thus, option C is the correct answer.\nAnswer:C"
    },
    {
        "sample_index": "14",
        "problem_index": "3",
        "problem_version": "Vision Dominant",
        "question": "As shown in the figure, BD bisects angle ABC, CD parallel AB, then the degree of angle CDB is ()\nChoices:\nA:55\u00b0\nB:50\u00b0\nC:45\u00b0\nD:30\u00b0",
        "image": "images_version_5/image_3.png",
        "answer": "A",
        "question_type": "multi-choice",
        "metadata": {
            "split": "testmini",
            "source": "GeoQA",
            "subject": "Plane Geometry",
            "subfield": "Angle"
        },
        "query_wo": "Please directly answer the question and provide the correct option letter, e.g., A, B, C, D.\nQuestion: As shown in the figure, BD bisects angle ABC, CD parallel AB, then the degree of angle CDB is ()\nChoices:\nA:55\u00b0\nB:50\u00b0\nC:45\u00b0\nD:30\u00b0",
        "query_cot": "Please first conduct reasoning, and then answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: As shown in the figure, BD bisects angle ABC, CD parallel AB, then the degree of angle CDB is ()\nChoices:\nA:55\u00b0\nB:50\u00b0\nC:45\u00b0\nD:30\u00b0",
        "question_for_eval": "As shown in the figure, BD bisects angle ABC, CD parallel AB, then the degree of angle CDB is ()\nChoices:\nA:55\u00b0\nB:50\u00b0\nC:45\u00b0\nD:30\u00b0",
        "as_example": "First perform reasoning, then finally select the correct option letter from the choices, e.g., A, B, C, D, in the following format: Answer: xxx.\nQuestion: As shown in the figure, BD bisects angle ABC, CD parallel AB, then the degree of angle CDB is ()\nChoices:\nA:55\u00b0\nB:50\u00b0\nC:45\u00b0\nD:30\u00b0\nFigure analysis: BC and BA intersect at point B, BD bisects angle ABC, CD parallel AB, if angle BCD = 70.0. Question analysis: Because CD is parallel to AB, we have angle CDB = angle ABD. Since BD bisects angle ABC, we have angle ABD = angle DBC. Therefore, angle CDB = angle DBC. Since angle CDB + angle DBC + angle DBC = 180\u00b0, we have angle CDB = 55\u00b0. Therefore, the answer is A.\nAnswer:A"
    },
]