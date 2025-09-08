import datasets
import numpy as np

def generate_prompt_pqa(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(sample):
        prompt = f"""
You will be given a multiple-choice question related to a biological protocol. The blank in the question (represented as '____') indicates where the correct choice should be filled in.

Question:
{sample['question']}

Choices:
{'\n'.join(sample['choices'])}

Your task:
- Choose the most likely correct answer from the given choices.
- You must always select *one* answer, even if you are unsure.
- The selected answer must match one of the choices exactly (including case and punctuation).
        """
        return {'prompt':prompt, 'target':sample['answer'], 'choices':sample['choices']}
    return dataset.map(_process_doc)

def generate_prompt_ord(dataset: datasets.Dataset) -> datasets.Dataset:

    def _process_doc(sample):
        answer = np.random.choice(["A","B"])

        if answer == "A":
            steps_1 = '\n'.join(['- ' + step for step in sample['correct_steps']])
            steps_2 = '\n'.join(['- ' + step for step in sample['wrong_steps']])
        else:
            steps_1 = '\n'.join(['- ' + step for step in sample['wrong_steps']])
            steps_2 = '\n'.join(['- ' + step for step in sample['correct_steps']])
        prompt = f"""
You will be given two versions of a series of steps in a biological protocol. 
One version has the steps in the incorrect order, while the other version has the steps in the right order. Tell me which order of steps is correct by outputting "A" or "B", no other words.
The steps could be: 
A:
{steps_1}
or B:
{steps_2}
        """
        return {'prompt': prompt, 'choices': ["A", "B"], 'answer': answer}
    
    return dataset.map(_process_doc)


def generate_prompt_err(dataset: datasets.Dataset) -> datasets.Dataset:

    correct_ds = dataset.filter(lambda x: x['is_correct'])
    incorrect_ds = dataset.filter(lambda x: not x['is_correct']) 

    # Prepare correct examples: keep corrected_text and is_correct=True
    def _to_correct(sample):
        sample['text'] = sample.get('corrected_text', '')
        sample['is_correct'] = True
        return sample
    correct_prepped = correct_ds.map(_to_correct)

    # For each incorrect example, produce two examples:
    #  - a "corrected" copy with corrected_text and is_correct=True
    #  - a "corrupted" copy with corrupted_text and is_correct=False
    incorrect_to_correct = incorrect_ds.map(
        lambda s: {**s, 'text': s.get('corrected_text', ''), 'is_correct': True}
    )
    incorrect_to_corrupted = incorrect_ds.map(
        lambda s: {**s, 'text': s.get('corrupted_text', ''), 'is_correct': False}
    )

    # Combine correct examples and duplicated incorrect examples
    combined = datasets.concatenate_datasets([
        correct_prepped,
        incorrect_to_correct,
        incorrect_to_corrupted
    ])

    def _format_prompt(sample):
        step = sample['text']
        prompt = f"""Determine whether the following target step in a protocol is True or False:
{step}

You may use the following context, which includes the purpose of the step, as well as the preceding and following steps, to inform your decision:
{sample['context']}

Please carefully evaluate if the step is logically consistent, necessary, and accurate in the context. If you find anything wrong, answer False.

- Please respond with only True or False, without any additional explanation.
            """
        return {'prompt': prompt, 'is_correct': sample['is_correct'], 'choices': ["True", "False"]}
    
    return combined.map(_format_prompt).shuffle(seed=42)
