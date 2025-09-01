import datasets
from random import shuffle

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):

        choices = doc["distractors"] + [doc["ideal"]]
        out_doc = {
            "protocol": doc["protocol"],
            "question":doc["question"],
            "choices": choices,
            "gold": int(len(doc["distractors"])),  # always the last one
        }
        return out_doc

    return dataset.map(_process_doc)