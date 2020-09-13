from os import listdir
import math
import string
import array
from keras.utils import to_categorical
from random import randint

def load_doc(filename):
    file = open(filename, encoding="utf-8")
    text = file.read()
    file.close()
    return text


def split_story(doc):
    index = doc.find("@highlight")
    story, highlights = doc[:index], doc[index:].split("@highlight")

    comment_index = doc.find("Share what you think")
    story = story[:(comment_index-1)]

    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights


def load_stories(directory):
    stories = list()
    limit = 500
    i = 0
    for name in listdir(directory):
        i += 1
        if i > limit:
            break
        filename = directory + "/" + name
        doc = load_doc(filename)
        story, highlights = split_story(doc)
        stories.append({"story": story, "highlights": highlights})
    return stories


def clean_lines(lines):
    cleaned = list()
    table = str.maketrans("", "", string.punctuation)
    for line in lines:
        line = line.split()
        line = [word.lower() for word in line]
        cleaned.append(" ".join(line))
    cleaned = [c for c in cleaned if len(c) > 0 or ]
    return cleaned


def get_preprocessed_stories():
    directory = "../stories/"
    stories = load_stories(directory)
    print("Loaded Stories {}".format(len(stories)))

    for example in stories:
        example["story"] = clean_lines(example["story"].split("\n"))
        example["highlights"] = clean_lines(example["highlights"])
    return stories


def generate_sequence(length, n_unique):
    return [randint(1, n_unique - 1) for _ in range(length)]


def get_datasets():
    stories = get_preprocessed_stories()
    stories_len = len(stories)

    divider = math.floor(stories_len * 0.7)
    train_data = get_dataset(stories[:divider])
    print("Train input size", len(train_data[0]))
    val_data = get_dataset(stories[divider:])
    print("Test input size", len(val_data[0]))

    return train_data, val_data


def get_dataset(data):
    input_texts = []
    target_texts = []

    input_sequences = set()
    target_sequences = set()

    for story in data:
        target_text = []
        for highlight in story["highlights"]:
            target_text.append(highlight + "\n")
        for seq in story["highlights"]:
            if seq not in target_sequences:
                target_sequences.add(seq)

        input_text = story["story"]
        for seq in input_text:
            if seq not in input_sequences:
                input_sequences.add(seq)

        input_texts.append(input_text)
        target_texts.append(target_text)

    input_sequences = sorted(list(input_sequences))
    target_sequences = sorted(list(target_sequences))
    return (input_texts, target_texts, input_sequences, target_sequences)
