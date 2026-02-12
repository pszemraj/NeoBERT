"""GLUE/SuperGLUE preprocessing utilities."""

from typing import Any

glue_task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "snli": ("premise", "hypothesis"),
    "allnli": ("premise", "hypothesis"),
}

super_glue_task_to_keys = {
    "axb": ("sentence1", "sentence2"),
    "axg": ("premise", "hypothesis"),
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "copa": ("premise", "choice1", "choice2", "question"),
    "rte": ("premise", "hypothesis"),
    "multirc": (
        "passage.text",
        "passage.questions.question",
        "passage.questions.answers.text",
    ),
    "record": (
        "passage.text",
        "passage.text.entities",
        "qas.query",
        "qas.answers.text",
    ),
    "wic": ("word", "sentence1", "sentence2", "start1", "start2", "end1", "end2"),
    "wsc": (
        "text",
        "target.span1_index",
        "target.span1_text",
        "target.span2_index",
        "target.span2_text",
    ),
}


def get_labels(examples: dict[str, Any], meta_task: str, task: str) -> list[int]:
    """Extract labels for GLUE/SuperGLUE tasks.

    :param dict[str, Any] examples: Batch of dataset examples.
    :param str meta_task: Meta task identifier (glue/superglue).
    :param str task: Specific task name.
    :return list[int]: List of labels for the batch.
    """
    if meta_task == "glue" or task in (
        "axb",
        "axg",
        "boolq",
        "cb",
        "rte",
        "wic",
        "wsc",
        "snli",
        "allnli",
    ):
        return examples["label"]

    elif task == "copa":
        labels = [
            int(x)
            for i in range(len(examples["label"]))
            for x in [examples["label"][i] == 0, examples["label"][i] == 1]
        ]

    elif task == "multirc":
        labels = [
            answer["label"]
            for passage in examples["passage"]
            for question in passage["questions"]
            for answer in question["answers"]
        ]

    elif task == "record":
        labels = [
            int(
                passage["text"][entity["start"] : entity["end"] + 1]
                in [answer["text"] for answer in qas["answers"]]
            )
            for passage, qas in zip(examples["passage"], examples["qas"])
            for entity in passage["entities"]
        ]

    return labels


def get_sentences(
    examples: dict[str, Any],
    meta_task: str,
    task: str,
    bos_token: str = "<s>",
    eos_token: str = "</s>",
    sep_token: str = "<s>",
) -> tuple[list[str], list[str] | None]:
    """Build sentence pairs for GLUE/SuperGLUE tasks.

    :param dict[str, Any] examples: Batch of dataset examples.
    :param str meta_task: Meta task identifier (glue/superglue).
    :param str task: Specific task name.
    :param str bos_token: BOS token to inject in some tasks.
    :param str eos_token: EOS token to inject in some tasks.
    :param str sep_token: Separator token to inject in some tasks.
    :return tuple[list[str], list[str] | None]: Sentence pairs or single sentences.
    """
    if meta_task == "glue":
        key1, key2 = glue_task_to_keys[task]
        return (
            (examples[key1], None) if key2 is None else (examples[key1], examples[key2])
        )

    output = {"sentence1": [], "sentence2": []}

    if task in {"axb", "axg", "boolq", "cb", "rte"}:
        key1, key2 = super_glue_task_to_keys[task]
        output["sentence1"] = examples[key1]
        output["sentence2"] = examples[key2]

    elif task == "wic":
        output["sentence1"] = [
            f"{word} {eos_token} {bos_token} {sentence}"
            for sentence, word in zip(examples["sentence1"], examples["word"])
        ]
        output["sentence2"] = examples["sentence2"]

    elif task == "copa":
        for premise, question, choice1, choice2 in zip(
            examples["premise"],
            examples["question"],
            examples["choice1"],
            examples["choice2"],
        ):
            marker = "because" if question == "cause" else "so"
            premise_marker = f"{premise} {marker}"
            output["sentence1"].extend(
                [f"{premise_marker} {choice1}", f"{premise_marker} {choice2}"]
            )

    elif task == "multirc":
        for passage in examples["passage"]:
            text = passage["text"]
            for question in passage["questions"]:
                query = question["question"]
                sentence1 = f"{text} {eos_token} {bos_token} {query}"
                output["sentence1"].extend([sentence1] * len(question["answers"]))
                output["sentence2"].extend(
                    [answer["text"] for answer in question["answers"]]
                )

    elif task == "record":
        for passage, qas in zip(examples["passage"], examples["qas"]):
            text = passage["text"]
            entities = passage["entities"]
            query = qas[0]["query"]
            placeholder_index = query.index("@placeholder")
            output["sentence1"].extend([text] * len(entities))
            output["sentence2"].extend(
                [
                    query[:placeholder_index]
                    + text[e["start"] : e["end"] + 1]
                    + query[placeholder_index + 12 :]
                    for e in entities
                ]
            )

    elif task == "wsc":
        output["sentence1"] = examples["text"]
        output["sentence2"].extend(
            [
                f"{target['span2_text']} refers to {target['span1_text']}."
                for target in examples["target"]
            ]
        )

    else:
        raise ValueError(f"Task {task} is not supported")

    return (output["sentence1"], output["sentence2"] or None)


def process_function(
    examples: dict[str, Any], cfg: Any, tokenizer: Any
) -> dict[str, Any]:
    """Tokenize a batch for GLUE/SuperGLUE tasks.

    :param dict[str, Any] examples: Batch of dataset examples.
    :param Any cfg: Task configuration with tokenizer settings.
    :param Any tokenizer: Tokenizer with GLUE-compatible special tokens.
    :return dict[str, Any]: Tokenized batch with labels when applicable.
    """
    glue_task = str(
        getattr(cfg.glue, "task_name", getattr(cfg, "task", "glue"))
    ).strip()
    sentences = get_sentences(
        examples=examples,
        meta_task=cfg.meta_task,
        task=glue_task,
        bos_token=tokenizer.bos_token,
        eos_token=tokenizer.eos_token,
        sep_token=tokenizer.sep_token,
    )
    result = tokenizer(
        *sentences,
        padding=False,
        max_length=int(cfg.tokenizer.max_length),
        truncation=bool(getattr(cfg.tokenizer, "truncation", True)),
    )
    if cfg.mode in ["train", "eval"]:
        result["labels"] = get_labels(
            examples=examples,
            meta_task=cfg.meta_task,
            task=glue_task,
        )
    return result
