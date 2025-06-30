from neobert.contrastive.datasets import (ALLNLI, AMAZONQA, CONCURRENTQA,
                                          FEVER, GITHUBISSUE, GOOAQ, MSMARCO,
                                          PAQ, PUBMEDQA, QQP, SENTENCECOMP,
                                          STACKEXCHANGE, STACKOVERFLOW, STS12,
                                          STSBENCHMARK, TRIVIAQA, WIKIHOW)

DATASETS = {
    "ALLNLI": ALLNLI,
    "AMAZONQA": AMAZONQA,
    "CONCURRENTQA": CONCURRENTQA,
    "FEVER": FEVER,
    "GITHUBISSUE": GITHUBISSUE,
    "GOOAQ": GOOAQ,
    "MSMARCO": MSMARCO,
    "PAQ": PAQ,
    "PUBMEDQA": PUBMEDQA,
    "QQP": QQP,
    "SENTENCECOMP": SENTENCECOMP,
    "STACKEXCHANGE": STACKEXCHANGE,
    "STACKOVERFLOW": STACKOVERFLOW,
    "STS12": STS12,
    "STSBENCHMARK": STSBENCHMARK,
    "TRIVIAQA": TRIVIAQA,
    "WIKIHOW": WIKIHOW,
}

for name in DATASETS.keys():
    subdataset = DATASETS[name]().dataset
