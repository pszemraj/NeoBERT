import argparse
import json
import os

import mteb

### GLOBAL VARIABLES ###

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
]


TASK_LIST_SUMMARIZATION = [
    "SummEval",
]

TASK_LIST_EN = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
    + TASK_LIST_SUMMARIZATION
)


TASK_LIST_NAMES = [
    ("Class.", TASK_LIST_CLASSIFICATION, ["en", "en-en"]),
    ("Clust.", TASK_LIST_CLUSTERING, ["en", "en-en"]),
    ("PairClass.", TASK_LIST_PAIR_CLASSIFICATION, ["en", "en-en"]),
    ("Rerank.", TASK_LIST_RERANKING, ["en", "en-en"]),
    ("Retr.", TASK_LIST_RETRIEVAL, ["en", "en-en"]),
    ("STS", TASK_LIST_STS, ["en", "en-en"]),
    ("Summ.", TASK_LIST_SUMMARIZATION, ["en", "en-en"]),
    # ("BitextMining", TASK_LIST_BITEXT, []),
    ("Avg.", TASK_LIST_EN, ["en", "en-en"]),
]


def compute_table():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", dest="result_folder", type=str)
    parser.add_argument("--model_name", dest="model_name", type=str)
    args = parser.parse_args()

    all_results = {}

    result_file = os.path.join(args.result_folder, f"{args.model_name}_avg_table.json")

    if os.path.exists(result_file):
        UserWarning("Overwriting existing result file.")
        os.remove(result_file)

    def explore(path):
        paths = []
        file_level = False
        files = os.listdir(path)
        if len(files) == 0:
            UserWarning(f"Empty folder path: {path}.")
        for file in files:
            if os.path.isdir(os.path.join(path, file)):
                paths.extend(explore(os.path.join(path, file)))
            else:
                file_level = True
        if file_level:
            paths.append(path)
        return paths

    for checkpoint in os.listdir(args.result_folder):
        path = os.path.join(args.result_folder, checkpoint)
        paths = explore(path)

        for path in paths:
            i = path.find(checkpoint) + len(checkpoint)
            j = path.find("no_model_name_available")
            model_name = f"{args.model_name}_{checkpoint}_{path[i + 1 : j - 1] if j != -1 else path[i + 1 :]}"

            all_results.setdefault(model_name, {})

            for file_name in os.listdir(path):
                if not file_name.endswith(".json"):
                    print(f"Skipping non-json {file_name}")
                    continue
                else:
                    with open(
                        os.path.join(path, file_name), "r", encoding="utf-8"
                    ) as f:
                        results = json.load(f)
                        all_results[model_name] = {
                            **all_results[model_name],
                            **{file_name.replace(".json", ""): results},
                        }

    avg_results = {}

    for model in all_results.keys():
        avg_results[model] = {}
        results = []
        for task_type, task_list, _ in TASK_LIST_NAMES:
            model_task_results = []
            for task in task_list:
                mteb_task = mteb.get_tasks(
                    tasks=[
                        task.replace("CQADupstackRetrieval", "CQADupstackTexRetrieval")
                    ]
                )
                assert len(mteb_task) == 1, (
                    f"Found {len(mteb_task)} for {task}. Expected 1."
                )
                test_result = all_results.get(model, {}).get(task, {})
                try:
                    model_task_results.append(
                        test_result["scores"]["test"][0].get("main_score")
                    )
                except:
                    continue

            if len(model_task_results) > 0:
                avg_results[model][task_type] = round(
                    100 * (sum(model_task_results) / len(model_task_results)), 2
                )
            else:
                avg_results[model][task_type] = 0

    with open(result_file, "w") as f:
        json.dump(avg_results, f)


if __name__ == "__main__":
    compute_table()
