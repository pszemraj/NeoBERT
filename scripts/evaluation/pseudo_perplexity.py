"""Compute pseudo-perplexity scores for masked language models."""

from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Tuple

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelWithLMHead,
    AutoTokenizer,
)
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings

from neobert.checkpointing import (
    MODEL_WEIGHTS_NAME,
    load_deepspeed_fp32_state_dict,
    load_model_safetensors,
)
from neobert.model import NeoBERTConfig, NeoBERTLMHead


def get_data(dataset: Any) -> Any:
    """Filter and subsample long-text examples.

    :param Any dataset: Dataset to filter.
    :return Any: Filtered dataset.
    """
    dataset = dataset.filter(lambda x: len(x["text"]) > 20000)
    # Select 100 samples from each length window (20000-30000, 30000-40000, etc.)

    def filter_by_length(length: int, example: dict[str, Any]) -> bool:
        """Filter examples by text length window.

        :param int length: Minimum length of the window.
        :param dict[str, Any] example: Dataset example.
        :return bool: True if example falls in the window.
        """
        text_length = len(example["text"])
        return length <= text_length < length + 10000

    datasets = []
    for length in range(20000, 100000, 10000):
        tmp = dataset.filter(partial(filter_by_length, length))
        tmp = tmp.select(range(min(200, len(tmp))))
        datasets.append(tmp)

    final = concatenate_datasets(datasets)

    return final


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials.

    :param int dim: Dimension of the frequency tensor.
    :param int end: End index for precomputing frequencies.
    :param float theta: Scaling factor for frequency computation.
    :return torch.Tensor: Precomputed frequency tensor.
    """

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def _resolve_neobert_checkpoint_dir(
    checkpoint_path: str | Path,
    checkpoint: str,
) -> Path:
    """Resolve a NeoBERT checkpoint directory for portable weight loading.

    ``checkpoint_path`` may point either at a checkpoint root containing
    ``<tag>/`` subdirectories or at a single step directory already.

    :param str | Path checkpoint_path: User-provided checkpoint path.
    :param str checkpoint: Requested checkpoint tag/step.
    :return Path: Resolved candidate checkpoint directory.
    :raises FileNotFoundError:
        If an explicit checkpoint tag is missing beneath a direct checkpoint path
        that already contains portable weights.
    """
    checkpoint_root = Path(checkpoint_path)
    requested_tag = str(checkpoint).strip()
    candidate = checkpoint_root / requested_tag
    if candidate.is_dir():
        return candidate
    if _checkpoint_path_matches_tag(checkpoint_root, requested_tag):
        return checkpoint_root
    if (checkpoint_root / MODEL_WEIGHTS_NAME).is_file():
        raise FileNotFoundError(
            f"Requested checkpoint '{requested_tag}' was not found under "
            f"{checkpoint_root}. Refusing to silently load portable weights from "
            f"the root path instead."
        )
    return checkpoint_root


def _checkpoint_path_matches_tag(checkpoint_path: Path, checkpoint: str) -> bool:
    """Return whether ``checkpoint_path`` already points at ``checkpoint``.

    This accepts both direct step directories (``.../<tag>``) and nested
    Accelerate DeepSpeed layouts (``.../<tag>/pytorch_model``).

    :param Path checkpoint_path: Candidate direct checkpoint path.
    :param str checkpoint: Requested checkpoint tag/step.
    :return bool: ``True`` when the path already targets the requested tag.
    """
    requested_tag = str(checkpoint).strip()
    return bool(requested_tag) and (
        checkpoint_path.name == requested_tag
        or checkpoint_path.parent.name == requested_tag
    )


def _load_neobert_checkpoint_weights(
    model: NeoBERTLMHead,
    *,
    checkpoint_path: str | Path,
    checkpoint: str,
) -> NeoBERTLMHead:
    """Load NeoBERT MLM weights from portable or legacy checkpoint formats.

    Portable ``model.safetensors`` payloads are preferred when present.
    Legacy DeepSpeed ZeRO checkpoints remain supported through the optional
    ``neobert[legacy-checkpoints]`` dependency.

    :param NeoBERTLMHead model: Model instance to populate.
    :param str | Path checkpoint_path: Checkpoint root or step directory.
    :param str checkpoint: Requested checkpoint tag/step.
    :return NeoBERTLMHead: Model with loaded weights.
    """
    checkpoint_root = Path(checkpoint_path)
    checkpoint_dir = _resolve_neobert_checkpoint_dir(checkpoint_root, checkpoint)
    weights_path = checkpoint_dir / MODEL_WEIGHTS_NAME

    if weights_path.is_file():
        state_dict = load_model_safetensors(checkpoint_dir, map_location="cpu")
    else:
        requested_tag = str(checkpoint).strip()
        try:
            state_dict = load_deepspeed_fp32_state_dict(
                checkpoint_root,
                tag=requested_tag,
            )
        except (FileNotFoundError, ValueError):
            checkpoint_root = checkpoint_root.resolve()
            # Only fall back to direct-path resolution when the caller already
            # pointed at the requested step/tag directory. Otherwise re-raise so
            # an explicit missing checkpoint cannot silently load ``latest``.
            if _checkpoint_path_matches_tag(checkpoint_root, requested_tag):
                state_dict = load_deepspeed_fp32_state_dict(checkpoint_root)
            else:
                raise
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    # Model
    # parser.add_argument("--source", type=str, help="")
    parser.add_argument("--model_name", type=str, help="")
    parser.add_argument("--from_hub", action="store_true", help="")
    parser.add_argument("--config_path", type=str, help="", required=False)
    parser.add_argument("--checkpoint_path", type=str, help="", required=False)
    parser.add_argument(
        "--checkpoint", type=str, default="final", help="", required=False
    )
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument(
        "--compile", action="store_true", help="Enable torch.compile for inference"
    )
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--max_length", type=int, default=4096, help="")
    # Dataset
    parser.add_argument("--data_name", type=str, help="")
    parser.add_argument("--dataset_shard", type=int, help="")
    parser.add_argument("--data_path", type=str, help="")
    parser.add_argument("--n_sentences", type=int, default=10000, help="")
    # Log
    parser.add_argument("--output_path", type=str, help="")
    args = parser.parse_args()

    # Get model and tokenizer
    if args.from_hub:
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        config.max_position_embeddings = max(
            args.max_length, config.max_position_embeddings
        )
        if "roberta" in args.model_name.lower():
            model = AutoModelWithLMHead.from_pretrained(
                args.model_name, trust_remote_code=True
            )
            model.roberta.embeddings = RobertaEmbeddings(config)
        elif "modern" in args.model_name.lower():
            model = AutoModelForMaskedLM.from_pretrained(
                args.model_name, trust_remote_code=True
            )
        else:
            model = AutoModelWithLMHead.from_pretrained(
                args.model_name, trust_remote_code=True
            )
        if hasattr(model.config, "max_position_embeddings"):
            model.config.max_position_embeddings = max(
                args.max_length, model.config.max_position_embeddings
            )
        if hasattr(model.config, "max_length"):
            model.config.max_length = max(args.max_length, model.config.max_length)
    if "neobert" in args.model_name:
        # Import our new config system
        from neobert.config import ConfigLoader

        if not args.config_path:
            raise ValueError("NeoBERT evaluation requires --config_path.")
        if not args.checkpoint_path:
            raise ValueError("NeoBERT evaluation requires --checkpoint_path.")

        model_pretraining_config = ConfigLoader.load(args.config_path)
        model_pretraining_config.model.max_position_embeddings = args.max_length
        model = NeoBERTLMHead(
            NeoBERTConfig(
                hidden_size=model_pretraining_config.model.hidden_size,
                num_hidden_layers=model_pretraining_config.model.num_hidden_layers,
                num_attention_heads=model_pretraining_config.model.num_attention_heads,
                intermediate_size=model_pretraining_config.model.intermediate_size,
                dropout=model_pretraining_config.model.dropout_prob,
                vocab_size=model_pretraining_config.model.vocab_size,
                max_position_embeddings=model_pretraining_config.model.max_position_embeddings,
                attn_backend=model_pretraining_config.model.attn_backend,
                kernel_backend=model_pretraining_config.model.kernel_backend,
                ngpt=model_pretraining_config.model.ngpt,
                hidden_act=model_pretraining_config.model.hidden_act,
                rope=model_pretraining_config.model.rope,
                rms_norm=model_pretraining_config.model.rms_norm,
                norm_eps=model_pretraining_config.model.norm_eps,
                pad_token_id=model_pretraining_config.model.pad_token_id,
            )
        )
        model = _load_neobert_checkpoint_weights(
            model,
            checkpoint_path=args.checkpoint_path,
            checkpoint=args.checkpoint,
        )
        model.model.freqs_cis = precompute_freqs_cis(
            model.config.hidden_size // model.config.num_attention_heads,
            model_pretraining_config.model.max_position_embeddings,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.model_max_length = max(args.max_length, tokenizer.model_max_length)
    model.to(args.device)
    if args.compile:
        if hasattr(torch, "compile"):
            model = torch.compile(model)
        else:
            print("torch.compile is not available; continuing without compilation.")

    # Prepare the dataset
    dataset = load_dataset("wikipedia", "20220301.en")["train"]
    dataset = dataset.remove_columns(["url", "title"])
    dataset = dataset.filter(
        lambda x: len(x["text"]) < 20000 and len(x["text"]) > 500
    ).select(range(args.n_sentences))
    dataset = get_data(dataset)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.shard(8, args.dataset_shard)

    # Generator that, for each sentence, tokenize, mask each position, and batch
    def batch_tokenize_mask(
        dataset: Any, tokenizer: Any, batch_size: int
    ) -> Iterator[Tuple[Any, torch.Tensor, torch.Tensor]]:
        """Yield masked token batches for pseudo-perplexity.

        :param Any dataset: Dataset with text/id columns.
        :param Any tokenizer: Tokenizer instance.
        :param int batch_size: Batch size for masked tokens.
        :return Iterator[tuple[Any, torch.Tensor, torch.Tensor]]: Batches of ids/x/y.
        """
        for x, id in zip(dataset["text"], dataset["id"]):
            tokenized_input = tokenizer(
                x,
                padding=False,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            x = tokenized_input["input_ids"]
            seq_length = x.shape[1]
            x = x.repeat(seq_length, 1)
            y = torch.where(torch.eye(seq_length, dtype=torch.bool), x, -100)
            x = torch.where(
                torch.eye(seq_length, dtype=torch.bool), tokenizer.mask_token_id, x
            )
            for _x, _y in zip(
                torch.split(x, batch_size, 0), torch.split(y, batch_size, 0)
            ):
                yield (id, _x, _y)

    dataloader = batch_tokenize_mask(dataset, tokenizer, args.batch_size)
    pbar = tqdm(total=len(dataset))

    # Save the pseudo-perplexities into a csv
    output_path = (
        Path(args.output_path) / args.model_name.replace("/", "_") / args.checkpoint
    )
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{args.data_name}_ppl_{str(args.dataset_shard)}.csv"
    if not output_file.exists():
        with output_file.open("a", encoding="utf-8") as file:
            file.write("name,pseudo-perplexity,per-residue-cross-entropy...\n")
    else:
        with output_file.open("r", encoding="utf-8") as file:
            line_count = len(file.readlines()) - 1
        num_batches_to_skip = line_count // args.batch_size
        print(f"Skipping first {num_batches_to_skip} batches...")
        for _ in range(num_batches_to_skip):
            _ = next(iter(dataloader))

    # Compute pseudo-perplexity
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=args.device, dtype=torch.bfloat16, enabled=args.bf16
        ),
    ):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        losses = dict()
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        for id, x, y in dataloader:
            x = x.to(args.device)
            y = y.to(args.device)
            output = model(x)
            logits = output["logits"]
            loss = loss_fn(logits.transpose(1, 2), y).sum(-1).tolist()
            if id in losses:
                losses[id] = losses[id] + loss
            else:
                pbar.update(1)

                with output_file.open("a", encoding="utf-8") as file:
                    for k, v in losses.items():
                        file.write(
                            f"{k},{np.exp(np.mean(v))},{','.join(map(str, v))}\n"
                        )

                losses = dict()
                losses[id] = loss
