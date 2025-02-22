import json
import os
from typing import Any, Dict, List, Optional

import click
import numpy as np
import torch
import tqdm
from PIL import Image

from cbc.caption import CAPTION_ENGINES_CLI
from cbc.caption.ic3.caption_by_committee import DEFAULT_CBC_PROMPT, get_prompt_for_candidates
from cbc.caption.utils import postprocess_caption
from cbc.lm import LM_ENGINES_CLI, LM_LOCAL_ENGINES
from cbc.metrics import (
    compute_and_add_base_metrics,
    compute_and_add_clip_recall,
    compute_and_add_content_recall,
    compute_and_add_mauve_score,
    compute_and_add_object_hallucinations,
    compute_and_add_self_bleu,
)
from cbc.plugins import IMAGE_PLUGINS


@click.command()
@click.argument("dataset_json_path", type=click.Path(exists=True))
@click.option(
    "--caption-engine",
    type=click.Choice(CAPTION_ENGINES_CLI.keys()),  # type: ignore
    default="ofa",
    help="The underlying captioning model to use.",
)
@click.option(
    "--lm-engine",
    type=click.Choice(LM_ENGINES_CLI.keys()),  # type: ignore
    default="gpt3_davinci3",
    help="The LM to use.",
)
@click.option(
    "--plugin",
    "-p",
    type=click.Choice(IMAGE_PLUGINS.keys()),  # type: ignore
    multiple=True,
    default=[],
    help="Plugins to use. Can be specified multiple times.",
)
@click.option("--num-candidates", type=int, default=15, help="Number of candidates to generate for each image.")
@click.option("--candidate-temperature", type=float, default=1.0, help="Temperature to use when generating candidates.")
@click.option(
    "--prompt",
    type=str,
    default=DEFAULT_CBC_PROMPT,
    help="The prompt to use when generating candidates. Will load from a file if it exists.",
)
@click.option("--output-json-path", type=str, default="output.json", help="The path to save the output to.")
@click.option("--candidate-key", type=str, default="candidates", help="The key to use for the candidates.")
@click.option("--reference-key", type=str, default="references", help="The key to use for the references.")
@click.option("--image-path-key", type=str, default="image_path", help="The key to use for the image path.")
@click.option("--image-root-dir", type=str, default=None, help="The root directory for the images.")
@click.option("--overwrite-candidates", is_flag=True, help="Whether to overwrite the candidates if they already exist.")
@click.option(
    "--overwrite-candidate-summaries",
    is_flag=True,
    help="Whether to overwrite the candidate summaries if they already exist.",
)
def evaluate_dataset(
    dataset_json_path: str,
    caption_engine: str,
    lm_engine: str,
    plugin: List[str],
    num_candidates: int,
    candidate_temperature: float,
    prompt: str,
    output_json_path: str,
    candidate_key: str,
    reference_key: str,
    image_path_key: str,
    image_root_dir: Optional[str] = None,
    overwrite_candidates: bool = False,
    overwrite_candidate_summaries: bool = False,
) -> None:
    # 1. Load the dataset (references + image paths)
    print(f"Loading dataset from {dataset_json_path}...")
    with open(dataset_json_path) as f:
        samples: List[Dict[str, Any]] = json.load(f)
        if isinstance(samples, dict):
            samples = samples["samples"]  # type: ignore

    # 1.1 Load the prompt (if not already loaded)
    if os.path.exists(prompt):
        print(f"Loading prompt from {prompt}...")
        with open(prompt) as f:
            prompt = f.read().strip()

    # 2. Compute candidate captions for each image (If not already computed)
    print(f"Loading caption engine {caption_engine}...")
    captioner = CAPTION_ENGINES_CLI[caption_engine](
        device="cuda" if torch.cuda.is_available() else "cpu",
    )  # type: ignore
    print(f"Generating candidates using {caption_engine}...")
    for sample in tqdm.tqdm(samples):
        if sample.get(candidate_key, None) is None or overwrite_candidates:
            sample[candidate_key] = captioner(
                Image.open(os.path.join(image_root_dir or ".", sample[image_path_key])).convert("RGB"),
                n_captions=num_candidates,
                temperature=candidate_temperature,
            )
        # The baseline is always the first candidate
        if sample.get("baseline", None) is None or overwrite_candidates:
            sample["baseline"] = sample[candidate_key][0]  # type: ignore

    # Save the output to a temporary file which will persist in case of a crash
    _save_json_tmp_file(output_json_path, samples)

    # 2.1 Compute the plugin features for each image (if not already computed)
    for plugin_name in plugin:
        print(f"Loading plugin {plugin_name}...")
        pl = IMAGE_PLUGINS[plugin_name]()
        print(f"Computing plugin output for {plugin_name}...")
        for sample in tqdm.tqdm(samples):
            if sample.get("plugin_outputs", None) is None:
                sample["plugin_outputs"] = {}
            if sample["plugin_outputs"].get(plugin_name, None) is None or overwrite_candidates:
                sample["plugin_outputs"][plugin_name] = pl(
                    Image.open(os.path.join(image_root_dir or ".", sample[image_path_key])).convert("RGB")
                )

    # Save the output to a temporary file which will persist in case of a crash
    _save_json_tmp_file(output_json_path, samples)

    # 3. Compute the summary captions for each image (both candidate + reference summaries, if not already computed)
    print(f"Loading LM engine {lm_engine}...")
    if lm_engine in LM_LOCAL_ENGINES:
        lm = LM_ENGINES_CLI[lm_engine](device="cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    else:
        lm = LM_ENGINES_CLI[lm_engine]()

    print(f"Generating summaries using {lm_engine}...")
    overwrite_candidate_summaries = overwrite_candidate_summaries or overwrite_candidates
    for sample in tqdm.tqdm(samples):
        if sample.get("candidate_summary") is None or overwrite_candidate_summaries:
            sample["candidate_summary_prompt"] = get_prompt_for_candidates(
                sample[candidate_key], prompt=prompt, plugin_outputs=list(sample.get("plugin_outputs", {}).values())
            )
            sample["candidate_summary"] = postprocess_caption(lm.best(prompt=sample["candidate_summary_prompt"]))
        if sample.get("reference_summary") is None or overwrite_candidate_summaries:
            sample["reference_summary_prompt"] = get_prompt_for_candidates(
                sample[reference_key], prompt=prompt, plugin_outputs=list(sample.get("plugin_outputs", {}).values())
            )
            sample["reference_summary"] = postprocess_caption(lm.best(prompt=sample["reference_summary_prompt"]))

    # Save the output to a temporary file which will persist in case of a crash
    _save_json_tmp_file(output_json_path, samples)

    # 4. Compute the metrics (Bleu, ROUGE, METEOR, CIDEr, SPICE) for each image (if not already computed)
    print("Computing base metrics...")
    samples = compute_and_add_base_metrics(samples, reference_key)

    # Save the output to a temporary file which will persist in case of a crash
    _save_json_tmp_file(output_json_path, samples)

    # 5. Compute the overall Mauve score for each set of samples (if not already computed)
    print("Computing Mauve score...")
    samples = compute_and_add_mauve_score(samples, reference_key)

    # Save the output to a temporary file which will persist in case of a crash
    _save_json_tmp_file(output_json_path, samples)

    # 6. Compute the CLIP Recall for each set of candidates (if not already computed)
    print("Computing CLIP recall...")
    samples = compute_and_add_clip_recall(samples, image_path_key, image_root_dir)

    # Save the output to a temporary file which will persist in case of a crash
    _save_json_tmp_file(output_json_path, samples)

    # 7. Compute the Content Recall for each set of candidates (if not already computed)
    print("Computing Content recall...")
    samples = compute_and_add_content_recall(samples, reference_key)

    # 8. Compute the Self-BLEU for the candidates/references (if not already computed)
    print("Computing Self-BLEU...")
    samples = compute_and_add_self_bleu(samples, candidate_key, reference_key)

    # 9. Compute the hallucination metrics for each set of candidates (if not already computed)
    print("Computing Object Hallucinations...")
    samples = compute_and_add_object_hallucinations(samples, candidate_key, reference_key)

    # Save the output to a temporary file which will persist in case of a crash
    _save_json_tmp_file(output_json_path, samples)

    # 8. Aggregate the metrics across all images
    metrics = _extract_and_aggregate_metrics(samples)

    # 8. Save the results to a JSON file
    with open(output_json_path, "w") as f:
        json.dump({"samples": samples, "metrics": metrics}, f, indent=2)

    # Remove the temporary file
    if os.path.exists(f"{output_json_path}.tmp"):
        os.remove(f"{output_json_path}.tmp")

    # 9. Print the results to the console
    print(json.dumps(metrics, indent=2))


def _save_json_tmp_file(output_json_path: str, samples: List[Dict[str, Any]]) -> None:
    with open(f"{output_json_path}.tmp", "w") as f:
        json.dump(samples, f)


def _extract_and_aggregate_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    return {
        "standard": {
            # Base Scores
            "candidate_summary_bleu_1": float(np.mean([s["scores"]["candidate_summary_bleu_1"] for s in samples])),
            "candidate_summary_bleu_2": float(np.mean([s["scores"]["candidate_summary_bleu_2"] for s in samples])),
            "candidate_summary_bleu_3": float(np.mean([s["scores"]["candidate_summary_bleu_3"] for s in samples])),
            "candidate_summary_bleu_4": float(np.mean([s["scores"]["candidate_summary_bleu_4"] for s in samples])),
            "candidate_summary_rouge": float(np.mean([s["scores"]["candidate_summary_rouge"] for s in samples])),
            "candidate_summary_cider": float(np.mean([s["scores"]["candidate_summary_cider"] for s in samples])),
            "reference_summary_bleu_1": float(np.mean([s["scores"]["reference_summary_bleu_1"] for s in samples])),
            "reference_summary_bleu_2": float(np.mean([s["scores"]["reference_summary_bleu_2"] for s in samples])),
            "reference_summary_bleu_3": float(np.mean([s["scores"]["reference_summary_bleu_3"] for s in samples])),
            "reference_summary_bleu_4": float(np.mean([s["scores"]["reference_summary_bleu_4"] for s in samples])),
            "reference_summary_rouge": float(np.mean([s["scores"]["reference_summary_rouge"] for s in samples])),
            "reference_summary_cider": float(np.mean([s["scores"]["reference_summary_cider"] for s in samples])),
            "baseline_bleu_1": float(np.mean([s["scores"]["baseline_bleu_1"] for s in samples])),
            "baseline_bleu_2": float(np.mean([s["scores"]["baseline_bleu_2"] for s in samples])),
            "baseline_bleu_3": float(np.mean([s["scores"]["baseline_bleu_3"] for s in samples])),
            "baseline_bleu_4": float(np.mean([s["scores"]["baseline_bleu_4"] for s in samples])),
            "baseline_rouge": float(np.mean([s["scores"]["baseline_rouge"] for s in samples])),
            "baseline_cider": float(np.mean([s["scores"]["baseline_cider"] for s in samples])),
            # Mauve Scores
            "candidate_summary_mauve": float(np.mean([s["scores"]["candidate_summary_mauve"] for s in samples])),
            "reference_summary_mauve": float(np.mean([s["scores"]["reference_summary_mauve"] for s in samples])),
            "baseline_mauve": float(np.mean([s["scores"]["baseline_mauve"] for s in samples])),
            # Self-BLEU
            "candidate_self_bleu": float(np.mean([s["scores"]["self_bleu"]["candidates"] for s in samples])),
            "reference_self_bleu": float(np.mean([s["scores"]["self_bleu"]["references"] for s in samples])),
        },
        # CLIP Scores
        "clip_recall": {
            "candidate_summary_clip_recall_rank": float(
                np.mean([s["scores"]["candidate_summary_clip_recall_rank"] for s in samples])
            ),
            "candidate_summary_clip_recall_mrr": float(
                np.mean([s["scores"]["candidate_summary_clip_recall_mrr"] for s in samples])
            ),
            "candidate_summary_clip_recall_at_1": float(
                np.mean([s["scores"]["candidate_summary_clip_recall_at_1"] for s in samples])
            ),
            "candidate_summary_clip_recall_at_5": float(
                np.mean([s["scores"]["candidate_summary_clip_recall_at_5"] for s in samples])
            ),
            "candidate_summary_clip_recall_at_10": float(
                np.mean([s["scores"]["candidate_summary_clip_recall_at_10"] for s in samples])
            ),
            "candidate_summary_clip_recall_max_rank": float(
                np.mean([s["scores"]["candidate_summary_clip_recall_max_rank"] for s in samples])
            ),
            "reference_summary_clip_recall_rank": float(
                np.mean([s["scores"]["reference_summary_clip_recall_rank"] for s in samples])
            ),
            "reference_summary_clip_recall_mrr": float(
                np.mean([s["scores"]["reference_summary_clip_recall_mrr"] for s in samples])
            ),
            "reference_summary_clip_recall_at_1": float(
                np.mean([s["scores"]["reference_summary_clip_recall_at_1"] for s in samples])
            ),
            "reference_summary_clip_recall_at_5": float(
                np.mean([s["scores"]["reference_summary_clip_recall_at_5"] for s in samples])
            ),
            "reference_summary_clip_recall_at_10": float(
                np.mean([s["scores"]["reference_summary_clip_recall_at_10"] for s in samples])
            ),
            "reference_summary_clip_recall_max_rank": float(
                np.mean([s["scores"]["reference_summary_clip_recall_max_rank"] for s in samples])
            ),
            "baseline_clip_recall_rank": float(np.mean([s["scores"]["baseline_clip_recall_rank"] for s in samples])),
            "baseline_clip_recall_mrr": float(np.mean([s["scores"]["baseline_clip_recall_mrr"] for s in samples])),
            "baseline_clip_recall_at_1": float(np.mean([s["scores"]["baseline_clip_recall_at_1"] for s in samples])),
            "baseline_clip_recall_at_5": float(np.mean([s["scores"]["baseline_clip_recall_at_5"] for s in samples])),
            "baseline_clip_recall_at_10": float(np.mean([s["scores"]["baseline_clip_recall_at_10"] for s in samples])),
            "baseline_clip_recall_max_rank": float(
                np.mean([s["scores"]["baseline_clip_recall_max_rank"] for s in samples])
            ),
        },
        # Content Scores
        "content_recall": {
            "candidate_summary_noun_recall": float(
                np.mean([s["scores"]["content_recall"]["candidate_summary_noun_recall"] for s in samples])
            ),
            "candidate_summary_verb_recall": float(
                np.mean([s["scores"]["content_recall"]["candidate_summary_verb_recall"] for s in samples])
            ),
            "candidate_summary_noun_fuzzy_recall": float(
                np.mean([s["scores"]["content_recall"]["candidate_summary_noun_fuzzy_recall"] for s in samples])
            ),
            "candidate_summary_verb_fuzzy_recall": float(
                np.mean([s["scores"]["content_recall"]["candidate_summary_verb_fuzzy_recall"] for s in samples])
            ),
            "reference_summary_noun_recall": float(
                np.mean([s["scores"]["content_recall"]["reference_summary_noun_recall"] for s in samples])
            ),
            "reference_summary_verb_recall": float(
                np.mean([s["scores"]["content_recall"]["reference_summary_verb_recall"] for s in samples])
            ),
            "reference_summary_noun_fuzzy_recall": float(
                np.mean([s["scores"]["content_recall"]["reference_summary_noun_fuzzy_recall"] for s in samples])
            ),
            "reference_summary_verb_fuzzy_recall": float(
                np.mean([s["scores"]["content_recall"]["reference_summary_verb_fuzzy_recall"] for s in samples])
            ),
            "baseline_noun_recall": float(
                np.mean([s["scores"]["content_recall"]["baseline_noun_recall"] for s in samples])
            ),
            "baseline_verb_recall": float(
                np.mean([s["scores"]["content_recall"]["baseline_verb_recall"] for s in samples])
            ),
            "baseline_noun_fuzzy_recall": float(
                np.mean([s["scores"]["content_recall"]["baseline_noun_fuzzy_recall"] for s in samples])
            ),
            "baseline_verb_fuzzy_recall": float(
                np.mean([s["scores"]["content_recall"]["baseline_verb_fuzzy_recall"] for s in samples])
            ),
        },
        # Hallucination Scores
        "hallucinations": {
            "hallucinated_objects_percentage": float(np.sum([s["hallucinated_object_count"] for s in samples]))
            / float(np.sum([s["object_count"] for s in samples])),
            "hallucinated_captions_percentage": float(
                np.sum([1 for s in samples if s["hallucinated_object_count"] > 0])
            )
            / float(len(samples)),
            "average_hungarian_matching_score": float(
                np.mean([s["scores"]["hungarian_matching_score"] for s in samples])
            ),
        },
    }
