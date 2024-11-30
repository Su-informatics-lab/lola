"""
This script estimates the probability of various medical conditions and patient
characteristics based on medication data using LLMs.

Supported Assessments:
- Binary: Type II diabetes, AUDIT-C, insurance status
- Ordinal (5 levels): Fatigue and anxiety

Usage:
    python estimate_prob_given_drug.py --model_name MODEL_NAME \
    --assessment ASSESSMENT_TYPE [options]

Required Arguments:
    --model_name     Huggingface model name (e.g., meta-llama/Llama-3.1-70B-Instruct)
    --assessment     Assessment type (diabetes|audit_c|fatigue|anxiety|insurance)

Optional Arguments:
    --cot           Enable chain-of-thought reasoning
    --num_gpus      Number of GPUs for tensor parallelism (default: 1)
    --temperature   Sampling temperature (default: 0.6)
    --batch_size    Batch size for estimation (default: 4)
    --input_file    Input parquet file with drug names
        (default: resources/drug_15980.parquet)

Output:
    Generates a parquet file containing:
    - Drug names
    - Estimated probabilities
    - Full LLM responses
    - Probabilities for each severity level (only for ordinal assessments)
"""

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import numpy as np
import torch
from torch import manual_seed
from tqdm import tqdm
from vllm import LLM, SamplingParams

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
MAX_MODEL_LENGTH = 4096


class QueryType(Enum):
    BINARY = "binary"
    ORDINAL = "ordinal"


@dataclass
class AssessmentConfig:
    name: str
    query_type: QueryType
    system_prompt: str
    question: str
    levels: Optional[List[str]] = None

    def create_prompt(self, drug: str, level: Optional[str] = None) -> str:
        if self.query_type == QueryType.BINARY:
            return (
                f"You are estimating the probability that a patient has {self.name} "
                f"given they are taking {drug}. Consider:\n"
                f"1. The primary uses and effects of this medication\n"
                f"2. Known associations with {self.name}\n"
                f"3. Typical patient demographics for this medication\n\n"
                f"Output a single probability between 0 and 1."
            )
        else:
            question_context = f"{self.question}\n\n" if self.question else ""
            return (
                f"{question_context}You are estimating the probability that a patient "
                f"taking {drug} would report '{level}'. Consider:\n"
                f"1. The primary uses and effects of this medication\n"
                f"2. Known side effects and impacts on patient well-being\n"
                f"3. Typical patient demographics for this medication\n\n"
                f"Output a single probability between 0 and 1."
            )


ASSESSMENT_CONFIGS = {
    "diabetes": AssessmentConfig(
        name="Type II diabetes",
        query_type=QueryType.BINARY,
        question="",
        system_prompt=(
            "You are a medical expert helping estimate probabilities based on medication data. "
            "You must always provide a numerical probability estimate between 0 and 1, even if uncertain. "
            "If you are unsure, provide your best estimate based on available knowledge. "
            "Never refuse to provide an estimate. "
            "Always output your final answer on a new line as 'Estimated Probability: X'\n\n"
            "You are estimating the probability that a patient has Type II diabetes based on "
            "their specific medicine."
        ),
    ),
    "audit_c": AssessmentConfig(
        name="high-risk AUDIT-C score",
        query_type=QueryType.BINARY,
        question="""The AUDIT-C score is based on three questions:

1. How often do you have a drink containing alcohol?
   a. Never (0 points)
   b. Monthly or less (1 point)
   c. 2-4 times a month (2 points)
   d. 2-3 times a week (3 points)
   e. 4 or more times a week (4 points)

2. How many standard drinks containing alcohol do you have on a typical day?
   a. 1 or 2 (0 points)
   b. 3 or 4 (1 point)
   c. 5 or 6 (2 points)
   d. 7 to 9 (3 points)
   e. 10 or more (4 points)

3. How often do you have six or more drinks on one occasion?
   a. Never (0 points)
   b. Less than monthly (1 point)
   c. Monthly (2 points)
   d. Weekly (3 points)
   e. Daily or almost daily (4 points)

Total score ranges from 0-12. For men, a score of 4+ indicates high-risk drinking.
For women, a score of 3+ indicates high-risk drinking.""",
        system_prompt=(
            "You are a medical expert helping estimate probabilities based on medication data. "
            "You must always provide a numerical probability estimate between 0 and 1, even if uncertain. "
            "If you are unsure, provide your best estimate based on available knowledge. "
            "Never refuse to provide an estimate. "
            "Always output your final answer on a new line as 'Estimated Probability: X'\n\n"
            "You are estimating the probability that a patient has a high-risk AUDIT-C score "
            "(4+ for men, 3+ for women) based on their medication."
        ),
    ),
    "fatigue": AssessmentConfig(
        name="fatigue level",
        query_type=QueryType.ORDINAL,
        question="In the past 7 days, how would you rate your fatigue?",
        levels=[
            "Average Fatigue 7 Days: None",
            "Average Fatigue 7 Days: Mild",
            "Average Fatigue 7 Days: Moderate",
            "Average Fatigue 7 Days: Severe",
            "Average Fatigue 7 Days: Very Severe",
        ],
        system_prompt=(
            "You are a medical expert helping estimate probabilities based on medication data. "
            "You must always provide a numerical probability estimate between 0 and 1, even if uncertain. "
            "If you are unsure, provide your best estimate based on available knowledge. "
            "Never refuse to provide an estimate. "
            "Always output your final answer on a new line as 'Estimated Probability: X'\n\n"
            "You are estimating the probability that a patient reports a specific level of "
            "fatigue in the past 7 days based on their medication."
        ),
    ),
    "anxiety": AssessmentConfig(
        name="emotional problems",
        query_type=QueryType.ORDINAL,
        question="In the past 7 days, how often have you been bothered by emotional problems such as feeling anxious, depressed or irritable?",
        levels=[
            "Emotional Problem 7 Days: Never",
            "Emotional Problem 7 Days: Rarely",
            "Emotional Problem 7 Days: Sometimes",
            "Emotional Problem 7 Days: Often",
            "Emotional Problem 7 Days: Always",
        ],
        system_prompt=(
            "You are a medical expert helping estimate probabilities based on medication data. "
            "You must always provide a numerical probability estimate between 0 and 1, even if uncertain. "
            "If you are unsure, provide your best estimate based on available knowledge. "
            "Never refuse to provide an estimate. "
            "Always output your final answer on a new line as 'Estimated Probability: X'\n\n"
            "You are estimating the probability that a patient reports a specific frequency of "
            "emotional problems in the past 7 days based on their medication."
        ),
    ),
    "insurance": AssessmentConfig(
        name="employer-based insurance",
        query_type=QueryType.BINARY,
        question="",
        system_prompt=(
            "You are a medical expert helping estimate probabilities based on medication data. "
            "You must always provide a numerical probability estimate between 0 and 1, even if uncertain. "
            "If you are unsure, provide your best estimate based on available knowledge. "
            "Never refuse to provide an estimate. "
            "Always output your final answer on a new line as 'Estimated Probability: X'\n\n"
            "You are estimating the probability that a patient has employer-based insurance "
            "based on their medication."
        ),
    ),
}


def create_conversation(
    drug: str, assessment_config: AssessmentConfig, level: Optional[str], cot: bool
) -> List[Dict]:
    """Create a conversation template for the given drug and assessment configuration."""
    prompt = assessment_config.create_prompt(drug, level)
    cot_suffix = "You may think aloud and reason step-by-step. " if cot else ""
    format_suffix = (
        "You should provide the final answer on a new line in the format: "
        "'Estimated Probability: X', where X is the probability."
    )

    return [
        {"role": "system", "content": assessment_config.system_prompt},
        {"role": "user", "content": prompt + cot_suffix + format_suffix},
    ]


def extract_probability(response_text: str) -> Optional[float]:
    """Extract probability from LLM response text using improved parsing."""
    # look for explicit probability format first
    probability_match = re.search(r"Estimated Probability:\s*(0?\.\d+|1\.0|1|0)",
                                  response_text)
    if probability_match:
        try:
            return float(probability_match.group(1))
        except ValueError:
            pass

    # look for percentage format
    percentage_match = re.search(r"(\d{1,3})%", response_text)
    if percentage_match:
        try:
            return float(percentage_match.group(1)) / 100
        except ValueError:
            pass

    # look for decimal numbers between 0 and 1
    decimal_match = re.search(r"\b(0?\.\d+|1\.0|1|0)\b", response_text)
    if decimal_match:
        try:
            return float(decimal_match.group(1))
        except ValueError:
            pass

    return None


def is_valid_probability(prob: Optional[float]) -> bool:
    """Check if the parsed probability is valid (between 0 and 1)."""
    if prob is None:
        return False
    return 0 <= prob <= 1


def generate_single_estimate(
    drug: str,
    level: Optional[str],
    assessment_config: AssessmentConfig,
    cot: bool,
    llm: LLM,
    sampling_params: SamplingParams,
    max_retries: int = 3,
) -> Tuple[Optional[float], str]:
    """Generate a single probability estimate with improved retry logic."""
    conversation = create_conversation(drug, assessment_config, level, cot)

    best_probability = None
    best_response = ""

    for attempt in range(max_retries):
        current_params = SamplingParams(
            temperature=sampling_params.temperature * (1 + attempt * 0.1),  # gradually increase temperature
            top_p=min(sampling_params.top_p + attempt * 0.05, 1.0),  # gradually increase top_p
            max_tokens=sampling_params.max_tokens,
            random_seed=42 + attempt,
        )

        output = llm.chat(messages=[conversation], sampling_params=current_params)[0]
        response_text = output.outputs[0].text
        probability = extract_probability(response_text)

        if is_valid_probability(probability):
            return probability, response_text

        # store the best attempt if we haven't found a valid probability yet
        if probability is not None and (best_probability is None or abs(probability - 0.5) < abs(best_probability - 0.5)):
            best_probability = probability
            best_response = response_text

    # if all retries failed, return the best attempt or a fallback value
    if best_probability is not None:
        return best_probability, best_response

    # use median probability as fallback for complete failures
    fallback_prob = 0.5
    logging.warning(f"Using fallback probability {fallback_prob} for drug '{drug}' after {max_retries} attempts")
    return fallback_prob, best_response or "Failed to generate valid probability estimate."


def get_checkpoint_filename(assessment: str, model_name: str, cot: bool) -> str:
    """Generate consistent checkpoint filename."""
    _model_name = model_name.split("/")[-1].lower()
    cot_suffix = "_cot" if cot else ""
    return f"results/{assessment}_{_model_name}{cot_suffix}.parquet"


def load_checkpoint(filename: str) -> Tuple[pd.DataFrame, Set[str]]:
    """Load checkpoint if it exists and return processed drugs."""
    if os.path.exists(filename):
        df = pd.read_parquet(filename)
        processed_drugs = set(df["drug"].unique())
        logging.info(f"Loaded checkpoint with {len(processed_drugs)} processed drugs")
        return df, processed_drugs
    return pd.DataFrame(), set()


def pivot_binary_results(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to pivot binary results into wide format."""
    results_df = pd.pivot_table(
        df,
        values=["probability", "llm_response"],
        index="drug",
        columns="level",
        aggfunc="first",
    ).reset_index()

    # flatten column names
    results_df.columns = [
        f"{col[0]}" if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in results_df.columns
    ]
    return results_df


def estimate_probabilities(
    drugs: List[str],
    assessment_name: str,
    cot: bool,
    model_name: str,
    llm: LLM,
    sampling_params: SamplingParams,
    batch_size: int = 1,
    checkpoint_interval: int = 100,
) -> pd.DataFrame:
    """Estimate probabilities for the specified assessment across all drugs."""
    assessment_config = ASSESSMENT_CONFIGS[assessment_name]

    # setup checkpoint
    checkpoint_file = get_checkpoint_filename(assessment_name, model_name, cot)
    results_df, processed_drugs = load_checkpoint(checkpoint_file)
    all_results = results_df.to_dict("records") if not results_df.empty else []

    # filter out already processed drugs
    remaining_drugs = [d for d in drugs if d not in processed_drugs]

    if not remaining_drugs:
        logging.info("All drugs have been processed. Using existing results.")
        return results_df

    if assessment_config.query_type == QueryType.BINARY:
        levels = [None]
    else:
        levels = assessment_config.levels

    logging.info(
        f"Starting estimation for {len(remaining_drugs)} remaining drugs with assessment '{assessment_name}'"
    )
    logging.info(f"Assessment type: {assessment_config.query_type.value}")
    if levels[0] is not None:
        logging.info(f"Levels to estimate: {len(levels)}")

    # create batches of drug-level combinations
    combinations = []
# create batches of drug-level combinations
    for drug in remaining_drugs:
        for level in levels:
            combinations.append((drug, level))

    # process combinations in batches
    drugs_since_last_save = 0
    current_drug = None
    drug_results = []

    for i in tqdm(
        range(0, len(combinations), batch_size),
        desc="Processing drug-level combinations",
        unit="batch",
    ):
        batch = combinations[i : i + batch_size]
        batch_conversations = [
            create_conversation(drug, assessment_config, level, cot)
            for drug, level in batch
        ]

        outputs = llm.chat(
            messages=batch_conversations,
            sampling_params=sampling_params,
        )

        for (drug, level), output in zip(batch, outputs):
            if current_drug != drug:
                if drug_results:  # save previous drug's results
                    all_results.extend(drug_results)
                    drugs_since_last_save += 1
                current_drug = drug
                drug_results = []

            response_text = output.outputs[0].text
            probability = extract_probability(response_text)

            level_key = level if level else "probability"
            result = {
                "drug": drug,
                "level": level_key,
                "probability": probability,
                "llm_response": response_text,
            }
            drug_results.append(result)

            # save checkpoint periodically
            if drugs_since_last_save >= checkpoint_interval:
                # add last drug's results
                all_results.extend(drug_results)
                drug_results = []

                # create and save checkpoint
                temp_df = pd.DataFrame(all_results)
                if assessment_config.query_type == QueryType.BINARY:
                    temp_df = pivot_binary_results(temp_df)
                temp_df.to_parquet(checkpoint_file, engine="pyarrow")

                logging.info(f"Checkpoint saved with {len(temp_df)} drugs")
                drugs_since_last_save = 0

        # log progress periodically
        if len(all_results) % 100 == 0:
            logging.info(f"Processed {len(all_results)} total estimations")

    # add any remaining results
    if drug_results:
        all_results.extend(drug_results)

    # create final DataFrame
    results_df = pd.DataFrame(all_results)

    # calculate and log some statistics
    total_nulls = results_df["probability"].isna().sum()
    if total_nulls > 0:
        logging.warning(
            f"Found {total_nulls} null probabilities out of {len(results_df)} total estimations "
            f"({total_nulls / len(results_df) * 100:.2f}%)"
        )

    # pivot the DataFrame for binary assessments
    if assessment_config.query_type == QueryType.BINARY:
        results_df = pivot_binary_results(results_df)

    logging.info(f"Completed estimation. Final dataset shape: {results_df.shape}")
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Estimate medical condition probabilities based on drugs."
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Huggingface model name to use."
    )
    parser.add_argument(
        "--assessment",
        type=str,
        required=True,
        choices=list(ASSESSMENT_CONFIGS.keys()),
        help="Type of assessment to perform.",
    )
    parser.add_argument(
        "--cot", action="store_true",
        help="Enable chain-of-thought reasoning."
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs to use."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature parameter for sampling.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for estimation."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="resources/drugs_15980.parquet",
        help="Input file containing drug names.",
    )

    args = parser.parse_args()

    # log the configuration
    logging.info(f"Starting estimation with configuration:")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Assessment: {args.assessment}")
    logging.info(f"Chain of thought: {args.cot}")
    logging.info(f"Number of GPUs: {args.num_gpus}")
    logging.info(f"Batch size: {args.batch_size}")

    # initialize LLM with appropriate configuration
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.num_gpus,
        dtype=torch.bfloat16,
        # use 4-bit quantization for 405B model
        quantization='bitsandbytes' if args.model_name == "meta-llama/Llama-3.1-405B-Instruct" else None,
        load_format='bitsandbytes' if args.model_name == "meta-llama/Llama-3.1-405B-Instruct" else 'auto',
        max_model_len=MAX_MODEL_LENGTH,
    )

    # set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=0.9, max_tokens=MAX_MODEL_LENGTH
    )

    # load drug data
    df = pd.read_parquet(args.input_file, engine="pyarrow")
    drugs = df["standard_concept_name"].tolist()
    logging.info(f"Loaded {len(drugs)} drugs from {args.input_file}")

    # run estimation
    results_df = estimate_probabilities(
        drugs=drugs,
        assessment_name=args.assessment,
        cot=args.cot,
        model_name=args.model_name,
        llm=llm,
        sampling_params=sampling_params,
        batch_size=args.batch_size,
    )

    # save final results (using same filename as checkpoint)
    output_file = get_checkpoint_filename(args.assessment, args.model_name, args.cot)
    results_df.to_parquet(output_file, engine="pyarrow")
    logging.info(f"Final results saved to {output_file}")


if __name__ == "__main__":
    manual_seed(42)
    main()
