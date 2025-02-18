"""
This script estimates the probability of various medical conditions and patient
characteristics based on medication data using LLMs.

Supported Assessments:
- Binary: Type II diabetes, AUDIT-C, insurance status, alcohol abuse
- Ordinal (5 levels): Fatigue and anxiety

Usage:
    python estimate_prob_given_drug.py --model_name MODEL_NAME \
    --assessment ASSESSMENT_TYPE [options]

Required Arguments:
    --model_name     Huggingface model name or local model path (e.g., meta-llama/Llama-3.1-70B-Instruct or ~/llama-dl/llama1_7b)
    --assessment     Assessment type (diabetes|audit_c|fatigue|anxiety|insurance|alcohol_abuse)

Optional Arguments:
    --cot           Enable chain-of-thought reasoning
    --enforce       Enforce LLMs to provide estimation even uncertain
    --num_gpus      Number of GPUs for tensor parallelism (default: 1)
    --temperature   Sampling temperature (default: 0.6)
    --batch_size    Batch size for estimation (default: 4)
    --input_file    Input parquet file with drug names
        (default: resources/drugs_15980.parquet)

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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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
    question: str
    system_prompt: str
    levels: Optional[List[str]] = None

    def create_prompt(self, drug: str, level: Optional[str] = None, cot: bool = False) -> str:
        """
        Create a simple, direct prompt for the assessment.

        Args:
            drug: Name of the drug
            level: Level for ordinal assessments (optional)
            cot: Whether to include chain-of-thought reasoning instruction
        """
        if self.query_type == QueryType.BINARY:
            # include questionnaire info if available
            questionnaire_info = f"\n\n{self.question}\n\n" if self.question else "\n\n"
            base_prompt = (
                f"Given that a patient took {drug}, estimate the probability that they have {self.name}."
                f"{questionnaire_info}"
                "Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
            )
        else:
            if self.question:
                base_prompt = (
                    f"For a patient taking {drug}, what is the probability they would report '{level}'?\n\n"
                    "Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
                )
            else:
                base_prompt = (
                    f"For a patient taking {drug}, estimate the probability of {level}. "
                    "Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
                )

        if cot:
            base_prompt += "\nYou may think aloud and reason step-by-step before reaching the final answer."

        return base_prompt

    def get_system_prompt(self, enforce: bool = False) -> str:
        """
        Get system prompt with optional enforcement language.

        Args:
            enforce: Whether to add enforcement language
        """
        base_system_prompt = self.system_prompt

        if enforce:
            enforcement_addition = (
                " You must always provide a numerical probability estimate between 0 and 1, "
                "even if uncertain. If you are unsure, provide your best estimate based on "
                "available knowledge. You cannot refuse to provide an estimate."
            )
            base_system_prompt += enforcement_addition

        return base_system_prompt


def create_conversation(
        drug: str, assessment_config: AssessmentConfig, level: Optional[str], cot: bool,
        enforce: bool = False
) -> List[Dict]:
    """
    Create a conversation template.

    Args:
        drug: Name of the drug
        assessment_config: Configuration for the assessment
        level: Level for ordinal assessments (optional)
        cot: Whether to include chain-of-thought reasoning
        enforce: Whether to add enforcement language
    """
    prompt = assessment_config.create_prompt(drug, level, cot)
    system_prompt = assessment_config.get_system_prompt(enforce)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

ASSESSMENT_CONFIGS = {
    "diabetes": AssessmentConfig(
        name="Type II diabetes",
        query_type=QueryType.BINARY,
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a patient has Type II diabetes based on the specific medicine they use. Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        ),
    ),
    "audit_c": AssessmentConfig(
        name="high-risk AUDIT-C score (4+ for men, 3+ for women)",
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
            "You are a medical language model designed to estimate the probability that a patient has a high-risk AUDIT-C score based on the specific medicine they use. Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        ),
    ),
    "fatigue": AssessmentConfig(
        name="fatigue level",
        query_type=QueryType.ORDINAL,
        question="In the past 7 days, how would you rate your fatigue?",
        levels=[
            "None",
            "Mild",
            "Moderate",
            "Severe",
            "Very Severe",
        ],
        system_prompt=(
             "You are a medical language model designed to estimate the probability of different fatigue levels a patient has based on the specific medicine they use. Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        ),
    ),
    "anxiety": AssessmentConfig(
        name="emotional problems",
        query_type=QueryType.ORDINAL,
        question="In the past 7 days, how often have you been bothered by emotional problems such as feeling anxious, depressed or irritable?",
        levels=[
            "Never",
            "Rarely",
            "Sometimes",
            "Often",
            "Always",
        ],
        system_prompt=(
            "You are a medical language model designed to estimate the probability of different frequencies of emotional problems based on the specific medicine they use. Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        ),
    ),
    "insurance": AssessmentConfig(
        name="employer-based insurance",
        query_type=QueryType.BINARY,
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a patient has employer-based insurance based on the specific medicine they use. Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        ),
    ),
    "alcohol_abuse": AssessmentConfig(
        name="alcohol abuse",
        query_type=QueryType.BINARY,
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a patient has alcohol abuse based on the specific medicine they use. "
            "For this task, refer to the following ICD-10 codes as definitions for alcohol abuse:\n\n"
            "F10 - Alcohol-related disorders:\n"
            "  F10.1 Alcohol abuse (F10.10 Uncomplicated, F10.11 In remission, F10.12 With intoxication (uncomplicated, delirium, unspecified), "
            "F10.13 With withdrawal (uncomplicated, delirium, perceptual disturbance, unspecified), F10.14 With alcohol-induced mood disorder, "
            "F10.15 With alcohol-induced psychotic disorder (delusions, hallucinations, unspecified), F10.18 With other alcohol-induced disorders "
            "(anxiety, sexual dysfunction, sleep disorder, other), F10.19 With unspecified alcohol-induced disorder)\n"
            "  F10.2 Alcohol dependence (F10.20 Uncomplicated, F10.21 In remission, F10.22 With intoxication (uncomplicated, delirium, unspecified), "
            "F10.23 With withdrawal (uncomplicated, delirium, perceptual disturbance, unspecified), F10.24 With alcohol-induced mood disorder, "
            "F10.25 With alcohol-induced psychotic disorder (delusions, hallucinations, unspecified), F10.26 With alcohol-induced persisting amnestic disorder, "
            "F10.27 With alcohol-induced persisting dementia, F10.28 With other alcohol-induced disorders (anxiety, sexual dysfunction, sleep disorder, other), "
            "F10.29 With unspecified alcohol-induced disorder)\n"
            "E52 - Niacin deficiency (pellagra)\n"
            "G62.1 - Alcoholic polyneuropathy\n"
            "I42.6 - Alcoholic cardiomyopathy\n"
            "K29.2 - Alcoholic gastritis\n"
            "K70 - Alcoholic liver disease (K70.0 Fatty liver, K70.3 Cirrhosis of liver, K70.9 Unspecified)\n"
            "T51 - Toxic effect of alcohol (T51.0 Ethanol, T51.1 Methanol, T51.2 2-Propanol, T51.3 Fusel oil, T51.8 Other alcohols, T51.9 Unspecified alcohol)\n"
            "Z50.2 - Alcohol rehabilitation\n"
            "Z71.4 - Alcohol abuse counseling and surveillance\n"
            "Z72.1 - Alcohol use\n\n"
            "Estimate the probability that a patient has alcohol abuse based solely on the medication data provided. "
            "Provide the probability enclosed within [ESTIMATION] and [/ESTIMATION] tags."
        )
    ),

}


def extract_probability(response_text: str) -> Optional[float]:
    """Extract probability from LLM response that uses [ESTIMATION] tags."""
    if not response_text:
        return None

    tag_match = re.search(r'\[ESTIMATION\](.*?)\[/ESTIMATION\]', response_text, re.DOTALL)
    if not tag_match:
        return None

    estimation_text = tag_match.group(1).strip()

    try:
        value = float(estimation_text)
        # check for nan/inf values explicitly
        if not np.isfinite(value):
            return None
        # validate probability bounds
        if 0 <= value <= 1:
            return value
        return None
    except ValueError:
        # handle percentage format
        percentage_match = re.search(r'(\d+(?:\.\d+)?)%', estimation_text)
        if percentage_match:
            try:
                value = float(percentage_match.group(1)) / 100
                if np.isfinite(value) and 0 <= value <= 1:
                    return value
            except ValueError:
                pass
    return None


def generate_single_estimate(
        drug: str,
        level: Optional[str],
        assessment_config: AssessmentConfig,
        cot: bool,
        enforce: bool,
        llm: LLM,
        sampling_params: SamplingParams,
        global_seed: int,
) -> Tuple[Optional[float], str]:
    """Generate a single probability estimate with global seed tracking."""
    conversation = create_conversation(drug, assessment_config, level, cot, enforce)
    MAX_RETRIES = 10

    for attempt in range(MAX_RETRIES):
        try:
            current_params = SamplingParams(
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                max_tokens=sampling_params.max_tokens,
                seed=global_seed + attempt  # increment from global seed
            )

            output = llm.chat(messages=[conversation], sampling_params=current_params)[0]
            if not output or not output.outputs:
                continue

            response_text = output.outputs[0].text
            probability = extract_probability(response_text)

            if probability is not None:
                return probability, response_text

            logging.warning(
                f"Attempt {attempt + 1}/{MAX_RETRIES}: No valid probability found "
                f"for drug '{drug}'. Retrying with seed {global_seed + attempt + 1}."
            )

        except Exception as e:
            logging.error(f"Error generating estimate for drug '{drug}' with seed {global_seed + attempt}: {str(e)}")
            continue

    return None, response_text if 'response_text' in locals() else ""


def estimate_probabilities(
        drugs: List[str],
        assessment_name: str,
        cot: bool,
        enforce: bool,
        model_name: str,
        llm: LLM,
        sampling_params: SamplingParams,
        global_seed: int,
        batch_size: int = 1,
        checkpoint_interval: int = 100,
) -> pd.DataFrame:
    """Main estimation function with improved seed tracking."""
    if not drugs:
        return pd.DataFrame()

    assessment_config = ASSESSMENT_CONFIGS.get(assessment_name)
    if not assessment_config:
        raise ValueError(f"Invalid assessment name: {assessment_name}")

    os.makedirs("results", exist_ok=True)

    # determine model shortname based on whether model_name is a local path or a remote identifier
    model_path = os.path.expanduser(model_name)
    if os.path.isdir(model_path):
        model_shortname = os.path.basename(os.path.normpath(model_path)).lower()
    else:
        model_shortname = model_name.split('/')[-1].lower()

    status_suffix = '_'.join(filter(None, [
        'cot' if cot else '',
        'enforce' if enforce else '',
        f'seed{global_seed}'
    ]))
    checkpoint_file = f"results/{assessment_name}_{model_shortname}{f'_{status_suffix}' if status_suffix else ''}.parquet"

    # load checkpoint with error handling
    results_df = pd.DataFrame()
    if os.path.exists(checkpoint_file):
        try:
            results_df = pd.read_parquet(checkpoint_file)
            processed_drugs = set(results_df["drug"].unique())
            remaining_drugs = [d for d in drugs if d not in processed_drugs]
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
            remaining_drugs = drugs
    else:
        remaining_drugs = drugs

    if not remaining_drugs:
        return results_df

    levels = [None] if assessment_config.query_type == QueryType.BINARY else assessment_config.levels
    all_results = []

    for i in tqdm(range(0, len(remaining_drugs), batch_size)):
        batch_drugs = remaining_drugs[i:i + batch_size]

        for drug in batch_drugs:
            drug_results = []
            for level in levels:
                probability, response = generate_single_estimate(
                    drug, level, assessment_config, cot, enforce, llm, sampling_params,
                    global_seed=global_seed
                )

                result = {
                    "drug": drug,
                    "probability": probability,
                    "llm_response": response,
                    "seed": global_seed,
                }
                if assessment_config.query_type == QueryType.ORDINAL:
                    result["level"] = level
                drug_results.append(result)

            all_results.extend(drug_results)

            if len(all_results) % checkpoint_interval == 0:
                try:
                    temp_df = pd.concat([results_df, pd.DataFrame(all_results)])
                    temp_df.to_parquet(checkpoint_file)
                    logging.info(f"Checkpoint saved with {len(temp_df)} total records (seed: {global_seed})")
                except Exception as e:
                    logging.error(f"Error saving checkpoint (seed: {global_seed}): {str(e)}")

    try:
        final_df = pd.concat([results_df, pd.DataFrame(all_results)])
        final_df.to_parquet(checkpoint_file)
        logging.info(f"Final results saved (seed: {global_seed})")
    except Exception as e:
        logging.error(f"Error saving final results (seed: {global_seed}): {str(e)}")
        final_df = pd.concat([results_df, pd.DataFrame(all_results)])

    return final_df

def main():
    parser = argparse.ArgumentParser(
        description="Estimate medical condition probabilities based on drugs."
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Huggingface model name or local model path to use"
    )
    parser.add_argument(
        "--assessment",
        type=str,
        required=True,
        choices=list(ASSESSMENT_CONFIGS.keys()),
        help="Type of assessment to perform",
    )
    parser.add_argument(
        "--cot", action="store_true",
        help="Enable chain-of-thought reasoning"
    )
    parser.add_argument(
        "--enforce", action="store_true",
        help="Enforce LLMs to provide estimation even when uncertain"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Temperature parameter for sampling"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for estimation"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="resources/drugs_15980.parquet",
        help="Input file containing drug names",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed for reproducibility"
    )

    args = parser.parse_args()
    manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.info(f"Starting estimation with configuration:")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Assessment: {args.assessment}")
    logging.info(f"Chain of thought: {args.cot}")
    logging.info(f"Enforce: {args.enforce}")

    # Expand model path and determine if it's local or remote
    model_path = os.path.expanduser(args.model_name)
    if os.path.isdir(model_path):
        logging.info(f"Loading local model from {model_path}")
    else:
        logging.info(f"Loading remote model: {args.model_name}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.num_gpus,
        dtype=torch.bfloat16,
        max_model_len=MAX_MODEL_LENGTH,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
        max_tokens=MAX_MODEL_LENGTH,
        seed=args.seed
    )

    df = pd.read_parquet(args.input_file)
    drugs = df["standard_concept_name"].tolist()

    results_df = estimate_probabilities(
        drugs=drugs,
        assessment_name=args.assessment,
        cot=args.cot,
        enforce=args.enforce,
        model_name=args.model_name,
        llm=llm,
        sampling_params=sampling_params,
        global_seed=args.seed,
        batch_size=args.batch_size,
    )

    logging.info(f"Estimation complete. Final dataset shape: {results_df.shape}")

if __name__ == "__main__":
    main()
