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
    --enforce       Enforce LLMs to provide estimation even they are uncertain
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
from typing import Dict, List, Optional, Set, Tuple

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
    system_prompt: str
    question: str
    levels: Optional[List[str]] = None

    def create_prompt(self, drug: str, level: Optional[str] = None, cot: bool = False
                      ) -> str:
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
                "Please provide your final answer on a new line in the format: "
                "'Estimated Probability: X', where X is the probability."
            )
        else:
            if self.question:
                base_prompt = (
                    f"For a patient taking {drug}, what is the probability they would report '{level}'?\n\n"
                    "Please provide your final answer on a new line in the format: "
                    "'Estimated Probability: X', where X is the probability."
                )
            else:
                base_prompt = (
                    f"For a patient taking {drug}, estimate the probability of {level}. "
                    "Please provide your final answer on a new line in the format: "
                    "'Estimated Probability: X', where X is the probability."
                )

        if cot:
            base_prompt += "\nYou may think aloud and reason step-by-step."

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
    prompt = assessment_config.create_prompt(drug, level, cot, enforce)
    system_prompt = assessment_config.get_system_prompt(enforce)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

# updated assessment configurations with non-enforcement prompts
ASSESSMENT_CONFIGS = {
    "diabetes": AssessmentConfig(
        name="Type II diabetes",
        query_type=QueryType.BINARY,
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a patient has "
            "Type II diabetes based on a specific medicine. Your goal is to provide the probability as a clear float. "
            "Please keep your reasoning concise and avoid unnecessary explanations. Always output your final answer "
            "as a float number on a new line starting with 'Estimated Probability:'"
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
            "You are a medical language model designed to estimate the probability that a patient has "
            "a high-risk AUDIT-C score based on their medication. Please provide the probability as a clear float. "
            "Always output your final answer on a new line starting with 'Estimated Probability:'"
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
            "You are a medical language model designed to estimate the probability of different "
            "fatigue levels based on medication data. Please provide the probability as a clear float. "
            "Always output your final answer on a new line starting with 'Estimated Probability:'"
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
            "You are a medical language model designed to estimate the probability of different "
            "frequencies of emotional problems based on medication data. Please provide the probability as a clear float. "
            "Always output your final answer on a new line starting with 'Estimated Probability:'"
        ),
    ),
    "insurance": AssessmentConfig(
        name="employer-based insurance",
        query_type=QueryType.BINARY,
        question="",
        system_prompt=(
            "You are a medical language model designed to estimate the probability that a patient has "
            "employer-based insurance based on their medication. Please provide the probability as a clear float. "
            "Always output your final answer on a new line starting with 'Estimated Probability:'"
        ),
    ),
}

def extract_probability(response_text: str) -> Optional[float]:
    """Extract probability from LLM response with improved parsing."""
    # first look for explicit probability format
    probability_match = re.search(r"Estimated Probability:\s*(0?\.\d+|1\.0|1|0)", response_text)
    if probability_match:
        try:
            return float(probability_match.group(1))
        except ValueError:
            pass

    # then look for percentage format
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
            prob = float(decimal_match.group(1))
            if 0 <= prob <= 1:
                return prob
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
    max_retries: int = 5,
) -> Tuple[Optional[float], str]:
    """Generate a single probability estimate with improved handling."""
    conversation = create_conversation(drug, assessment_config, level, cot, enforce)
    
    for attempt in range(max_retries):
        current_params = SamplingParams(
            temperature=sampling_params.temperature * (1 + attempt * 0.1),
            top_p=min(sampling_params.top_p + attempt * 0.05, 1.0),
            max_tokens=sampling_params.max_tokens,
        )
        
        output = llm.chat(messages=[conversation], sampling_params=current_params)[0]
        response_text = output.outputs[0].text
        probability = extract_probability(response_text)
        
        if probability is not None and 0 <= probability <= 1:
            return probability, response_text
    
    # if we couldn't get a valid probability after reties, log warning and return None
    logging.warning(f"Could not extract valid probability for drug '{drug}' after {max_retries} attempts")
    return None, response_text

def estimate_probabilities(
    drugs: List[str],
    assessment_name: str,
    cot: bool,
    enforce: bool,
    model_name: str,
    llm: LLM,
    sampling_params: SamplingParams,
    batch_size: int = 1,
    checkpoint_interval: int = 100,
) -> pd.DataFrame:
    """Main estimation function with improved handling and checkpointing."""
    assessment_config = ASSESSMENT_CONFIGS[assessment_name]
    checkpoint_file = f"results/{assessment_name}_{model_name.split('/')[-1].lower()}_{'cot' if cot else 'nocot'}.parquet"
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        results_df = pd.read_parquet(checkpoint_file)
        processed_drugs = set(results_df["drug"].unique())
        remaining_drugs = [d for d in drugs if d not in processed_drugs]
    else:
        results_df = pd.DataFrame()
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
                    drug, level, assessment_config, cot, enforce, llm, sampling_params
                )
                
                result = {
                    "drug": drug,
                    "level": level if level else "probability",
                    "probability": probability,
                    "llm_response": response,
                }
                drug_results.append(result)
            
            all_results.extend(drug_results)
            
            # save checkpoint periodically
            if len(all_results) % checkpoint_interval == 0:
                temp_df = pd.concat([results_df, pd.DataFrame(all_results)])
                temp_df.to_parquet(checkpoint_file)
                logging.info(f"Checkpoint saved with {len(temp_df)} total records")

    # final save
    final_df = pd.concat([results_df, pd.DataFrame(all_results)])
    final_df.to_parquet(checkpoint_file)
    
    return final_df

def main():
    parser = argparse.ArgumentParser(
        description="Estimate medical condition probabilities based on drugs."
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Huggingface model name to use"
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
        help="Enable chain-of-thought reasoning"
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

    args = parser.parse_args()

    logging.info(f"Starting estimation with configuration:")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Assessment: {args.assessment}")
    logging.info(f"Chain of thought: {args.cot}")
    logging.info(f"Chain of thought: {args.enforce}")

    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.num_gpus,
        dtype=torch.bfloat16,
        max_model_len=MAX_MODEL_LENGTH,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
        max_tokens=MAX_MODEL_LENGTH,
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
        batch_size=args.batch_size,
    )

    logging.info(f"Estimation complete. Final dataset shape: {results_df.shape}")

if __name__ == "__main__":
    manual_seed(42)
    main()