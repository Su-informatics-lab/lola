"""
This script estimates the probability of various medical conditions and patient
characteristics based on medication data using LLMs.

Supported Assessments:
- Binary: Type II diabetes, AUDIT-C, insurance status
- Ordinal (5 levels): Fatigue and anxiety

Usage:
    python estimate_prob_given_drug.py --model MODEL_NAME --assessment ASSESSMENT_TYPE [options]

Required Arguments:
    --model          Huggingface model name (e.g., meta-llama/Llama-3.1-70B-Instruct)
    --assessment     Assessment type (diabetes|audit_c|fatigue|anxiety|insurance)

Optional Arguments:
    --cot           Enable chain-of-thought reasoning
    --num_gpus      Number of GPUs for tensor parallelism (default: 1)
    --temperature   Sampling temperature (default: 0.6)
    --batch_size    Batch size for estimation (default: 4)
    --input_file    Input parquet file with drug names
        (default: resources/drug_15980.parquet)
    --output_prefix Output file prefix (default: results)

Output:
    Generates a parquet file containing:
    - Drug names
    - Estimated probabilities
    - Full LLM responses
    - Probabilities for each severity level (only for ordinal assessments)
"""

import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from torch import manual_seed
from tqdm import tqdm
import logging
import sys

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
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
            return f"Given that a patient took {drug}, estimate the probability of {self.name}. "
        else:
            question_context = f"{self.question}\n\n" if self.question else ""
            return f"{question_context}Given that a patient took {drug}, estimate the probability that their response is '{level}'. "


ASSESSMENT_CONFIGS = {
    "diabetes": AssessmentConfig(
        name="Type II diabetes",
        query_type=QueryType.BINARY,
        question="",
        system_prompt="You are a medical language model designed to estimate the probability that a patient has "
                      "Type II diabetes based on a specific medicine. Always output your final answer "
                      "as a float number on a new line starting with 'Estimated Probability:'."
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
        system_prompt="You are a medical language model designed to estimate the probability that a patient has "
                      "a high-risk AUDIT-C score based on their medication. A high-risk score is 4+ for men and 3+ for women. "
                      "Always output your final answer as a float number on a new line starting with 'Estimated Probability:'."
    ),
    "fatigue": AssessmentConfig(
        name="fatigue level",
        query_type=QueryType.ORDINAL,
        question="In the past 7 days, how would you rate your fatigue?",
        levels=['Average Fatigue 7 Days: None',
                'Average Fatigue 7 Days: Mild',
                'Average Fatigue 7 Days: Moderate',
                'Average Fatigue 7 Days: Severe',
                'Average Fatigue 7 Days: Very Severe'],
        system_prompt="You are a medical language model designed to estimate the probability that a patient reports "
                      "a specific level of fatigue in the past 7 days based on their medication. "
                      "Always output your final answer as a float number on a new line starting with 'Estimated Probability:'."
    ),
    "anxiety": AssessmentConfig(
        name="emotional problems",
        query_type=QueryType.ORDINAL,
        question="In the past 7 days, how often have you been bothered by emotional problems such as feeling anxious, depressed or irritable?",
        levels=['Emotional Problem 7 Days: Never',
                'Emotional Problem 7 Days: Rarely',
                'Emotional Problem 7 Days: Sometimes',
                'Emotional Problem 7 Days: Often',
                'Emotional Problem 7 Days: Always'],
        system_prompt="You are a medical language model designed to estimate the probability that a patient reports "
                      "a specific frequency of emotional problems in the past 7 days based on their medication. "
                      "Always output your final answer as a float number on a new line starting with 'Estimated Probability:'."
    ),
    "insurance": AssessmentConfig(
        name="employer-based insurance",
        query_type=QueryType.BINARY,
        question="",
        system_prompt="You are a medical language model designed to estimate the probability that a patient has "
                      "employer-based insurance based on their medication. "
                      "Always output your final answer as a float number on a new line starting with 'Estimated Probability:'."
    )
}


def get_model_config(model_name: str) -> Dict:
    """
    Get model-specific configuration including quantization settings.
    """
    configs = {
        "meta-llama/Llama-2-7b-chat-hf": {
            "max_model_len": MAX_MODEL_LENGTH,
            "dtype": "bfloat16",
            "quantization": None
        },
        "meta-llama/Llama-2-13b-chat-hf": {
            "max_model_len": MAX_MODEL_LENGTH,
            "dtype": "bfloat16",
            "quantization": None
        },
        "meta-llama/Llama-2-70b-chat-hf": {
            "max_model_len": MAX_MODEL_LENGTH,
            "dtype": "bfloat16",
            "quantization": None
        },
        "meta-llama/Llama-3.1-8B-Instruct": {
            "max_model_len": MAX_MODEL_LENGTH,
            "dtype": "bfloat16",
            "quantization": None
        },
        "meta-llama/Llama-3.1-70B-Instruct": {
            "max_model_len": MAX_MODEL_LENGTH,
            "dtype": "bfloat16",
            "quantization": None
        },
        "meta-llama/Llama-3.1-405B-Instruct": {
            "max_model_len": MAX_MODEL_LENGTH,
            "dtype": "int8",  # using 8-bit quantization for 405B model
            "quantization": "int8"
        }
    }
    return configs.get(model_name, {
        "max_model_len": MAX_MODEL_LENGTH,
        "dtype": "bfloat16",
        "quantization": None
    })


def create_conversation(drug: str, assessment_config: AssessmentConfig,
                        level: Optional[str], cot: bool) -> List[Dict]:
    """Create a conversation template for the given drug and assessment configuration."""
    prompt = assessment_config.create_prompt(drug, level)
    cot_suffix = (
        "You may think aloud and reason step-by-step. "
        if cot else ""
    )
    format_suffix = (
        "You should provide the final answer on a new line in the format: "
        "'Estimated Probability: X', where X is the probability."
    )

    return [
        {
            "role": "system",
            "content": assessment_config.system_prompt
        },
        {
            "role": "user",
            "content": prompt + cot_suffix + format_suffix
        }
    ]


def is_valid_probability(prob: Optional[float]) -> bool:
    """Check if the parsed probability is valid (between 0 and 1)."""
    if prob is None:
        return False
    return 0 <= prob <= 1


def extract_probability(response_text: str) -> Optional[float]:
    """Extract probability from LLM response text."""
    probability_line = [line for line in response_text.split("\n")
                        if "Estimated Probability" in line]
    if not probability_line:
        return None

    try:
        prob = float(probability_line[0].split(":")[1].strip())
        return prob
    except (IndexError, ValueError):
        return None


def generate_single_estimate(
        drug: str,
        level: Optional[str],
        assessment_config: AssessmentConfig,
        cot: bool,
        llm: LLM,
        sampling_params: SamplingParams,
        max_retries: int = 100
) -> Tuple[Optional[float], str]:
    """
    Generate a single probability estimate with retry logic for invalid outputs.
    Returns both the probability and the full response text.
    """
    conversation = create_conversation(drug, assessment_config, level, cot)

    for attempt in range(max_retries):
        if attempt > 0:
            logging.warning(f"Retry {attempt}/{max_retries - 1} for drug '{drug}' "
                            f"(level: {level if level else 'binary'})")

        # modify seed for retry attempts
        current_params = SamplingParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            max_tokens=sampling_params.max_tokens,
            random_seed=42 + attempt  # vary seed for each attempt
        )

        output = llm.chat(
            messages=[conversation],
            sampling_params=current_params
        )[0]

        response_text = output.outputs[0].text
        probability = extract_probability(response_text)

        if is_valid_probability(probability):
            if attempt > 0:
                logging.info(
                    f"Successfully got valid probability after {attempt + 1} attempts "
                    f"for drug '{drug}' (level: {level if level else 'binary'})")
            return probability, response_text

    # if all retries failed, return None and the last response
    logging.error(f"Failed to get valid probability after {max_retries} attempts "
                  f"for drug '{drug}' (level: {level if level else 'binary'})")
    return None, response_text


def estimate_probabilities(
        drugs: List[str],
        assessment_name: str,
        cot: bool,
        llm: LLM,
        sampling_params: SamplingParams,
        batch_size: int = 1
) -> pd.DataFrame:
    """
    Estimate probabilities for the specified assessment across all drugs.
    Returns a single DataFrame with drug names, probabilities, and full responses.
    Processes drugs in batches to improve throughput.
    """
    assessment_config = ASSESSMENT_CONFIGS[assessment_name]
    all_results = []

    if assessment_config.query_type == QueryType.BINARY:
        levels = [None]
    else:
        levels = assessment_config.levels

    logging.info(
        f"Starting estimation for {len(drugs)} drugs with assessment '{assessment_name}'")
    logging.info(f"Assessment type: {assessment_config.query_type.value}")
    if levels[0] is not None:
        logging.info(f"Levels to estimate: {len(levels)}")

    # create batches of drug-level combinations
    combinations = []
    for drug in drugs:
        for level in levels:
            combinations.append((drug, level))

    # process combinations in batches
    for i in tqdm(range(0, len(combinations), batch_size),
                  desc="Processing drug-level combinations",
                  unit="batch"):
        batch = combinations[i:i + batch_size]
        batch_conversations = [
            create_conversation(drug, assessment_config, level, cot)
            for drug, level in batch
        ]

        # generate estimations for the batch
        outputs = llm.chat(
            messages=batch_conversations,
            sampling_params=sampling_params,
        )

        # process batch outputs
        for (drug, level), output in zip(batch, outputs):
            response_text = output.outputs[0].text
            probability = extract_probability(response_text)

            level_key = level if level else "probability"
            result = {
                "drug": drug,
                "level": level_key,
                "probability": probability,
                "llm_response": response_text
            }
            all_results.append(result)

        # log progress periodically
        if len(all_results) % 100 == 0:
            logging.info(f"Processed {len(all_results)} total estimations")

    # create DataFrame with all information
    results_df = pd.DataFrame(all_results)

    # calculate and log some statistics
    total_nulls = results_df['probability'].isna().sum()
    if total_nulls > 0:
        logging.warning(
            f"Found {total_nulls} null probabilities out of {len(results_df)} total estimations "
            f"({total_nulls / len(results_df) * 100:.2f}%)")

    # pivot the DataFrame for binary assessments to make it more compact
    if assessment_config.query_type == QueryType.BINARY:
        results_df = pd.pivot_table(
            results_df,
            values=['probability', 'llm_response'],
            index='drug',
            columns='level',
            aggfunc='first'
        ).reset_index()

        # flatten column names
        results_df.columns = [
            f"{col[0]}_{col[1]}".rstrip('_probability')
            for col in results_df.columns
        ]

    logging.info(f"Completed estimation. Final dataset shape: {results_df.shape}")
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Estimate medical condition probabilities based on drugs.")
    parser.add_argument('--model', type=str, required=True,
                        help='Huggingface model name to use.')
    parser.add_argument('--assessment', type=str, required=True,
                        choices=list(ASSESSMENT_CONFIGS.keys()),
                        help='Type of assessment to perform.')
    parser.add_argument('--cot', action='store_true',
                        help='Enable chain-of-thought reasoning.')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use.')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Temperature parameter for sampling.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for estimation.')
    parser.add_argument('--input_file', type=str,
                        default='resources/drug_15980.parquet',
                        help='Input file containing drug names.')
    parser.add_argument('--output_prefix', type=str, default='results',
                        help='Prefix for output files.')

    args = parser.parse_args()

    # log the configuration
    logging.info(f"Starting estimation with configuration:")
    logging.info(f"Model: {args.model}")
    logging.info(f"Assessment: {args.assessment}")
    logging.info(f"Chain of thought: {args.cot}")
    logging.info(f"Number of GPUs: {args.num_gpus}")
    logging.info(f"Batch size: {args.batch_size}")

    # get model-specific configuration
    model_config = get_model_config(args.model)

    # initialize LLM with appropriate configuration
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.num_gpus,
        dtype=model_config["dtype"],
        quantization=model_config["quantization"],
        max_model_len=model_config["max_model_len"]
    )

    # set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
        max_tokens=MAX_MODEL_LENGTH
    )

    # load drug data
    df = pd.read_parquet(args.input_file, engine='pyarrow')
    drugs = df['standard_concept_name'].tolist()
    logging.info(f"Loaded {len(drugs)} drugs from {args.input_file}")

    # run estimation
    results_df = estimate_probabilities(
        drugs=drugs,
        assessment_name=args.assessment,
        cot=args.cot,
        llm=llm,
        sampling_params=sampling_params,
        batch_size=args.batch_size
    )

    # save results
    cot_suffix = "_cot" if args.cot else ""
    output_file = f"{args.output_prefix}_{args.assessment}{cot_suffix}.parquet"
    results_df.to_parquet(output_file, engine='pyarrow')
    logging.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    manual_seed(42)
    main()
