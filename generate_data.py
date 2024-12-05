"""Generate training data for LLM
Step 1: Parse data from file
Step 2: Generate questions and answers chunk by chunk
    Step 2.1: generate questions batch by batch and filter out similar questions
    Step 2.2: generate answers for the questions
Step 3: Save the data to a file
"""

import sys
import os
import argparse
from typing import List, Dict, Optional, Union
import json
import shutil

import tqdm

from litellm import completion
from pydantic import BaseModel
from dotenv import load_dotenv
from loguru import logger

logger.remove()
import prompts
from filter import SemanticFilter
from file_parser import parse_file


class LLMConfig(BaseModel):
    """LLM config"""
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.5


def chat(
    llm_config: LLMConfig,
    messages: List[Dict[str, str]],
    response_format: BaseModel = None,
    retry: int = 3,
) -> Union[str, BaseModel, None]:
    """Chat with LLM response string context
    if response_format is provided , response will be parsed to response_format

    """
    for _ in range(retry):
        try:
            resp = completion(
                model=llm_config.model,
                messages=messages,
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
                response_format=response_format,
            )
            content = resp.choices[0].message.content
            if response_format is not None:
                return response_format(**json.loads(content))
            return content
        except json.JSONDecodeError:
            logger.error(f"Failed to parse response from LLM: {content}")
            # parse content from string ```json\n{content}\n```
            try:
                content = content.split("```json")[1].split("```")[0]
                return response_format(**json.loads(content))
            except Exception as e:
                logger.error(f"Failed to parse response from LLM: {e}")
        except Exception as e:
            logger.error(f"Failed to get response from LLM: {e}")
    logger.error("Failed to get response from LLM")
    return None


def filter_questions(
    questions: List[str], semantic_filter: SemanticFilter, thredshold=0.7
) -> List[str]:
    """Filter out similar questions by similarity threshold if similarity is greater than threshold, the question will be filtered out"""
    if semantic_filter is None:
        logger.warning("SemanticFilter is not initialized, skip filtering")
        return questions
    num_before_filter = len(questions)
    questions = semantic_filter.filter(questions, thredshold)
    num_after_filter = len(questions)
    logger.info(
        f"Filtered {num_before_filter - num_after_filter} questions, {num_after_filter} questions left"
    )
    return questions


def generate_questions(
    llm_config: LLMConfig,
    context: str,
    batch_num: int = 10,
    history_questions: List[str] = [],
) -> List[str]:
    """Generate questions from data need to avoid duplicate questions so use history questions to avoid duplicate"""

    class Questions(BaseModel):
        questions: List[str]

    question_list = []
    questions: Questions = chat(
        llm_config,
        [
            {
                "role": "user",
                "content": prompts.GENERATE_QUESTION_PROMPT.format(
                    context=context,
                    avoid_duplicate="\n".join(history_questions),
                    target_num=batch_num,
                ),
            }
        ],
        response_format=Questions,
        retry=3,
    )
    if questions is None:
        logger.error("Failed to generate questions")
        return []
    logger.debug(f"Generated questions output: {questions}")
    question_list.extend(questions.questions)
    logger.info(f"Generated {len(question_list)} questions")
    return question_list


class QAPair(BaseModel):
    question: str
    answer: Optional[str] = None


def generate_answer(llm_config: LLMConfig, question: str, context: str) -> str:
    """LLM generate answer for the question"""

    class Answer(BaseModel):
        answer: str

    answer: Answer = chat(
        llm_config,
        [
            {
                "role": "user",
                "content": prompts.ANSWER_QUESTION_PROMPT.format(
                    context=context, question=question
                ),
            }
        ],
        response_format=Answer,
        retry=3,
    )
    if answer is None:
        logger.error(f"Failed to generate answer for question: {question}")
        return ""
    return answer.answer


def save_qa_pairs(qa_pairs: List[QAPair], output_file: str):
    """Save QA pairs to a file to jsonl format"""
    content = "\n".join(
        [
            json.dumps(
                {"question": qa_pair.question, "answer": qa_pair.answer},
                ensure_ascii=False,
            )
            for qa_pair in qa_pairs
        ]
    )
    # overwrite the file
    with open(output_file, "w") as f:
        f.write(content)


def save_training_data(qa_pairs: List[QAPair], output_file: str):
    """Save training data to a file to jsonl format"""
    content = "\n".join(
        [
            json.dumps(
                {"instruction": qa_pair.question, "output": qa_pair.answer},
                ensure_ascii=False,
            )
            for qa_pair in qa_pairs
        ]
    )
    with open(output_file, "w") as f:
        f.write(content)


def load_qa_pairs(file_path: str) -> List[QAPair]:
    """Load QA pairs from a file load from jsonl format"""
    return [QAPair(**json.loads(line)) for line in load_lines(file_path)]


def split_context(context: str, chunk_size: Optional[int] = None) -> List[str]:
    """Split context into chunks"""
    context = context.strip()
    if chunk_size is None or chunk_size == -1 or chunk_size < 100:
        logger.warning(
            "Chunk size is not provided, return the whole context as one chunk, might cuase llm OOM"
        )
        return [context]
    return [context[i : i + chunk_size] for i in range(0, len(context), chunk_size)]


def save_lines(lines: List[str], file_path: str):
    """Save lines to a file"""
    with open(file_path, "w") as f:
        for line in lines:
            f.write(line + "\n")


def load_lines(file_path: str) -> List[str]:
    """Load lines from a file"""
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def merge_qa_pairs(
    qa_pairs: List[QAPair], loaded_qa_pairs: List[QAPair]
) -> List[QAPair]:
    """Merge two qa pairs"""
    loaded_questions = {x.question: x for x in loaded_qa_pairs}
    for qa_pair in qa_pairs:
        if qa_pair.question in loaded_questions:
            qa_pair.answer = loaded_questions[qa_pair.question].answer
    return qa_pairs


def main(args):
    # load environment variables
    env_path = args.env_path
    load_dotenv(env_path)

    # set log level
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger.add(sys.stdout, level=log_level)

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"The file {args.input_file} does not exist")

    # load env
    LLM_MODEL = os.getenv("LLM_MODEL")
    if LLM_MODEL == "":
        raise ValueError("LLM_MODEL is not set")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    if LLM_BASE_URL == "":
        LLM_BASE_URL = None
    # set llm config
    llm_config = LLMConfig(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        temperature=float(os.getenv("LLM_TEMPERATURE", 0.5)),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", 4096)),
    )

    try:
        similarity_model_name = os.getenv(
            "SIMILARITY_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        logger.info(f"Start loading semantic filter model: {similarity_model_name}")
        # init semantic filter
        semantic_filter = SemanticFilter(similarity_model_name)
        logger.info(f"Semantic filter model loaded")
        # semantic_filter = None
    except Exception as e:
        logger.error(f"Failed to initialize SemanticFilter: {e}")
        semantic_filter = None

    GENERATE_QUESTIONS_BATCH_NUM = int(os.getenv("GENERATE_QUESTIONS_BATCH_NUM", 10))

    logger.info(f"Step 1: Parsing file {args.input_file}")
    context = parse_file(args.input_file)
    logger.debug(f"Parsed context: {context}")

    MIN_CONTEXT_CHUNK_SIZE = 2000
    # Split context into chunks
    CONTEXT_CHUNK_SIZE = int(os.getenv("CONTEXT_CHUNK_SIZE", MIN_CONTEXT_CHUNK_SIZE))
    if CONTEXT_CHUNK_SIZE == -1:
        logger.warning("Context chunk size is -1, disable chunking")
        CONTEXT_CHUNK_SIZE = None
    elif CONTEXT_CHUNK_SIZE < MIN_CONTEXT_CHUNK_SIZE:
        logger.warning(
            f"Context chunk size is too small, set to {MIN_CONTEXT_CHUNK_SIZE}, got {CONTEXT_CHUNK_SIZE}"
        )
        CONTEXT_CHUNK_SIZE = MIN_CONTEXT_CHUNK_SIZE

    context_chunks = split_context(context, chunk_size=CONTEXT_CHUNK_SIZE)
    chunk_num = len(context_chunks)
    each_chunk_qa_num = args.qa_num // chunk_num
    qa_pairs = []  # all chunks qa pairs
    logger.info(
        f"Step 2: Generating questions and answers chunk by chunk, chunk_size:{chunk_num}"
    )
    RESUME = args.resume
    OUTPUT_FOLDER = args.output_folder
    if RESUME:
        logger.info(f"Resume from checkpoint, output folder: {OUTPUT_FOLDER}")
    else:
        shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for i in tqdm.tqdm(range(chunk_num)):
        context_chunk = context_chunks[i]
        chunk_questions_path = os.path.join(OUTPUT_FOLDER, f"chunk_{i}_questions.txt")
        if os.path.exists(chunk_questions_path):

            chunk_questions = load_lines(chunk_questions_path)
            logger.info(
                f"Load {len(chunk_questions)} questions from {chunk_questions_path}"
            )
        else:
            chunk_questions = []
        while len(chunk_questions) < each_chunk_qa_num:
            # generate questions batch by batch
            chunk_questions.extend(
                generate_questions(
                    llm_config,
                    context_chunk,
                    history_questions=chunk_questions,
                    batch_num=GENERATE_QUESTIONS_BATCH_NUM,
                )
            )
            # filter questions
            chunk_questions = filter_questions(chunk_questions, semantic_filter)
            save_lines(chunk_questions, chunk_questions_path)

        # format qa pairs from questions
        chunk_qa_pairs = [QAPair(question=x) for x in chunk_questions]

        # load qa pairs from file
        chunk_qa_pairs_path = os.path.join(OUTPUT_FOLDER, f"chunk_{i}_qa_pairs.jsonl")
        if os.path.exists(chunk_qa_pairs_path):
            logger.info(f"Loading chunk {i} qa pairs from {chunk_qa_pairs_path}")
            loaded_qa_pairs = load_qa_pairs(chunk_qa_pairs_path)
            chunk_qa_pairs = merge_qa_pairs(chunk_qa_pairs, loaded_qa_pairs)

        # loop through qa pair to generate answer
        for qa_pair in chunk_qa_pairs:
            if qa_pair.answer in [None, ""]:
                # generate answer and update qa pair
                qa_pair.answer = generate_answer(
                    llm_config, qa_pair.question, context_chunk
                )
                # save qa pairs
                save_qa_pairs(chunk_qa_pairs, chunk_qa_pairs_path)

        # extend qa pair to all chunks qa pairs
        qa_pairs.extend(chunk_qa_pairs)

    logger.info(f"Step 3: Saving data to {args.output_folder}")
    output_file_path = os.path.join(args.output_folder, "qa_pairs.jsonl")
    save_qa_pairs(qa_pairs, output_file_path)
    output_file_path = os.path.join(args.output_folder, "training_data.jsonl")
    save_training_data(qa_pairs, output_file_path)


def parse_args():
    """Parse arguments
    ARGS:
        input_file: input file path
        env_path: environment file path
        qa_num: number of QA pairs to generate
        output_folder: output folder path
        resume: resume from checkpoint
    """
    parser = argparse.ArgumentParser(description="Generate training data for LLM")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Data file path current only support txt file",
    )
    parser.add_argument(
        "--env_path",
        type=str,
        required=False,
        help="Environment file path",
        default=".env",
    )
    parser.add_argument(
        "--qa_num",
        type=int,
        required=False,
        help="Number of QA pairs to generate",
        default=20,
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=False,
        help="Output folder path",
        default="output",
    )
    parser.add_argument(
        "--resume",
        type=bool,
        required=False,
        help="Resume from checkpoint",
        default=True,
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    main(args)
