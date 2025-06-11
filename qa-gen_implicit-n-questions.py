"""
Simple QA Generator for Markdown Documentation

This script processes entire markdown files at once and uses LangChain with an LLM 
to generate comprehensive question-answer pairs for the whole file.
"""

import os
import re
import json
import yaml
import csv
from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm
from dotenv import load_dotenv

# LangChain imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Local imports
from prompts import WHOLE_FILE_QA_GENERATION_PROMPT_IMPLICIT_N_QUESTIONS_V2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Env
load_dotenv(verbose=True, override=True)

class SimpleMarkdownQAGenerator:
    """Generates QA pairs from entire markdown files using LLMs."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.llm = self._initialize_llm()
        self.docs_dir = Path(self.config["docs"]["dir"])
        self.output_file = "simple_implicit_n_qa_results.csv"
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _initialize_llm(self) -> BaseChatModel:
        """Initialize the LLM based on the configuration."""
        provider = self.config["model"]["provider"]
        model_params = self.config["model"].get("model-parameters", {})
        
        if provider == "google-genai":
            model_id = self.config["model"]["id"]
            return ChatGoogleGenerativeAI(
                model=model_id,
                temperature=model_params.get("temperature", 0.1),
                # top_p=model_params.get("top-p", 0.9),
                # top_k=model_params.get("top-k", 50),
                # max_output_tokens=model_params.get("max-tokens", 2048),
            )
        elif provider == "openai":
            model_id = self.config["model"]["id"]
            return ChatOpenAI(
                model_name=model_id,
                temperature=model_params.get("temperature", 0.1)
            )
        elif provider == "anthropic":
            model_id = self.config["model"]["id"]
            return ChatAnthropic(
                model_name=model_id,
                temperature=model_params.get("temperature", 0.1),
                # max_tokens=model_params.get("max-tokens", 2048)
            )
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    def estimate_question_count(self, content: str) -> int:
        """Estimate the appropriate number of questions based on content length."""
        # Simple heuristic: roughly 1 question per 200 characters, with min of 3 and max of 20
        # You could make this more sophisticated based on your content
        char_count = len(content)
        estimated_count = max(3, min(20, char_count // 200))
        return estimated_count
    
    def process_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Process an entire markdown file and generate QA pairs."""
        logger.info(f"Processing file: {file_path}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Get relative path for source tracking
            rel_path = file_path.relative_to(self.docs_dir.parent.parent)
            
            # Simple estimation of how many questions to generate
            num_questions = self.estimate_question_count(content)
            logger.info(f"Estimated {num_questions} questions for {file_path}")
            # Generate QA pairs for the entire file
            response = self.llm.invoke(
                WHOLE_FILE_QA_GENERATION_PROMPT_IMPLICIT_N_QUESTIONS_V2.format(
                    content=content,
                )
            )
            logger.info(f"Received response for {file_path}")
            # Extract JSON from response
            response_text = response.content
            
            # Extract JSON if it's wrapped in markdown code blocks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text
            logger.info(f"Extracted JSON from response for {file_path}")

            # Parse the JSON
            qa_pairs = json.loads(json_str)
            logger.info(f"Parsed {len(qa_pairs)} QA pairs from {file_path}")
            # Add source file information to each QA pair
            for qa_pair in qa_pairs:
                # If a topic wasn't provided, use the filename
                if "topic" not in qa_pair:
                    qa_pair["topic"] = file_path.stem
                # Format the source as file_path/topic
                qa_pair["source"] = f"{rel_path}/{qa_pair['topic']}"
            logger.info(f"Added source information to QA pairs for {file_path}")
            
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise e
            return []
    
    def save_to_csv(self, qa_pairs: List[Dict[str, str]], output_file: str = None) -> None:
        """Save QA pairs to a CSV file."""
        if not output_file:
            output_file = self.output_file
            
        if not qa_pairs:
            logger.warning("No QA pairs to save.")
            return
        
        field_names = ["question", "answer", "source"]
        
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for qa_pair in qa_pairs:
                # Create a clean row with only the required fields
                row = {
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "source": qa_pair["source"]
                }
                writer.writerow(row)
            
        logger.info(f"Saved {len(qa_pairs)} QA pairs to {output_file}")
    
    def run(self) -> None:
        """Process all markdown files in the configured directory."""
        all_qa_pairs = []
        
        # Get all markdown files in the directory
        md_files = list(self.docs_dir.glob("*.md"))
        
        if not md_files:
            logger.warning(f"No markdown files found in {self.docs_dir}")
            return
        
        logger.info(f"Found {len(md_files)} markdown files to process")
        
        for md_file in tqdm(md_files, desc="Processing files", unit="file"):
            qa_pairs = self.process_file(md_file)
            all_qa_pairs.extend(qa_pairs)
            
        self.save_to_csv(all_qa_pairs)
        logger.info("Simple QA generation complete!")

def main():
    """Main entry point."""
    generator = SimpleMarkdownQAGenerator()
    generator.run()

if __name__ == "__main__":
    main()
