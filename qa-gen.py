"""
QA Generator for Markdown Documentation

This script parses markdown files, extracts headings and content,
and uses LangChain with any LLM to generate question-answer pairs
that cover the entire content.
"""

import os
import re
import json
import yaml
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

# LangChain imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.json import SimpleJsonOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Local imports
from prompts import QA_GENERATION_PROMPT, QUESTION_COUNT_PROMPT

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Regular expression to match markdown headings
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

class MarkdownQAGenerator:
    """Generates QA pairs from markdown content using LLMs."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.llm = self._initialize_llm()
        self.docs_dir = Path(self.config["docs"]["dir"])
        self.output_file = "qa_results.csv"
        self.json_parser = SimpleJsonOutputParser()
    
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
                temperature=model_params.get("temperature", 0.7),
                top_p=model_params.get("top-p", 0.9),
                top_k=model_params.get("top-k", 50),
                max_output_tokens=model_params.get("max-tokens", 2048),
            )
        elif provider == "openai":
            model_id = self.config["model"]["id"]
            return ChatOpenAI(
                model_name=model_id,
                temperature=model_params.get("temperature", 0.7)
            )
        elif provider == "anthropic":
            model_id = self.config["model"]["id"]
            return ChatAnthropic(
                model_name=model_id,
                temperature=model_params.get("temperature", 0.7),
                max_tokens=model_params.get("max-tokens", 2048)
            )
        else:
            raise ValueError(f"Unsupported model provider: {provider}")
    
    def extract_headings_and_content(self, md_text: str) -> List[Tuple[str, str, str]]:
        """
        Extract headings and their content from markdown text.
        
        Returns:
            List of tuples (heading_text, content, heading_path)
            heading_path is a hierarchical representation like "top_heading/sub_heading"
        """
        matches = list(HEADING_RE.finditer(md_text))
        results = []
        
        if not matches:
            return results
        
        # Track heading hierarchy
        heading_stack = []
        
        for i, match in enumerate(matches):
            level = len(match.group(1))  # Count the number of # characters
            title = match.group(2).strip()
            
            # Update heading stack based on current heading level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            
            heading_stack.append((level, title))
            
            # Find the content (text between this heading and the next)
            start = match.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(md_text)
            content = md_text[start:end].strip()
            
            # Create heading path for tracking
            heading_path = "/".join(h[1] for h in heading_stack)
            
            results.append((title, content, heading_path))
        
        return results
    
    def determine_question_count(self, heading: str, content: str) -> int:
        """Determine the optimal number of questions for the content length and complexity."""
        try:
            result = self.llm.invoke(
                QUESTION_COUNT_PROMPT.format(
                    heading=heading,
                    content=content
                )
            )
            # Extract the number from the response
            count_text = result.content.strip()
            # Use regex to extract the first number in the text
            match = re.search(r'\d+', count_text)
            if match:
                count = int(match.group())
                return min(max(count, 1), 10)  # Ensure between 1 and 10
            return 3  # Default if parsing fails
        except Exception as e:
            logger.warning(f"Error determining question count: {e}")
            return 3  # Default if any errors
    
    def generate_qa_pairs(self, heading: str, content: str, source_path: str) -> List[Dict[str, str]]:
        """Generate QA pairs for a section using the LLM."""
        if not content.strip():
            logger.info(f"Skipping empty section: {heading}")
            return []
        
        try:
            # Determine how many questions to generate
            num_questions = self.determine_question_count(heading, content)
            
            # Generate QA pairs
            response = self.llm.invoke(
                QA_GENERATION_PROMPT.format(
                    heading=heading,
                    content=content,
                    num_questions=num_questions
                )
            )
            
            # Extract JSON from response
            response_text = response.content
            
            # Extract JSON if it's wrapped in markdown code blocks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text
                
            # Parse the JSON
            qa_pairs = json.loads(json_str)
            
            # Add source information
            for qa_pair in qa_pairs:
                qa_pair["source"] = source_path
            
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error generating QA pairs for {heading}: {e}")
            return []
    
    def process_markdown_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Process a single markdown file and generate QA pairs for each section."""
        logger.info(f"Processing file: {file_path}")
        all_qa_pairs = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            
            # Get relative path for source tracking
            rel_path = file_path.relative_to(self.docs_dir.parent.parent)
            
            # Extract headings and content
            sections = self.extract_headings_and_content(md_content)
            
            for heading, content, heading_path in sections:
                source_path = f"{rel_path}/{heading_path}"
                logger.info(f"Generating QA pairs for section: {heading_path}")
                
                qa_pairs = self.generate_qa_pairs(heading, content, source_path)
                all_qa_pairs.extend(qa_pairs)
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
        
        return all_qa_pairs
    
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
            writer.writerows(qa_pairs)
            
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
        
        for md_file in md_files:
            qa_pairs = self.process_markdown_file(md_file)
            all_qa_pairs.extend(qa_pairs)
            
        self.save_to_csv(all_qa_pairs)
        logger.info("QA generation complete!")

def main():
    """Main entry point."""
    generator = MarkdownQAGenerator()
    generator.run()

if __name__ == "__main__":
    main()