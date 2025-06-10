import os
import re
import csv
from pathlib import Path
from typing import List, Tuple

from langchain.llms import OpenAI  # Swap this with Gemini, Claude, etc. as needed
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Directory containing the markdown files
dir_path = Path("c:/Users/Ravindu/Documents/My Projects/new-choreo-qa-gen/choreo-docs/developer-docs/docs/choreo-concepts")
output_csv = "choreo_concepts_qa.csv"

# Regex to match markdown headings
HEADING_RE = re.compile(r"^(#{1,6})\\s+(.+)", re.MULTILINE)

def extract_headings_and_content(md_text: str) -> List[Tuple[str, str]]:
    """
    Returns a list of (heading_path, content) tuples.
    heading_path: e.g. 'build', 'build/repeatable builds'
    content: the text under that heading (until next heading of same or higher level)
    """
    matches = list(HEADING_RE.finditer(md_text))
    results = []
    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(md_text)
        content = md_text[start:end].strip()
        # Build heading path
        parent_titles = []
        for j in range(i-1, -1, -1):
            prev_level = len(matches[j].group(1))
            if prev_level < level:
                parent_titles.insert(0, matches[j].group(2).strip())
                level = prev_level
        heading_path = "/".join(parent_titles + [title]).lower().replace(" ", "-")
        results.append((heading_path, content))
    return results

def generate_qa_pairs(llm, heading: str, content: str, max_questions: int = 10) -> List[Tuple[str, str]]:
    """
    Uses the LLM to generate question-answer pairs for the given heading and content.
    Returns a list of (question, answer) tuples.
    """
    prompt = PromptTemplate(
        input_variables=["heading", "content", "max_questions"],
        template=(
            "Given the following section from documentation under the heading '{heading}':\n"
            """\n{content}\n"""\n"
            "Generate as many question and answer pairs as needed to cover the entire content. "
            "Each question should focus on a specific aspect of the content, and the set of questions should together cover all information in the section. "
            "Return the result as a numbered list in the format: Q: <question>\nA: <answer>\n. Limit to {max_questions} questions if possible."
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"heading": heading, "content": content, "max_questions": max_questions})
    # Parse response into (question, answer) pairs
    qa_pairs = []
    for qa in re.split(r"\n\d+\. ", response):
        if not qa.strip():
            continue
        q_match = re.search(r"Q:\s*(.+?)\nA:\s*(.+)", qa, re.DOTALL)
        if q_match:
            question = q_match.group(1).strip()
            answer = q_match.group(2).strip()
            qa_pairs.append((question, answer))
    return qa_pairs

def main():
    llm = OpenAI(temperature=0.2)  # Swap with Gemini, Claude, etc. as needed
    rows = []
    for md_file in dir_path.glob("*.md"):
        with open(md_file, encoding="utf-8") as f:
            md_text = f.read()
        headings = extract_headings_and_content(md_text)
        for heading_path, content in headings:
            if not content.strip():
                continue
            qa_pairs = generate_qa_pairs(llm, heading_path, content)
            for q, a in qa_pairs:
                rows.append({
                    "question": q,
                    "answer": a,
                    "source_section": f"{md_file.as_posix()}/{heading_path}"
                })
    # Write to CSV
    with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["question", "answer", "source_section"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved {len(rows)} question-answer pairs to {output_csv}")

if __name__ == "__main__":
    main()
