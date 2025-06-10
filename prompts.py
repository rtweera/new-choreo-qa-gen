"""
Prompts for QA generation from markdown content.
"""
from langchain.prompts import ChatPromptTemplate

# Prompt for generating question-answer pairs from an entire file
WHOLE_FILE_QA_GENERATION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert knowledge extractor. Your task is to analyze an entire documentation file and generate comprehensive question-answer pairs that cover all the important topics and content.

# FILE CONTENT
{content}

# INSTRUCTIONS
1. First, identify the main topics/sections in this documentation file.
2. For each topic/section, generate multiple question-answer pairs that thoroughly cover all the key information.
3. Ensure your questions and answers collectively cover ALL important information in the document.
4. Questions should be clear, specific, and directly answerable from the content.
5. Answers should be comprehensive, accurate, and directly based on the provided content.
6. Do not introduce information that isn't present in the content.
7. Generate approximately {num_questions} question-answer pairs in total, distributed appropriately across topics.

# OUTPUT FORMAT
Return your response in the following JSON format:
```json
[
  {
    "question": "Question 1?",
    "answer": "Answer to question 1.",
    "topic": "Topic/section this QA relates to"
  },
  {
    "question": "Question 2?", 
    "answer": "Answer to question 2.",
    "topic": "Topic/section this QA relates to"
  }
]
```

Generate question-answer pairs that comprehensively cover the entire document:
""")

# Prompt for generating question-answer pairs from a section
QA_GENERATION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert knowledge extractor. Your task is to generate comprehensive question-answer pairs 
from the provided documentation section.

# SECTION HEADING
{heading}

# SECTION CONTENT
{content}

# INSTRUCTIONS
1. Generate {num_questions} question-answer pairs that thoroughly cover the content in this section.
2. Ensure the questions explore different aspects and collectively cover all information in the section.
3. Questions should be clear, specific, and directly answerable from the content.
4. Answers should be comprehensive, accurate, and directly based on the provided content.
5. Do not introduce information that isn't present in the content.

# OUTPUT FORMAT
Return your response in the following JSON format:
```json
[
  {
    "question": "Question 1?",
    "answer": "Answer to question 1."
  },
  {
    "question": "Question 2?",
    "answer": "Answer to question 2."
  }
]
```

Generate {num_questions} question-answer pairs:
""")

# Prompt to determine how many questions to generate for a section
QUESTION_COUNT_PROMPT = ChatPromptTemplate.from_template("""
Given the following section from a document, determine how many question-answer pairs would be needed 
to thoroughly cover all the information. Consider:

1. The length of the content
2. The number of distinct concepts or points
3. The complexity of the information

# SECTION HEADING
{heading}

# SECTION CONTENT
{content}

# INSTRUCTIONS
Analyze the content and determine the optimal number of questions needed to cover all information.
Return only a single number between 1 and 10.

Number of questions:
""")
