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
  {{
    "question": "Question 1?",
    "answer": "Answer to question 1.",
    "topic": "Topic/section this QA relates to"
  }}
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


# Prompt for generating question-answer pairs from an entire  implicit number of questions
WHOLE_FILE_QA_GENERATION_PROMPT_IMPLICIT_N_QUESTIONS = ChatPromptTemplate.from_template("""
You are an expert knowledge extractor. Your task is to analyze an entire documentation file and 
generate comprehensive questions that cover all the content given to you.

# FILE CONTENT
{content}

# INSTRUCTIONS
1. First, identify the main topics/sections in this documentation file. Use markdown headings to identify the topics.
2. For each heading, generate as many as questions you can that thoroughly cover all the information.
3. Ensure your questions collectively cover ALL the information in the document.
4. Questions should be clear, specific, and directly answerable from the content.
5. Do not introduce information that isn't present in the content.
6. You questions should cover the entire document content, so there shouldn't be any piece of content that is not covered by your questions.
7. With those questions only, you should get the 100%% of the domain knowledge of the document. 
8. For each question, provide a comprehensive answer that directly addresses the question based on the content.
9. You are free to copy the content as much as you like, but align it with the question.
10. Extract the topic that the question is related to. Use the markdown headings to identify the topic.
11. Always use the markdown headings to identify the topic of the question.                                                                                                                                                                                                                                                                      

# OUTPUT FORMAT
Return your response in the following JSON format:
```json
[
  {{
    "question": "Question 1?",
    "answer": "Answer to question 1 in a VERY COMPREHENSIVE way.",
    "topic": "MARKDOWN HEADING the QA relates to "
  }}
]
```
""")

# Prompt for generating question-answer pairs from an entire  implicit number of questions
WHOLE_FILE_QA_GENERATION_PROMPT_IMPLICIT_N_QUESTIONS_V2 = ChatPromptTemplate.from_template("""
You are an expert knowledge extractor. Your task is to analyze an entire documentation file and
generate comprehensive questions that cover all the content given to you.

# FILE CONTENT
{content}

# INSTRUCTIONS
1. First, identify the main topics/sections in this documentation file. Use markdown headings to identify the topics.
2. For each heading, generate as many as questions you can that thoroughly cover all the information.
3. When generating questions, ensure that you can understand the what the question is asking without needing to refer back to the content
   for specific context. 
    EX:
    i. - Bad Question: "What is the content of this section?"
       - Good Question: "What is the purpose of the section on 'User Authentication'?"                 
    ii. - Bad Question: "How to search for service?"
        - Good Question: "How to search for a service in the `Choreo marketplace` interface?"                                                                                                                                                                                                                                                  
4. Ensure your questions collectively cover ALL the information in the document.
5. Questions should be clear, specific, and directly answerable from the content.
6. Do not introduce information that isn't present in the content.
7. You questions should cover the entire document content, so there shouldn't be any piece of content that is NOT covered by your questions.
8. With those questions only, you should get the 100%% of the domain knowledge of the document. 
9. For each question, provide a comprehensive answer that directly addresses the question based on the content.
10. You are free to copy the content as much as you like, but it should be RELEVANT and align it with the question.
11. Extract the topic that the question is related to.
12. Always use the markdown headings to identify the topic of the question.                                                                                                                                                                                                                                                                      

# OUTPUT FORMAT
Return your response in the following JSON format:
```json
[
  {{
    "question": "Question 1 WITH some CONTEXT?",
    "answer": "Answer to question 1 in a VERY COMPREHENSIVE way.",
    "topic": "MARKDOWN HEADING the QA relates to "
  }}
]
```
""")
