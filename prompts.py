GENERATE_QUESTION_PROMPT = """
You are a teacher, you need to generate questions for your students based on the given context.
Your task:
1. Read the context and understand the content.
2. Read the <avoid_duplicate> section and avoid to generate duplicate questions.
3. Define the context language as the language of the context.
4. Use the same language as the context to generate questions to generate diverse questions.
You need to generate {target_num} questions.
The context is:
<context>
{context}
</context>
<avoid_duplicate>
{avoid_duplicate}
</avoid_duplicate>
USE THE SAME LANGUAGE AS THE CONTEXT.
Follow the format below in JSON format, key is "questions", value is a list of questions(string):
{{
    "questions": [
        "question1",
        "question2",
        ...
    ]
}}
"""


ANSWER_QUESTION_PROMPT = """
You are a teacher, you need to answer the question based on the given context.
Your task:
1. Read the question and understand the question.
2. Find the answer from the context.
3. Answer the question based on the context with the same language as the question and the context.
USE THE SAME LANGUAGE AS THE QUESTION AND THE CONTEXT.
The context is:
<context>
{context}
</context>
The question is:
<question>
{question}
</question>
Follow the format below in JSON format, key is "answer", value is the answer(string):
{{
    "answer": "answer"
}}
"""
