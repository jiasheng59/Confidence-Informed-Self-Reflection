import getpass
import os
import time
import math
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
from typing import Any, Dict, Iterable
from huggingface_hub import login
from datasets import load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

system_prompt = """
Let's take it step by step. You are student sitting in an reading comprehension exam. Given a context and question, 
give the answer in a short span of text, and state your explanation in [Reasoning] section. Some of the questions may be unanswerable, in that case, 
give an answer by [Answer]: Unanswerable



[Context]: ...
[Question]: When were the Normans in Normandy

Answer in the following format:

[Reasoning]: [Provide your explanation here if any]
[Answer]: [Answer here]

======
You will receive feedback from teacher in subsequent conversation, based on the feedback, 
reflect on your previous answer and answer again in the following format:

[Reflection]: [Provide your reflection]
[Answer]: [Answer here]
"""

question_format = """
[Context]: {context}
[Question]: {question}
[Reasoning]: [Your reasoning here]
[Answer]: [Your short answer here]
"""


def generate(model, tokenizer, messages, user_question):
    """
    :param model: model
    :param tokenizer: tokenizer
    :param messages: conversation
    :param user_question: user_question
    :return: outputs, input_ids, messages with user_question
    """
    messages.append({"role": "user", "content": user_question})
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    start_time = time.time()
    print("--- generate begins ---")

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        output_scores=True,
        output_logits=True,
        output_attentions=True,
        return_dict_in_generate=True
    )
    print("--- generate ends, time taken: %s seconds ---" % (time.time() - start_time))
    return outputs, input_ids, messages


def predict(context, question):
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    formatted_question = question_format.format(context=context, question=question)
    outputs, input_ids, output_messages = generate(model, tokenizer, messages, formatted_question)
    generated_ids = outputs['sequences']
    generated_text = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    output_messages.append({"role": "assistant", "content": generated_text})
    print("LLM response:", generated_text)
    return generated_text, outputs, output_messages


def extract_answer(text):
    # Regular expression pattern to match everything after "[Answer]: "
    pattern = r"\[Answer\]:\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        answer_text = match.group(1)
        return answer_text
    else:
        print("No match found")
        return ""


def extract_reasoning(text):
    # Regular expression pattern to match everything after "[Reasoning]: "
    pattern = r"\[Reasoning\]:\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        reasoning = match.group(1)
        return reasoning
    else:
        print("No match found")
        return ""


def extract_reflection(text):
    # Regular expression pattern to match everything after "[Reflection]: "
    pattern = r"\[Reflection\]:\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        reflection = match.group(1)
        return reflection
    else:
        print("No match found")
        return ""


# Logit-based approaches
def compute_sequence_likelihood(input_ids, outputs):
    sequence_likelihood = 0

    for i, logits in enumerate(outputs['logits']):
        # Apply softmax over the last dimension (vocab size) to get probabilities
        probabilities = F.softmax(logits, dim=-1)

        # Get the index of the generated token for this step
        generated_token_id = outputs['sequences'][0][input_ids.shape[-1] + i].item()

        # Get the probability of the generated token
        generated_token_prob = probabilities[0, generated_token_id].item()

        # Print the token ID and its probability
        # print(f"Token {i + 1}: ID {generated_token_id}, Probability: {generated_token_prob}")

        # sequence_likelihood *= generated_token_prob
        sequence_likelihood += math.log(generated_token_prob, 10)

    sequence_likelihood = math.pow(10, sequence_likelihood)
    # print("sequence likelihood:", sequence_likelihood)
    return sequence_likelihood


def mean_logit_probs(input_ids, outputs):
    sum_probs = 0
    count = 0
    for i, logits in enumerate(outputs['logits']):
        probabilities = F.softmax(logits, dim=-1)
        generated_token_id = outputs['sequences'][0][input_ids.shape[-1] + i].item()
        generated_token_prob = probabilities[0, generated_token_id].item()

        sum_probs += generated_token_prob
        count += 1
    if count == 0:
        return 0
    return sum_probs / count


# P(True)
grader_prompt = """
You are a grader grading a reading comprehension exam. Students are provided context and question, 
and they give an answer together with their reasoning to derive the answer. Some of the questions may be unanswerable, 
in that case, the expected [Answer] section is empty.

Answer either True or False given the student's answer.

Example 1:
[Context]: Math
[Question]: 1 + 2 * 3
[Reasoning]: Multiplication is performed before addition
[Answer]: 7

True

Example 2:
[Context]: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.
[Question]: In what country is Normandy located?
[Reasoning]: Random guess
[Answer]: China

False
"""

input_to_grader = """
[Context]: {context}
[Question]: {question}
[Reasoning]: {reasoning}
[Answer]: {answer}

Is the proposed answer: True / False
The proposed answer is:
"""


def elicit_logit_confidence(context, question, reasoning, answer):
    grader_messages = [
        {"role": "system", "content": grader_prompt}
    ]
    formatted_question = input_to_grader.format(context=context, question=question, reasoning=reasoning, answer=answer)
    outputs, input_ids, output_messages = generate(model, tokenizer, grader_messages, formatted_question)
    generated_ids = outputs['sequences']
    generated_text = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    output_messages.append({"role": "assistant", "content": generated_text})
    return generated_text, outputs, input_ids, output_messages


# def compute_p_true(grader_text, grader_input_ids, grader_outputs):
#   if 'true' in grader_text.lower():
#     return compute_sequence_likelihood(grader_input_ids, grader_outputs)
#   elif 'false' in grader_text.lower():
#     return 1 - compute_sequence_likelihood(grader_input_ids, grader_outputs)
#   else:
#     print("p_true is missing")
#     return 0.5


def compute_p_true(grader_input_ids, grader_outputs):
    target_word_probability = None

    # Tokenize the target word
    true_token_id = tokenizer.encode("True", add_special_tokens=False)[0]
    false_token_id = tokenizer.encode("False", add_special_tokens=False)[0]

    for i, logits in enumerate(grader_outputs['logits']):
        # Apply softmax over the last dimension (vocab size) to get probabilities
        probabilities = F.softmax(logits, dim=-1)

        # Get the index of the generated token for this step
        generated_token_id = grader_outputs['sequences'][0][grader_input_ids.shape[-1] + i].item()

        # Check if this token matches the target word
        if generated_token_id == true_token_id:
            target_word_probability = probabilities[0, generated_token_id].item()  # Extract probability for the target word
            print(f"P(True) = {target_word_probability}", target_word_probability)
            return target_word_probability
        if generated_token_id == false_token_id:
            target_word_probability = probabilities[0, generated_token_id].item()
            print(f"P(False) = {target_word_probability}", target_word_probability)
            return 1 - target_word_probability
    print("target word not found")
    return 0.5


verifier_prompt = """
You are a teacher reading student's attempt in a reading comprehension exam. Students are provided context and question, 
and they give an answer together with their reasoning to derive the answer. Some of the questions may be unanswerable, 
in that case, the expected [Answer] section is empty. Student's confidence is also attached with their answer.

Give your feedback to the student's answer below. 
"""

input_to_verifier = """
[Context]: {context}
[Question]: {question}
[Reasoning]: {reasoning}
[Answer]: {answer}
[Confidence]: {confidence}
"""


def feedback(context, question, reasoning, answer, confidence):
    # pass reasoning, answer and confidence to verifier to generate feedback
    verifier_messages = [
        {"role": "system", "content": verifier_prompt}
    ]
    formatted_question = input_to_verifier.format(
        context=context, question=question, reasoning=reasoning, answer=answer, confidence=confidence)
    outputs, input_ids, output_messages = generate(model, tokenizer, verifier_messages, formatted_question)
    generated_ids = outputs['sequences']
    feedback = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print("verifier feedback:", feedback)
    return feedback


reflection_prompt = """
Here is the feedback you receive:
{feedback}


Based on the feedback, reflect on your previous answer and answer again in the following format:

[Reflection]: [Provide your reflection]
[Answer]: [Answer here]

If you are uncertain about your answer, answer by [Answer]: Unanswerable
"""


def reflect(messages, feedback):
    formatted_reflection_prompt = reflection_prompt.format(feedback=feedback)
    outputs, input_ids, output_messages = generate(model, tokenizer, messages, formatted_reflection_prompt)
    generated_ids = outputs['sequences']
    generated_text = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print("Post reflection:", generated_text)
    return generated_text, outputs, output_messages


post_reflection_grader_prompt = """
Here is another attempt on the question after the student reflected on the feedback he received.

[Reflection]: {reflection}
[Answer]: {answer}

Is the proposed answer: True / False
"""


def elicit_logit_confidence_post_reflection(grader_messages, reflection, answer):
    formatted_question = post_reflection_grader_prompt.format(reflection=reflection, answer=answer)
    outputs, input_ids, output_messages = generate(model, tokenizer, grader_messages, formatted_question)
    generated_ids = outputs['sequences']
    generated_text = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    output_messages.append({"role": "assistant", "content": generated_text})
    return generated_text, outputs, input_ids, output_messages


context = """The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."""
question = """In what country is Normandy located?"""

generated_text, outputs, output_messages = predict(context, question)

reasoning = extract_reasoning(generated_text)
answer = extract_answer(generated_text)


print("generated_text:")
print(generated_text)
print("answer:")
print(answer)


grader_text, grader_outputs, grader_input_ids, grader_messages = elicit_logit_confidence(context, question, reasoning, answer)
p_true = compute_p_true(grader_input_ids, grader_outputs)


print(grader_text)
print(p_true)


# Verbalised Confidence
verbal_system_prompt = """
Let's take it step by step. You are student sitting in an reading comprehension exam. Given a context and question, 
give the answer in a short span of text, and state your explanation in [Reasoning] section, give your confidence in a integer from 0 - 100. Some of the questions may be unanswerable, in that case, 
give an answer by [Answer]: Unanswerable



[Context]: ...
[Question]: When were the Normans in Normandy

Answer in the following format:

[Reasoning]: [Provide your explanation here if any]
[Answer]: [Answer here]
[Confidence]: [Your confidence]

======
You will receive feedback from teacher in subsequent conversation, based on the feedback, 
reflect on your previous answer and answer again in the following format:

[Reflection]: [Provide your reflection]
[Answer]: [Answer here]
[Confidence]: [Your confidence, between 0 - 100]
"""

verbal_question_format = """
[Context]: {context}
[Question]: {question}
[Reasoning]: [Your reasoning here]
[Answer]: [Your short answer here]
[Confidence]: [Your confidence]
"""


def verbal_predict(context, question):
    messages = [
        {"role": "system", "content": verbal_system_prompt}
    ]
    formatted_question = verbal_question_format.format(context=context, question=question)
    outputs, input_ids, output_messages = generate(model, tokenizer, messages, formatted_question)
    generated_ids = outputs['sequences']
    generated_text = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    output_messages.append({"role": "assistant", "content": generated_text})
    return generated_text, outputs, input_ids, output_messages


def extract_verbal_confidence(text):
    # Regular expression pattern to match everything after "[Confidence]: "
    pattern = r"\[Confidence\]:\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        confidence_text = match.group(1)
        return confidence_text
    else:
        print("No match found")
        return ""


verbal_reflection_prompt = """
Here is the feedback you receive:
{feedback}


Based on the feedback, reflect on your previous answer and answer again in the following format:

[Reflection]: [Provide your reflection]
[Answer]: [Answer here]
[Confidence]: [Your confidence, 0 - 100]

If you are uncertain about your answer, answer by [Answer]: Unanswerable
"""


def verbal_reflect(messages, feedback):
    formatted_reflection_prompt = verbal_reflection_prompt.format(feedback=feedback)
    outputs, input_ids, output_messages = generate(model, tokenizer, messages, formatted_reflection_prompt)
    generated_ids = outputs['sequences']
    generated_text = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print("Post reflection:", generated_text)
    return generated_text, outputs, output_messages

