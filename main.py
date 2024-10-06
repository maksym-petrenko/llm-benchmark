import json
from openai import OpenAI
import numpy as np


def llm(query: str, model: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        messages=[{"role": "system","content": query}], 
        model=model
    )
    return response.choices[0].message.content


def get_embedding(text: str, model: str="text-embedding-3-small") -> list[float]:
    client = OpenAI()
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding 
    

def get_answer(type: str, question: str, model: str) -> str:
    instructions = {
        "choose": "You are given a question that you have to answer with a list of items. Your answer should be an integer, You must not include anything else. Your answer should look like ['item1', 'item2', ...]",
        "number": "You are given a quesiton that you have to answer with a number. You answer must look like n. Do not use any commas in the number. Do not include anything apart from the number. Use the plain number format. Under no circumstances include any symbols apart form digits. You instead of 1 million use 1000000 and so on.",
        "text": "You are given a quesiton that you have to answer precisely. Make sure to be brief, don't write more than 2 sentences."
    }

    return llm(f"{instructions[type]} \n question: \n {question}", model)


def get_intersection(list1: list[str], list2: list[str]) -> int:

    tmp = llm(f"""You have to compare two lists. The main task is to find number of intersections in them, but you actually have to look at the meaning and not the exact string value. 
For example, "POC" and "proof of concept" should count ans the same.
Make sure to cout all intersections of the lists.
Your response must be a json without any additional characters. It should look like 

{{
    "thinking": <put thoughs and reasons here>,
    "answer": <number of intersections>
}}

The lists are:
list1: {list1}
list2: {list2}
""", "gpt-4o-mini")
    
    return int(json.loads(tmp)["answer"])


def cosine(a: list[float], b: list[float]) -> float:
    v1, v2 = np.array(a), np.array(b)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def evaluate(queston_type: str, gen_answer: str, real_answer: str | int) -> float:
    match queston_type:
        case "choose":
            return get_intersection(list(real_answer), gen_answer)/len(list(real_answer)) 
        case "number":
            return np.exp(-(abs(int(gen_answer) - real_answer)/abs(real_answer)))
        case "text":
            return cosine(get_embedding(gen_answer), get_embedding(real_answer))

if __name__ == "__main__":
    with open('questions.json', 'r') as file:
        data = json.load(file)
    
    models = [
        "gpt-4o-mini",
        "gpt-4-turbo",
    ]
    
    for model in models:  # delete info from previous runs
        open(f"benchmarks/{model}.txt", 'w').close()

    for model in models:
        for i, question in enumerate(data["questions"]):
            answer = get_answer(question["type"], question["question"], model)
            with open(f"benchmarks/{model}.txt", "a") as f:
                f.write(f"benchmark: {i + 1}, score: {'{:.4f}'.format(evaluate(question['type'], answer, question['answer']))}, question: {question['question']}, \nanswer: {answer}\nreal answer: {question['answer']}\n")

