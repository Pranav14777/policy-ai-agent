import json
from pathlib import Path
from agent import ask, IDK_PHRASE


BASE_DIR = Path(__file__).parent
TEST_DATA_PATH = BASE_DIR / "test_data.json"


def load_test_data():
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate():
    tests = load_test_data()

    total = len(tests)
    correct_answers = 0
    correct_idk = 0
    wrong_answers = 0
    hallucinations = 0

    print(f"Running evaluation on {total} test cases...\n")

    for idx, item in enumerate(tests, start=1):
        question = item["question"]
        expected_keywords = item["expected_keywords"]
        should_answer = item["should_answer"]

        result = ask(question)
        answer = (result.get("answer") or "").strip()
        sources = result.get("sources", [])

        print(f"Test {idx}")
        print(f"Question: {question}")
        print(f"Answer:   {answer}")
        print(f"Sources:  {sources}")

        answer_lower = answer.lower()

        if should_answer:
            # Expected to provide a real answer
            keyword_match = any(kw.lower() in answer_lower for kw in expected_keywords)

            if keyword_match and IDK_PHRASE.lower() not in answer_lower:
                correct_answers += 1
                print("Result: Correct answer")
            elif IDK_PHRASE.lower() in answer_lower:
                wrong_answers += 1
                print("Result: Incorrectly responded with 'I don't know'")
            else:
                wrong_answers += 1
                print("Result: Wrong answer")
        else:
            # Expected to say "I don't know"
            if IDK_PHRASE.lower() in answer_lower:
                correct_idk += 1
                print("Result: Correct 'I don't know'")
            else:
                hallucinations += 1
                print("Result: Hallucination (should not have answered)")

        print()

    print("===== SUMMARY =====")
    print(f"Total questions:                {total}")
    print(f"Correct normal answers:         {correct_answers}")
    print(f"Correct 'I don't know' answers: {correct_idk}")
    print(f"Wrong answers:                  {wrong_answers}")
    print(f"Hallucinations:                 {hallucinations}")


if __name__ == "__main__":
    evaluate()

