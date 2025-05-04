import os
import json
import argparse
from collections import Counter
from termcolor import cprint
from tqdm import tqdm

from pup_vector_store import PUPVectorStoreConfig
from dp_rag_engine import DPRAGEngine, DPGenerationConfig
from test_data import medical_dirichlet_documents, medical_dirichlet_full

class Evaluator:
    def __init__(self):
        self.counter = Counter()

    def symptoms(self, disease: str, epsilon: float, success: bool):
        self.counter[json.dumps(("symptoms", "*", epsilon))] += 1
        self.counter[json.dumps(("symptoms", "*", epsilon, "success"))] += int(success)
        self.counter[json.dumps(("symptoms", disease, epsilon))] += 1
        self.counter[json.dumps(("symptoms", disease, epsilon, "success"))] += int(success)

    def dump(self, epsilon_tag=None):
        os.makedirs("results", exist_ok=True)
        fname = f"evaluation_eps_{epsilon_tag}.json" if epsilon_tag else "evaluation.json"
        with open(os.path.join("results", fname), "w", encoding="utf-8") as f:
            json.dump(self.counter, f, indent=2, ensure_ascii=False)

    def load(self, path="results/evaluation.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.counter = Counter(json.load(f))
        except FileNotFoundError:
            cprint("No previous evaluation file found.", "yellow")

class MedicalRAGTests:
    def __init__(self, epsilon=0.5):
        self.docs = medical_dirichlet_documents()
        self.dre = DPRAGEngine(
            pup_vector_store_config=PUPVectorStoreConfig(top_p=0.02, epsilon=epsilon),
            dp_generation_config=DPGenerationConfig(
                temperature=1.0,
                max_new_tokens=70,
                alpha=1.0,
                omega=0.01,
                epsilon=epsilon,
            ),
        )
        for doc in self.docs:
            self.dre.add(doc)

    def test_symptoms(self, n_tests=500):
        evaluator = Evaluator()
        os.makedirs("out", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        pbar = tqdm(total=n_tests, desc="Testing", ncols=80)

        for i, data in enumerate(medical_dirichlet_full()):
            if i >= n_tests:
                break
            eps_before = round(self.dre.privacy_loss_distribution.get_epsilon_for_delta(0.001), 3)
            question = f"I am experiencing the following symptoms: {', '.join(data['symptom'])}. What is my disease?"
            answer = self.dre.dp_chat(question)
            disease = data["disease"]
            success = disease.lower() in answer.lower()
            evaluator.symptoms(disease, eps_before, success)
            total_so_far = evaluator.counter[json.dumps(("symptoms", "*", eps_before))]
            cprint(f"[ε={eps_before:0.3f}] #{total_so_far:3d}", "yellow", end=" ")
            cprint(question, "white")
            cprint(disease, "grey")
            cprint(answer, "green" if success else "red")
            pbar.update(1)
        pbar.close()

        final_eps = round(self.dre.privacy_loss_distribution.get_epsilon_for_delta(0.001), 3)
        eval_path = f"results/evaluation_eps_{final_eps}.json"
        out_path = f"out/test_symptoms_eps_{final_eps}.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(self.counter_to_dict(evaluator.counter), f, indent=2, ensure_ascii=False)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.counter_to_dict(evaluator.counter), f, indent=2, ensure_ascii=False)
        cprint(f"Saved evaluation to {eval_path} and {out_path}", "cyan")

    @staticmethod
    def counter_to_dict(counter):
        return dict(counter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=0.5, help="Initial ε value")
    parser.add_argument("--n_tests", type=int, default=500, help="Number of test cases")
    args = parser.parse_args()
    cprint(f"Testing for epsilon={args.epsilon}\n", "cyan")
    mrt = MedicalRAGTests(epsilon=args.epsilon)
    mrt.test_symptoms(n_tests=args.n_tests)