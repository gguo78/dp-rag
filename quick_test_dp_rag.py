# quick_test_dp_rag.py
import os
import torch
from collections import Counter
import json
from termcolor import cprint
from test_data import medical_dirichlet_documents, medical_dirichlet_full
from dp_rag_engine import DPRAGEngine, DPGenerationConfig, PUPVectorStoreConfig

class QuickEvaluator:
    def __init__(self):
        self.counter = Counter()
        # global accuracy counter
        self.total = 0
        self.successes = 0
        
    def symptoms(self, disease: str, epsilon: float, success: bool, question: str = "", answer: str = ""):
        self.total += 1
        self.successes += 1 if success else 0
        
        self.counter[json.dumps(None)] += 1
        self.counter[json.dumps(("*", "*"))] += 1
        self.counter[json.dumps(("symptoms", "*"))] += 1
        self.counter[json.dumps(("symptoms", "*", epsilon, "success"))] += 1 if success else 0
        self.counter[json.dumps(("symptoms", disease,))] += 1
        self.counter[json.dumps(("symptoms", disease, epsilon, "success"))] += 1 if success else 0

        os.makedirs('quick_test_results', exist_ok=True)
        with open("quick_test_results/qa_log.jsonl", "a") as f:
            f.write(json.dumps({
                "question": question,
                "answer": answer,
                "expected_disease": disease,
                "success": success,
                "epsilon": epsilon
            }) + "\n")
    
    def dump(self):
        os.makedirs('quick_test_results', exist_ok=True)
        with open(f"quick_test_results/quick_evaluation.json", 'w') as f:
            json.dump(self.counter, f, indent=2)


class QuickMedicalRAGTest:
    def __init__(self, num_docs=50):
        # Print device info
        cprint(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}", "yellow")
        
        # Use data from Huggingface
        self.docs = medical_dirichlet_documents()
        # Setup the DP RAG Engine
        self.dre = DPRAGEngine(
            pup_vector_store_config = PUPVectorStoreConfig(
                top_p = 0.02,
                epsilon=0.5,
            ),
            dp_generation_config=DPGenerationConfig(
                temperature=1.0,
                max_new_tokens=70,
                alpha = 1.0,
                omega = 0.01,
                epsilon = 5.0,
                model_id="facebook/opt-125m",
            ),
        )
        # Add all docs to RAG engine (but only a sample for speed)
        # Adding just 20 documents for quick testing
        cprint(f"Adding {min(num_docs, len(self.docs))} documents to the RAG engine...", "blue")
        for doc in self.docs[:num_docs]:
            self.dre.add(doc)

    def test_single_example(self):
        evaluator = QuickEvaluator()
        
        # Get just the first example from the dataset
        data = next(iter(medical_dirichlet_full()))
        
        cprint("Running a single test case...", "blue")
        cprint("-" * 50, "blue")
        
        # Format the question from symptoms
        question = f"I am experiencing the following symptoms: {', '.join(data['symptom'])}. What is my disease?"
        
        cprint(f"QUESTION: {question}", "white")
        cprint(f"EXPECTED DISEASE: {data['disease']}", "yellow")
        
        try:
            # Get the model response
            cprint("Generating answer (this may take a moment)...", "blue")
            answer = self.dre.dp_chat(question)
            
            # Evaluate success
            success = data['disease'] in answer
            epsilon = round(self.dre.privacy_loss_distribution.get_epsilon_for_delta(0.001), 1)
            
            # Print results
            cprint(f"ANSWER: {answer}", "green" if success else "red")
            cprint(f"SUCCESS: {success}", "green" if success else "red")
            cprint(f"PRIVACY BUDGET (EPSILON): {epsilon}", "cyan")
            
            # Save results
            evaluator.symptoms(data['disease'], epsilon, success, question=question, answer=answer)
            evaluator.dump()
            
            if evaluator.total:
                cprint(f"ACCURACY: {evaluator.successes}/{evaluator.total} = {evaluator.successes / evaluator.total:.2%}", "cyan")
            
            cprint("-" * 50, "blue")
            cprint("Results saved to quick_test_results/quick_evaluation.json", "blue")
        
        except Exception as e:
            cprint(f"ERROR: {str(e)}", "red")
            cprint(f"ERROR TYPE: {type(e).__name__}", "red")
            import traceback
            cprint(traceback.format_exc(), "red")


if __name__ == "__main__":
    cprint("Starting quick DP-RAG test with a single example", "yellow")
    test = QuickMedicalRAGTest()
    test.test_single_example()
