import argparse
import json
import os
import time
import math
from datetime import datetime
from typing import List, Dict, Any


import ray
from vllm import LLM, SamplingParams

from evaluation.datasets import get_dataset
from evaluation.evaluators import get_evaluator


@ray.remote(num_gpus=1)
class VllmWorker:
    
    def __init__(self, model_path: str, trust_remote_code: bool = True, **kwargs):

        gpu_id = ray.get_gpu_ids()[0] if ray.get_gpu_ids() else "N/A"
        print(f"[Worker PID={os.getpid()}, GPU={gpu_id}] Initializing vLLM engine...")
        
        self.llm = LLM(
            model=model_path, 
            trust_remote_code=trust_remote_code, 
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.9, 
            **kwargs
        )
        print(f"[Worker PID={os.getpid()}, GPU={gpu_id}] vLLM engine initialized.")

    def warmup(self, warmup_prompt: str = "Hello, world!"):
        print(f"[Worker PID={os.getpid()}] Running warmup inference...")
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=10
        )
        _ = self.llm.generate([warmup_prompt], sampling_params)
        print(f"[Worker PID={os.getpid()}] Warmup complete.")

    def generate_batch(self, batch_data: List[Dict[str, Any]], sampling_params: SamplingParams) -> List[Dict[str, Any]]:
        prompts = [item['prompt'] for item in batch_data]
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        for i, item in enumerate(batch_data):
            generated_text = outputs[i].outputs[0].text
            item['generation'] = generated_text
            
        return batch_data


def create_workers_sequentially(model_path: str, num_gpus: int, trust_remote_code: bool, vllm_kwargs: dict) -> List:
    workers = []
    
    for i in range(num_gpus):
        print(f"\n[INFO] Creating worker {i+1}/{num_gpus}...")
        worker = VllmWorker.remote(model_path, trust_remote_code=trust_remote_code, **vllm_kwargs)
        
        if i == 0:
            print(f"[INFO] Warming up first worker (this may take a while)...")
            ray.get(worker.warmup.remote())
            print(f"[INFO] First worker warmed up. Subsequent workers will be faster.")
        else:
            time.sleep(2)
        
        workers.append(worker)
        print(f"[INFO] Worker {i+1}/{num_gpus} created successfully.")
    
    return workers


def main(args):

    print("[INFO] Initializing Ray...")
    ray.init(num_gpus=args.num_gpus)
    print(f"[INFO] Ray initialized. Cluster resources: {ray.available_resources()}")

    print(f"[INFO] Loading dataset '{args.dataset_name}'...")
    try:
        DatasetClass = get_dataset(args.dataset_name)
        dataset = DatasetClass()
        all_data = dataset.load()
        print(f"[INFO] Loaded {len(all_data)} samples from dataset '{dataset.name}'.")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        ray.shutdown()
        return

    recommended_config = dataset.get_recommended_config()
    if recommended_config:

        if 'per_gpu_batch_size' in recommended_config:
            print(f"[INFO] Mapping dataset config 'per_gpu_batch_size' to 'per_worker_batch_size'.")
            recommended_config['per_worker_batch_size'] = recommended_config.pop('per_gpu_batch_size')

        print(f"[INFO] Applying recommended config from dataset '{dataset.name}':")
        for key, value in recommended_config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                print(f"  - Setting '{key}' to {value}")
                setattr(args, key, value)

    if args.per_worker_batch_size is None:
        args.per_worker_batch_size = 8
        print(f"[INFO] Using default 'per_worker_batch_size': {args.per_worker_batch_size}")
    if args.max_new_tokens is None:
        args.max_new_tokens = 512
        print(f"[INFO] Using default 'max_new_tokens': {args.max_new_tokens}")
    if args.temperature is None:
        args.temperature = 0.0
        print(f"[INFO] Using default 'temperature': {args.temperature}")
    if args.top_p is None:
        args.top_p = 1.0
        print(f"[INFO] Using default 'top_p': {args.top_p}")

    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    print(f"[INFO] Using final SamplingParams: {sampling_params}")

    print(f"\n[INFO] Creating {args.num_gpus} VllmWorker actors SEQUENTIALLY to avoid cache conflicts...")
    vllm_kwargs = {'dtype': 'auto'} 
    
    workers = create_workers_sequentially(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        trust_remote_code=args.trust_remote_code,
        vllm_kwargs=vllm_kwargs
    )
    
    print(f"\n[INFO] All {args.num_gpus} workers created and ready.")

    print("[INFO] Distributing tasks to workers...")
    start_time = time.time()
    
    chunks = [
        all_data[i:i + args.per_worker_batch_size] 
        for i in range(0, len(all_data), args.per_worker_batch_size)
    ]
    
    tasks = []
    for i, chunk in enumerate(chunks):
        worker_id = i % args.num_gpus
        tasks.append(workers[worker_id].generate_batch.remote(chunk, sampling_params))

    results_nested = ray.get(tasks)
    
    total_time = time.time() - start_time
    print(f"\n[INFO] Generation finished in {total_time:.2f} seconds.")
    ray.shutdown()

    print("[INFO] Aggregating results...")

    final_results = [item for sublist in results_nested for item in sublist]

    final_results.sort(key=lambda x: x['task_id'])
    
    if len(final_results) != len(all_data):
        print(f"[WARNING] Number of results ({len(final_results)}) does not match number of samples ({len(all_data)}).")


    print(f"[INFO] Running evaluator '{dataset.evaluator_name}'...")
    try:
        evaluator = get_evaluator(dataset.evaluator_name)
        metrics = evaluator.evaluate(final_results)
    except Exception as e:
        print(f"[ERROR] Failed to run evaluation: {e}")
        metrics = {"error": str(e)}


    print("[INFO] Saving results and metrics...")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir_name = f"{timestamp}"
    output_dir = os.path.join(args.output_dir, args.dataset_name, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    

    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    with open(predictions_path, 'w', encoding='utf-8') as f:
        for item in final_results:
            f.write(json.dumps(item) + '\n')
    print(f"  - Predictions saved to: {predictions_path}")
    

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print(f"  - Metrics saved to: {metrics_path}")


    print("\n--- Evaluation Metrics ---")


    metrics_to_print = metrics.copy()


    if "detailed_predictions" in metrics_to_print and len(metrics_to_print["detailed_predictions"]) > 10:
        total_predictions = len(metrics_to_print["detailed_predictions"])
        metrics_to_print["detailed_predictions"] = metrics_to_print["detailed_predictions"][:10]
        metrics_to_print["note"] = f"Showing first 10 of {total_predictions} detailed predictions"

    print(json.dumps(metrics_to_print, indent=4))
    print("--------------------------\n")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation using vLLM and existing dataset/evaluator components.")
    
    parser.add_argument("--model-path", type=str, required=True, help="Path to the Hugging Face model directory.")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs to use for data parallelism.")
    parser.add_argument("--trust-remote-code", action='store_true', help="Trust remote code when loading models from Hugging Face Hub.")

    parser.add_argument("--dataset-name", type=str, required=True, help="Name of the dataset to evaluate (e.g., 'math', 'gsm8k').")
    parser.add_argument("--output-dir", type=str, required=True, help="Root directory to save evaluation results.")
    
    parser.add_argument("--per-worker-batch-size", type=int, default=None, help="Number of prompts to send to each worker in one go. Overridden by dataset config if not set.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Maximum number of new tokens to generate. Overridden by dataset config if not set.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature. Overridden by dataset config if not set.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (nucleus) sampling. Overridden by dataset config if not set.")

    parsed_args = parser.parse_args()
    main(parsed_args)
    
"""
python -m evaluation.ar_eval_main \
    --model-path /mnt/cephfs_nj/aiweiliu/huggingface_models/Qwen3-8B-Instruct \
    --dataset-name "gsm8k" \
    --output-dir "output/Qwen3-8B-Instruct/gsm8k/" \
    --num-gpus 8 \
    --trust-remote-code
"""