import os
import pickle
import argparse
import torch
from datetime import datetime
from datasets import load_dataset
import vllm
from metric import rouge


datasets = {'govreport': {'data': load_dataset('launch/gov_report')['test'], 'metric': 'rouge'},
            'summscreenfd-sum': {'data': load_dataset('learn3r/summ_screen_fd_bp')['test'], 'metric': 'rouge'},
            'qmsum': {'data': load_dataset('pszemraj/qmsum-cleaned')['validation'], 'metric': 'rouge'},}

models = {
    "Open-Orca/LlongOrca-7B-16k": {'max_len': 16384, 'prompt_prefix': '<|im_start|>user\n', 'prompt_suffix':'<|im_end|><|im_start|>assistant\n',},
    "Yukang/LongAlpaca-7B": {'max_len': 32768, 'prompt_prefix': '[INST]\n', 'prompt_suffix':'\n[\INST]',},
    "lmsys/longchat-7b-v1.5-32k": {'max_len': 32768, 'prompt_prefix': 'USER: ', 'prompt_suffix':'ASSISTANT: ',},
    "togethercomputer/Llama-2-7B-32K-Instruct": {'max_len': 32768, 'prompt_prefix': '[INST]\n', 'prompt_suffix':'\n[\INST]',},
    "Ahrefs/LongCamel-7b-32k": {'max_len': 32768, 'prompt_prefix': '', 'prompt_suffix':'[EOT]',},
}

def construct_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_sets", type=str, default=None)

    args = parser.parse_args()
    return args


def print_rank_0(message):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(message)

def get_model(model_dir):
    engine_kwargs = {'tensor_parallel_size': 8}
    llm = vllm.LLM(model=model_dir, max_model_len=models[model_dir]['max_len'], **engine_kwargs)
    return llm



def get_dataset_samples(dataset_name, model_name, prompt_template='pre-instruction'):
    dataset = datasets[dataset_name]['data']
    prefix = models[model_name]['prompt_prefix']
    suffix = models[model_name]['prompt_suffix']
    if prompt_template=='pre-instruction':

        if dataset_name == 'govreport':
            samples = [{'id': f"{dataset_name}_{i}",
                        "prompt": f"{prefix}You are given a report by a government agency. Write a one-page summary of the report.\n\n\nReport:{sample['document']}{suffix}",
                        "answer": sample['summary']} for i, sample in enumerate(dataset)]
        elif dataset_name == 'summscreenfd-sum':
            assert len([sample for sample in dataset if len(sample['output'].split(' Summary: ')) != 2]) == 0
            samples = [{'id': f"{dataset_name}_{i}",
                        "prompt": f"{prefix}You are given a script of a TV episode. Summarize the episode in a paragraph.\n\n\nEpisode Script:{sample['input']}{suffix}",
                        "answer": sample['output'].split(' Summary: ')[-1]} for i, sample in enumerate(dataset)]
        elif dataset_name == 'qmsum':
            samples = []
            for i, sample in enumerate(dataset):
                query = sample['input'].split('\n')[0]
                context = sample['input'].replace(query, '', 1).strip()
                samples.append({'id': f"{dataset_name}_{i}",
                                "prompt": f"{prefix}You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\n\nTranscript:{context}\n\n\nQuery:\n{query}{suffix}",
                                "answer": sample['output']})
    else:
        if dataset_name == 'govreport':
            samples = [{'id': f"{dataset_name}_{i}",
                        "prompt": f"{prefix}{sample['document']}\nSummarize the above text in one paragraph.{suffix}",
                        "answer": sample['summary']} for i, sample in enumerate(dataset)]
        elif dataset_name == 'summscreenfd-sum':
            assert len([sample for sample in dataset if len(sample['output'].split(' Summary: ')) != 2]) == 0
            samples = [{'id': f"{dataset_name}_{i}",
                        "prompt": f"{prefix}{sample['input']}\nSummarize the above text in 3-5 sentences.{suffix}",
                        "answer": sample['output'].split(' Summary: ')[-1]} for i, sample in enumerate(dataset)]
        elif dataset_name == 'qmsum':
            samples = []
            for i, sample in enumerate(dataset):
                query = sample['input'].split('\n')[0]
                context = sample['input'].replace(query, '', 1).strip()
                samples.append({'id': f"{dataset_name}_{i}",
                                "prompt": f"{prefix}{context}\nAnswer the question in 3-5 sentences. {query}{suffix}",
                                "answer": sample['output']})
    return samples


def get_sampling_params(model_dir, dataset_name):
    sampling_params = {"skip_special_tokens": True, "temperature": 0, "top_p": 1, "max_tokens": 512}
    if model_dir == "Ahrefs/LongCamel-7b-32k":
        sampling_params.update({'stop': '[EOT]'})
    if dataset_name == 'govreport':
        sampling_params.update({'max_tokens': 2048})
    return vllm.SamplingParams(**sampling_params)

def get_generations(model, samples, sampling_params):
    outputs = model.generate([f"{sample['prompt']}" for sample in samples], sampling_params)
    outputs = [output.outputs[0].text.strip() for output in outputs]
    for i in range(5):
        print_rank_0(f"{datetime.now().strftime('%H:%M:%S')}, {i}th sample output: {outputs[i]}")
    return outputs


def run_eval_sets(model_dir, output_dir, eval_sets):
    os.makedirs(output_dir, exist_ok=True)
    model = get_model(model_dir)
    print_rank_0(f"{datetime.now().strftime('%H:%M:%S')} loaded model from {model_dir}")

    for dataset_name in eval_sets:
        print_rank_0(f"{datetime.now().strftime('%H:%M:%S')} starting evaluation for model: {model_dir}, dataset: {dataset_name}")
        metric_name = datasets[dataset_name]['metric']
        sampling_params = get_sampling_params(model_dir, dataset_name)
        samples = get_dataset_samples(dataset_name, model_dir)
        outputs = get_generations(model, samples, sampling_params)
        score = rouge(predictions=outputs,
                            references=[sample['answer'] for sample in samples],
                            return_geometric_mean=True)
        samples = [dict(sample, **{'generation':output}) for sample, output in zip(samples, outputs)]
        pickle.dump(samples, open(f"{output_dir}/{dataset_name}_{metric_name}{score}.pkl", "wb"))
        print_rank_0(f"{datetime.now().strftime('%H:%M:%S')} evaluated model: {model_dir}, dataset: {dataset_name}")
    del model

    return


def main():
    args = construct_arguments()
    if args.model_dir is not None:
        assert args.output_dir is not None and args.eval_sets is not None
        run_eval_sets(model_dir=args.model_dir, output_dir=args.output_dir,eval_sets=args.eval_sets.strip().split('_'))
    else:
        with open("/home/data/xiaohong/exploration_data/exploration_evaluation_zero_scrolls_config.txt", "r") as f:
            config = f.readlines()
        for line in config:
            run_eval_sets(model_dir=line.split(',')[0], output_dir=line.split(',')[1], eval_sets=line.split(',')[2].strip().split('_'))
        return
if __name__ == '__main__':
    main()