# This script is adopted from with adjustment https://github.com/microsoft/DeepSpeedExamples/blob/8f8099a813f3b223d5df39e0c15c748de4eb1669/training/bing_bert/deepspeed_train.py
import numpy as np
from datetime import datetime
import argparse
import random
import wandb
import time
import os
import math

import torch
import torch.distributed as dist

import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from dataset import ConcatDataset, JsonlDataset, DefaultDataCollatorForFinetune
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

IGNORE_INDEX = -100
HUGGINGFACE_TOKEN="xxx"

def get_argument_parser():
    parser = argparse.ArgumentParser()

    # Required_parameter
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num_epoches", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--rope_scaling_factor", type=float, default=None)
    parser.add_argument("--rope_scaling_type", type=str, default=None)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--decay_style", type=str, default='cosine')
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--train_data_weight", type=str)
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--check_data", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--eval_micro_batch_size_per_gpu", type=int)
    parser.add_argument("--gradient_checkpointing", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--neftune_noise_alpha", type=float, default=None)
    # in DS config too
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--zero_stage", type=int, default=1)

    return parser

def construct_arguments():
    parser = get_argument_parser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    args.train_data = args.train_data.split(',')
    args.train_data_weight = [float(w) for w in args.train_data_weight.split(',')]
    args.eval_data = args.eval_data.split(',') if args.eval_data is not None else args.eval_data
    if args.neftune_noise_alpha is not None:
        # TODO, @LydiaXiaohongLi to support evaluation of neftune
        assert args.eval_data is None
    args.shuffle = True
    # no cuda mode is not supported
    args.no_cuda = False
    # Setting the distributed variables
    print("Args = {}".format(args))

    deepspeed.init_distributed(dist_backend='nccl')
    args.dp_world_size = dist.get_world_size()

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return args


def wandb_init():
    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    wandb.init(
        project="finetune-longcamel",
    )



def prepare_optimizer_parameters(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                                     'weight_decay': args.weight_decay},
                                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                                     'weight_decay': 0.0}]
    return optimizer_grouped_parameters

def neftune_forward(self, input: torch.Tensor):
    """
    Implements the NEFTune forward pass for the model. Note this works only for
    torch.nn.Embedding layers. This method is slightly adapted from the original source code
    that can be found here: https://github.com/neelsjain/NEFTune

    Args:
        input (`torch.Tensor`):
            The input tensor to the model.
        noise_alpha (`float`):
            The noise alpha value to use for the NEFTune forward pass.
    """
    embeddings = torch.nn.functional.embedding(
        input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
    )

    if self.training:
        dims = torch.tensor(embeddings.size(1) * embeddings.size(2))
        mag_norm = self.neftune_noise_alpha / torch.sqrt(dims)
        embeddings = embeddings + torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)

    return embeddings

def _activate_neftune(args, model):
    r"""
    Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914
    """
    embeddings = model.get_input_embeddings()


    embeddings.neftune_noise_alpha = args.neftune_noise_alpha
    old_forward = embeddings.forward

    # This hack seems to be needed to properly use a custom forward pass
    # all credits to: https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/11
    bound_method = neftune_forward.__get__(embeddings, embeddings.__class__)
    setattr(embeddings, "forward", bound_method)

    # embeddings.forward = neftune_forward
    embeddings._trl_old_forward = old_forward

    return model

def prepare_model_optimizer(args):

    config = AutoConfig.from_pretrained(args.model_name, token=HUGGINGFACE_TOKEN)
    config._flash_attn_2_enabled = True
    if args.rope_scaling_type is not None and args.rope_scaling_factor is not None:
        config.rope_scaling = {'type': args.rope_scaling_type, 'factor': args.rope_scaling_factor}
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(AutoModelForCausalLM.from_pretrained(args.model_name, token=HUGGINGFACE_TOKEN).state_dict())
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=HUGGINGFACE_TOKEN)
    special_tokens_dict = {'additional_special_tokens': ['[EOT]'], 'pad_token': "[PAD]"}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(int(8 *math.ceil(len(tokenizer) / 8.0)))
    model.model.gradient_checkpointing = args.gradient_checkpointing
    if args.neftune_noise_alpha is not None:
        # this is adopted from https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py#L354,
        # but mag_norm does not consider actual input length without padding
        model = _activate_neftune(args, model)

    # Optimizer parameters
    optimizer_grouped_parameters = prepare_optimizer_parameters(args, model)
    optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.95), )

    # DeepSpeed initializer handles FP16, distributed, optimizer automatically.
    model, optimizer, _, _ = deepspeed.initialize(args=args, model=model, optimizer=optimizer)

    # Overwrite application configs with DeepSpeed config
    args.train_micro_batch_size_per_gpu = model.train_micro_batch_size_per_gpu()
    args.device = model.device
    args.gradient_accumulation_steps = model.gradient_accumulation_steps()

    return model, optimizer, tokenizer


def get_dataloader(args, tokenizer):
    data_collator = DefaultDataCollatorForFinetune(tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    train_dataset = ConcatDataset([JsonlDataset(file) for file in args.train_data], shuffle=args.shuffle, weights=args.train_data_weight)
    # Input data_files already random shuffled, for deterministic data samples given global_steps
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_micro_batch_size_per_gpu,
                                               sampler=train_sampler,
                                               collate_fn=data_collator,
                                               drop_last=True)
    eval_loader = None
    if args.eval_data is not None:
        eval_dataset = ConcatDataset([JsonlDataset(file) for file in args.eval_data])
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
        eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                                  batch_size=args.eval_micro_batch_size_per_gpu,
                                                  sampler=eval_sampler,
                                                  collate_fn=data_collator,
                                                  drop_last=True)
    return train_loader, eval_loader


def process_batch(batch):
    input_ids = batch['input_ids'].long()
    labels = batch['labels'].long()
    return input_ids, labels

def save_model_hf(model, tokenizer, args, sub_folder=""):
    def _z3_params_to_fetch(param_list):
        return [p for p in param_list if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.save, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    if dist.get_rank() == 0:
        tokenizer.save_pretrained(output_dir)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    if args.zero_stage==3:
        model_to_save = model.module if hasattr(model, 'module') else model
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]),enabled=True):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if dist.get_rank() == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if dist.get_rank() == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict
    else:
        save_dict = model_to_save.state_dict()
        for key in list(save_dict.keys()):
            if "lora" in key:
                del save_dict[key]
        torch.save(save_dict, output_model_file)
    if dist.get_rank() == 0:
        model_to_save.config.to_json_file(output_config_file)


def main():
    args = construct_arguments()
    wandb_init()
    model, optimizer, tokenizer = prepare_model_optimizer(args)
    train_dataloader, eval_dataloader = get_dataloader(args, tokenizer)
    args.total_steps = len(train_dataloader)*args.num_epoches/(args.train_batch_size/args.train_micro_batch_size_per_gpu/args.dp_world_size)
    print(f"{datetime.now().strftime('%H:%M:%S')}, on rank: {dist.get_rank()}, train_dataloader size: {len(train_dataloader)}")


    def evaluation():
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            input_ids, labels = process_batch(batch)
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=labels)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        torch.distributed.all_reduce(losses, op=torch.distributed.ReduceOp.SUM)
        losses = losses / torch.distributed.get_world_size()
        return losses


    for epoch in range(args.num_epoches):
        model.train()
        start_time = time.time()
        for step, batch in enumerate(train_dataloader):
            input_ids, labels = process_batch(batch)
            if args.check_data and step <5 and epoch==0:
                for input_id, label in zip(input_ids, labels):
                    print(f"*****prompt*****\n{tokenizer.decode(input_id)}\n*****label*****\n{tokenizer.decode(label)}")
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)

            # Calculate forward pass
            loss = model(input_ids=input_ids, labels=labels)[0]

            model.backward(loss)

            if model.is_gradient_accumulation_boundary():
                print(f"{datetime.now().strftime('%H:%M:%S')}, on rank: {dist.get_rank()}, epoch: {epoch}, step: {step}, losses_reduced: {loss}, grad_norm: {model.get_global_grad_norm()}, global_steps {model.global_steps}, takes {time.time()-start_time}s")
                wandb.log({"loss": loss, "iteration_time": time.time()-start_time, "grad_norm": model.get_global_grad_norm()})
                start_time = time.time()
                model.step()
                if (model.global_steps % args.save_interval) == 0:
                    save_model_hf(model, tokenizer, args, sub_folder=f"epoch{epoch}_globalstep{model.global_steps}")
            else:
                model.step()

        save_model_hf(model, tokenizer, args, sub_folder=f"epoch{epoch}")

        if args.eval_data is not None:
            eval_losses = evaluation()
            if dist.get_rank() == 0:
                print(f"{datetime.now().strftime('%H:%M:%S')}, epoch: {epoch}, eval_loss: {eval_losses}")

    wandb.finish()
    return

if __name__ == "__main__":
    main()