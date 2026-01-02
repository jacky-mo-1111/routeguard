# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import TYPE_CHECKING, Optional

from transformers import LogitsProcessorList

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.logits_processors import build_tag_debug_processor
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if generating_args.enable_tag_debug:
        tag_processor, missing_tokens = build_tag_debug_processor(
            tokenizer, generating_args.tag_debug_trigger_token, generating_args.tag_debug_candidate_tokens
        )
        if missing_tokens:
            logger.warning_rank0(f"Tag debug skipped tokens not in vocab: {','.join(missing_tokens)}")
        if tag_processor is not None:
            gen_kwargs["logits_processor"] = LogitsProcessorList([tag_processor])

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        eval_dataset = dataset_module["eval_dataset"]

        # Case 1: eval_dataset already a dict -> run predict per dataset
        if isinstance(eval_dataset, dict):
            for dataset_name, dataset in eval_dataset.items():
                dataset_output_dir = os.path.join(training_args.output_dir, dataset_name)
                os.makedirs(dataset_output_dir, exist_ok=True)

                logger.info_rank0(f"Predicting on dataset: {dataset_name}")
                predict_results = trainer.predict(dataset, metric_key_prefix=f"predict_{dataset_name}", **gen_kwargs)
                trainer.log_metrics(f"predict_{dataset_name}", predict_results.metrics)
                trainer.save_metrics(f"predict_{dataset_name}", predict_results.metrics)
                trainer.save_predictions(
                    dataset,
                    predict_results,
                    generating_args.skip_special_tokens,
                    output_dir=dataset_output_dir,
                    skip_eot_id=getattr(generating_args, "skip_eot_id", False),
                )

        # Case 2: merged eval dataset but multiple names specified -> predict once, then split and save per dataset
        elif data_args.eval_dataset is not None and len(data_args.eval_dataset) > 1 and not data_args.eval_on_each_dataset:
            logger.info_rank0("Multiple eval datasets were merged. Results will be split by dataset size.")
            predict_results = trainer.predict(eval_dataset, metric_key_prefix="predict", **gen_kwargs)
            trainer.log_metrics("predict", predict_results.metrics)
            trainer.save_metrics("predict", predict_results.metrics)

            # Get dataset sizes from the original datasets
            from ...data.loader import get_dataset_list, _load_single_dataset

            dataset_list = get_dataset_list(data_args.eval_dataset, data_args.dataset_dir)
            dataset_sizes = {}
            for dataset_name, dataset_attr in zip(data_args.eval_dataset, dataset_list):
                original_dataset = _load_single_dataset(dataset_attr, model_args, data_args, training_args)
                dataset_sizes[dataset_name] = len(original_dataset)

            # Split and save predictions
            total_size = len(predict_results.predictions)
            start_idx = 0
            for dataset_name, size in dataset_sizes.items():
                end_idx = min(start_idx + size, total_size)

                dataset_output_dir = os.path.join(training_args.output_dir, dataset_name)
                os.makedirs(dataset_output_dir, exist_ok=True)

                # Create subset of results and dataset
                from datasets import Dataset
                subset_dataset = Dataset.from_dict(
                    {
                        "input_ids": eval_dataset["input_ids"][start_idx:end_idx],
                        "attention_mask": eval_dataset["attention_mask"][start_idx:end_idx],
                    }
                )

                subset_predictions = predict_results.predictions[start_idx:end_idx]
                subset_label_ids = predict_results.label_ids[start_idx:end_idx] if predict_results.label_ids is not None else None

                from transformers.trainer import PredictionOutput
                subset_results = PredictionOutput(predictions=subset_predictions, label_ids=subset_label_ids, metrics={})

                trainer.save_predictions(
                    subset_dataset,
                    subset_results,
                    generating_args.skip_special_tokens,
                    output_dir=dataset_output_dir,
                    skip_eot_id=getattr(generating_args, "skip_eot_id", False),
                )
                logger.info_rank0(f"Saved predictions for {dataset_name} ({end_idx - start_idx} samples) to {dataset_output_dir}")

                start_idx = end_idx
                if start_idx >= total_size:
                    break

        # Case 3: single eval dataset
        else:
            predict_results = trainer.predict(eval_dataset, metric_key_prefix="predict", **gen_kwargs)
            trainer.log_metrics("predict", predict_results.metrics)
            trainer.save_metrics("predict", predict_results.metrics)
            trainer.save_predictions(
                eval_dataset,
                predict_results,
                generating_args.skip_special_tokens,
                skip_eot_id=getattr(generating_args, "skip_eot_id", False),
            )

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
