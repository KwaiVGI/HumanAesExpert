# Copyright (c) Alibaba, Inc. and its affiliates.
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from torch import nn
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available

from swift.torchacc_utils import patch_clip_grad_norm, ta_trim_graph
from swift.utils import use_torchacc
from .loss import get_loss_func
from .mixin import SwiftMixin
from .push_to_ms import PushToMsHubMixin

from transformers.trainer import *


class Trainer(PushToMsHubMixin, SwiftMixin, HfTrainer):
    pass


class Seq2SeqTrainer(PushToMsHubMixin, SwiftMixin, HfSeq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # performance
        if not hasattr(self, 'perf'):
            self.perf = {}
        self.perf.update({
            'gen_time': 0.,
            'gen_len': 0,
        })
        self._acc = torch.tensor(0.).to(self.args.device)
        if use_torchacc():
            patch_clip_grad_norm(self.accelerator)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)

        inputs.pop('loss_scale', None)
        has_labels = 'labels' in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()

        if len(gen_kwargs) == 0 and hasattr(self, '_gen_kwargs'):
            gen_kwargs = self._gen_kwargs.copy()
            if hasattr(self.model, 'generation_config'):
                gen_kwargs.update(self.model.generation_config.to_dict())

        if gen_kwargs.get('max_length') is None and gen_kwargs.get('max_new_tokens') is None:
            gen_kwargs['max_length'] = self.model.config.max_length
        gen_kwargs['num_beams'] = (
            gen_kwargs['num_beams'] if gen_kwargs.get('num_beams') is not None else self.model.config.num_beams)
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs['synced_gpus'] = (
            gen_kwargs['synced_gpus'] if gen_kwargs.get('synced_gpus') is not None else default_synced_gpus)

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if ('labels' in inputs and 'decoder_input_ids' in inputs
                and inputs['labels'].shape == inputs['decoder_input_ids'].shape):
            inputs = {k: v for k, v in inputs.items() if k != 'decoder_input_ids'}

        gen_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        gen_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        # fix generate warning
        if 'max_length' in gen_kwargs and 'max_new_tokens' in gen_kwargs and gen_kwargs['max_new_tokens'] is not None:
            gen_kwargs.pop('max_length')
        gen_time = time.time()
        generate_inputs = inputs.copy()
        if has_labels:
            _labels = inputs['labels'][0]
            n_mask = 0
            for i in range(len(_labels)):
                if _labels[i] != -100:
                    n_mask = i
                    break

            for k in ['input_ids', 'attention_mask']:
                generate_inputs[k] = generate_inputs[k][:, :n_mask]
            generate_inputs['labels'] = generate_inputs['labels'][:, n_mask:]

        generated_tokens = self.model.generate(**generate_inputs, **gen_kwargs)
        gen_time = time.time() - gen_time

        if hasattr(self.model, 'encoder') and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = generate_inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = generate_inputs[self.model.main_input_name]

        generated_tokens = generated_tokens[:, generation_inputs.shape[1]:]
        gen_len = len(generated_tokens[0])
        self.perf['gen_time'] = self.perf['gen_time'] + gen_time
        self.perf['gen_len'] = self.perf['gen_len'] + gen_len

        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get('max_length') is not None and generated_tokens.shape[-1] < gen_kwargs['max_length']:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs['max_length'])
        elif gen_kwargs.get('max_new_tokens') is not None and generated_tokens.shape[-1] < (gen_kwargs['max_new_tokens']
                                                                                            + 1):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs['max_new_tokens'] + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs['labels']).mean().detach()
                else:
                    loss = (outputs['loss'] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = generate_inputs['labels']
            if gen_kwargs.get('max_length') is not None and labels.shape[-1] < gen_kwargs['max_length']:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs['max_length'])
            elif gen_kwargs.get('max_new_tokens') is not None and labels.shape[-1] < (gen_kwargs['max_new_tokens'] + 1):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs['max_new_tokens'] + 1))
        else:
            labels = None

        return loss, generated_tokens, labels
    
    # def get_batch_samples(self, epoch_iterator, num_batches):
    #     batch_samples = []
    #     num_items_in_batch = None
    #     for _ in range(num_batches):
    #         try:
    #             batch_samples += [next(epoch_iterator)]
    #         except StopIteration:
    #             break
    #     if len(batch_samples) > 0 and "labels" in batch_samples[0]:
    #         # For now we don't support object detection
    #         try:
    #             num_items_in_batch = sum(
    #                 [data_batch["labels"][..., 1:].ne(-100).sum().item() for data_batch in batch_samples]
    #             )
    #         except TypeError:
    #             pass
    #     return batch_samples, num_items_in_batch

    
    # def _inner_training_loop(
    #     self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    # ):
    #     self.accelerator.free_memory()
    #     self._train_batch_size = batch_size
    #     if self.args.auto_find_batch_size:
    #         if self.state.train_batch_size != self._train_batch_size:
    #             from accelerate.utils import release_memory

    #             (self.model_wrapped,) = release_memory(self.model_wrapped)
    #             self.model_wrapped = self.model

    #             # Check for DeepSpeed *after* the intial pass and modify the config
    #             if self.is_deepspeed_enabled:
    #                 # Temporarily unset `self.args.train_batch_size`
    #                 original_bs = self.args.per_device_train_batch_size
    #                 self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
    #                 self.propagate_args_to_deepspeed(True)
    #                 self.args.per_device_train_batch_size = original_bs
    #         self.state.train_batch_size = self._train_batch_size
    #     logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
    #     # Data loader and number of training steps
    #     train_dataloader = self.get_train_dataloader()
    #     if self.is_fsdp_xla_v2_enabled:
    #         train_dataloader = tpu_spmd_dataloader(train_dataloader)

    #     # Setting up training control variables:
    #     # number of training epochs: num_train_epochs
    #     # number of training steps per epoch: num_update_steps_per_epoch
    #     # total number of training steps to execute: max_steps
    #     total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

    #     len_dataloader = None
    #     num_train_tokens = None
    #     if has_length(train_dataloader):
    #         len_dataloader = len(train_dataloader)
    #         num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
    #         num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    #         num_examples = self.num_examples(train_dataloader)
    #         if args.max_steps > 0:
    #             max_steps = args.max_steps
    #             num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
    #                 args.max_steps % num_update_steps_per_epoch > 0
    #             )
    #             # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
    #             # the best we can do.
    #             num_train_samples = args.max_steps * total_train_batch_size
    #             if args.include_tokens_per_second:
    #                 num_train_tokens = (
    #                     self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
    #                 )
    #         else:
    #             max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    #             num_train_epochs = math.ceil(args.num_train_epochs)
    #             num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
    #             if args.include_tokens_per_second:
    #                 num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
    #     elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
    #         max_steps = args.max_steps
    #         # Setting a very large number of epochs so we go as many times as necessary over the iterator.
    #         num_train_epochs = sys.maxsize
    #         num_update_steps_per_epoch = max_steps
    #         num_examples = total_train_batch_size * args.max_steps
    #         num_train_samples = args.max_steps * total_train_batch_size
    #         if args.include_tokens_per_second:
    #             num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
    #     else:
    #         raise ValueError(
    #             "args.max_steps must be set to a positive value if dataloader does not have a length, was"
    #             f" {args.max_steps}"
    #         )

    #     if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
    #         if self.args.n_gpu > 1:
    #             # nn.DataParallel(model) replicates the model, creating new variables and module
    #             # references registered here no longer work on other gpus, breaking the module
    #             raise ValueError(
    #                 "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
    #                 " (torchrun or torch.distributed.launch (deprecated))."
    #             )
    #         else:
    #             debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

    #     delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

    #     # We need to reset the scheduler, as its parameters may be different on subsequent calls
    #     if self._created_lr_scheduler:
    #         self.lr_scheduler = None
    #         self._created_lr_scheduler = False

    #     if self.is_deepspeed_enabled:
    #         self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

    #     if not delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     self.state = TrainerState(
    #         stateful_callbacks=[
    #             cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
    #         ]
    #     )
    #     self.state.is_hyper_param_search = trial is not None
    #     self.state.train_batch_size = self._train_batch_size

    #     # Compute absolute values for logging, eval, and save if given as ratio
    #     if args.logging_steps is not None:
    #         if args.logging_steps < 1:
    #             self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
    #         else:
    #             self.state.logging_steps = args.logging_steps
    #     if args.eval_steps is not None:
    #         if args.eval_steps < 1:
    #             self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
    #         else:
    #             self.state.eval_steps = args.eval_steps
    #     if args.save_steps is not None:
    #         if args.save_steps < 1:
    #             self.state.save_steps = math.ceil(max_steps * args.save_steps)
    #         else:
    #             self.state.save_steps = args.save_steps

    #     # Activate gradient checkpointing if needed
    #     if args.gradient_checkpointing:
    #         self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

    #     model = self._wrap_model(self.model_wrapped)

    #     # as the model is wrapped, don't use `accelerator.prepare`
    #     # this is for unhandled cases such as
    #     # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
    #     use_accelerator_prepare = True if model is self.model else False

    #     if delay_optimizer_creation:
    #         if use_accelerator_prepare:
    #             self._fsdp_qlora_plugin_updates()
    #             self.model = self.accelerator.prepare(self.model)
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     # prepare using `accelerator` prepare
    #     if use_accelerator_prepare:
    #         self.model.train()
    #         if hasattr(self.lr_scheduler, "step"):
    #             if self.use_apex:
    #                 model = self.accelerator.prepare(self.model)
    #             else:
    #                 model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
    #         else:
    #             # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
    #             model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
    #                 self.model, self.optimizer, self.lr_scheduler
    #             )
    #     elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
    #         # In this case we are in DDP + LOMO, which should be supported
    #         self.optimizer = self.accelerator.prepare(self.optimizer)

    #     if self.is_fsdp_enabled:
    #         self.model = self.model_wrapped = model

    #     # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #     if model is not self.model:
    #         self.model_wrapped = model

    #     # backward compatibility
    #     if self.is_deepspeed_enabled:
    #         self.deepspeed = self.model_wrapped

    #     # ckpt loading
    #     if resume_from_checkpoint is not None:
    #         if self.is_deepspeed_enabled:
    #             deepspeed_load_checkpoint(
    #                 self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
    #             )
    #         elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
    #             self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

    #     # Check if saved optimizer or scheduler states exist
    #     self._load_optimizer_and_scheduler(resume_from_checkpoint)

    #     # important: at this point:
    #     # self.model         is the Transformers Model
    #     # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
    #     # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

    #     # Train!
    #     logger.info("***** Running training *****")
    #     logger.info(f"  Num examples = {num_examples:,}")
    #     logger.info(f"  Num Epochs = {num_train_epochs:,}")
    #     logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
    #     if self.args.per_device_train_batch_size != self._train_batch_size:
    #         logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
    #     logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
    #     logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    #     logger.info(f"  Total optimization steps = {max_steps:,}")
    #     logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

    #     self.state.epoch = 0
    #     start_time = time.time()
    #     epochs_trained = 0
    #     steps_trained_in_current_epoch = 0
    #     steps_trained_progress_bar = None

    #     # Check if continuing training from a checkpoint
    #     if resume_from_checkpoint is not None and os.path.isfile(
    #         os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    #     ):
    #         self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
    #         self.compare_trainer_and_checkpoint_args(self.args, self.state)
    #         self._load_callback_state()
    #         epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
    #         if not args.ignore_data_skip:
    #             steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
    #             steps_trained_in_current_epoch *= args.gradient_accumulation_steps
    #         else:
    #             steps_trained_in_current_epoch = 0

    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info(f"  Continuing training from epoch {epochs_trained}")
    #         logger.info(f"  Continuing training from global step {self.state.global_step}")
    #         if not args.ignore_data_skip:
    #             logger.info(
    #                 f"  Will skip the first {epochs_trained} epochs then the first"
    #                 f" {steps_trained_in_current_epoch} batches in the first epoch."
    #             )

    #     # Update the references
    #     self.callback_handler.model = self.model
    #     self.callback_handler.optimizer = self.optimizer
    #     self.callback_handler.lr_scheduler = self.lr_scheduler
    #     self.callback_handler.train_dataloader = train_dataloader
    #     if self.hp_name is not None and self._trial is not None:
    #         # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
    #         # parameter to Train when using DDP.
    #         self.state.trial_name = self.hp_name(self._trial)
    #     if trial is not None:
    #         assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
    #         self.state.trial_params = hp_params(assignments)
    #     else:
    #         self.state.trial_params = None
    #     # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    #     # to set this after the load.
    #     self.state.max_steps = max_steps
    #     self.state.num_train_epochs = num_train_epochs
    #     self.state.is_local_process_zero = self.is_local_process_zero()
    #     self.state.is_world_process_zero = self.is_world_process_zero()

    #     # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    #     tr_loss = torch.tensor(0.0).to(args.device)
    #     # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    #     self._total_loss_scalar = 0.0
    #     self._globalstep_last_logged = self.state.global_step
    #     model.zero_grad()
    #     grad_norm: Optional[float] = None
    #     self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    #     if args.eval_on_start:
    #         self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

    #     total_batched_samples = 0
    #     for epoch in range(epochs_trained, num_train_epochs):
    #         epoch_dataloader = train_dataloader
    #         if hasattr(epoch_dataloader, "set_epoch"):
    #             epoch_dataloader.set_epoch(epoch)

    #         # Reset the past mems state at the beginning of each epoch if necessary.
    #         if args.past_index >= 0:
    #             self._past = None

    #         steps_in_epoch = (
    #             len(epoch_dataloader)
    #             if len_dataloader is not None
    #             else args.max_steps * args.gradient_accumulation_steps
    #         )
    #         self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

    #         if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
    #             self._load_rng_state(resume_from_checkpoint)

    #         rng_to_sync = False
    #         steps_skipped = 0
    #         if steps_trained_in_current_epoch > 0:
    #             epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
    #             steps_skipped = steps_trained_in_current_epoch
    #             steps_trained_in_current_epoch = 0
    #             rng_to_sync = True

    #         step = -1
    #         epoch_iterator = iter(epoch_dataloader)
    #         # We chunkify the epoch iterator into gradient accumulation steps `n` batches
    #         remainder = num_examples % args.gradient_accumulation_steps
    #         num_items_in_batch = None
    #         if remainder == 0:
    #             remainder = args.gradient_accumulation_steps
    #         update_step = -1
    #         total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
    #         for _ in range(total_updates):
    #             update_step += 1
    #             num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
    #             batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
    #             print('\n\n\n_inner_training_loop:', batch_samples[0].keys(),'\n\n\n')
    #             for inputs in batch_samples:
    #                 step += 1
    #                 total_batched_samples += 1
    #                 is_last_step_and_steps_less_than_grad_acc = (
    #                     steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
    #                 )
    #                 do_sync_step = is_last_step_and_steps_less_than_grad_acc or (
    #                     total_batched_samples % args.gradient_accumulation_steps == 0
    #                 )
    #                 # Since we perform prefetching, we need to manually set sync_gradients
    #                 if not do_sync_step:
    #                     self.accelerator.gradient_state._set_sync_gradients(False)
    #                 else:
    #                     self.accelerator.gradient_state._set_sync_gradients(True)

    #                 if self.args.include_num_input_tokens_seen:
    #                     main_input_name = getattr(self.model, "main_input_name", "input_ids")
    #                     if main_input_name not in inputs:
    #                         logger.warning(
    #                             "Tried to track the number of tokens seen, however the current model is "
    #                             "not configured properly to know what item is the input. To fix this, add "
    #                             "a `main_input_name` attribute to the model class you are using."
    #                         )
    #                     else:
    #                         input_tokens = inputs[main_input_name].numel()
    #                         input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
    #                         self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).cpu().item()
    #                 if rng_to_sync:
    #                     self._load_rng_state(resume_from_checkpoint)
    #                     rng_to_sync = False

    #                 # Skip past any already trained steps if resuming training
    #                 if steps_trained_in_current_epoch > 0:
    #                     steps_trained_in_current_epoch -= 1
    #                     if steps_trained_progress_bar is not None:
    #                         steps_trained_progress_bar.update(1)
    #                     if steps_trained_in_current_epoch == 0:
    #                         self._load_rng_state(resume_from_checkpoint)
    #                     continue
    #                 elif steps_trained_progress_bar is not None:
    #                     steps_trained_progress_bar.close()
    #                     steps_trained_progress_bar = None

    #                 if step % args.gradient_accumulation_steps == 0:
    #                     self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

    #                 with self.accelerator.accumulate(model):
    #                     tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

    #                 if (
    #                     args.logging_nan_inf_filter
    #                     and not is_torch_xla_available()
    #                     and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
    #                 ):
    #                     # if loss is nan or inf simply add the average of previous logged losses
    #                     tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
    #                 else:
    #                     if tr_loss.device != tr_loss_step.device:
    #                         raise ValueError(
    #                             f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
    #                         )
    #                     tr_loss = tr_loss + tr_loss_step

    #                 self.current_flos += float(self.floating_point_ops(inputs))

    #                 if do_sync_step:
    #                     # Since we perform prefetching, we need to manually set sync_gradients to True
    #                     self.accelerator.gradient_state._set_sync_gradients(True)

    #                     # Gradient clipping
    #                     if args.max_grad_norm is not None and args.max_grad_norm > 0:
    #                         # deepspeed does its own clipping

    #                         if is_sagemaker_mp_enabled() and args.fp16:
    #                             _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
    #                         elif self.use_apex:
    #                             # Revert to normal clipping otherwise, handling Apex or full precision
    #                             _grad_norm = nn.utils.clip_grad_norm_(
    #                                 amp.master_params(self.optimizer),
    #                                 args.max_grad_norm,
    #                             )
    #                         else:
    #                             _grad_norm = self.accelerator.clip_grad_norm_(
    #                                 model.parameters(),
    #                                 args.max_grad_norm,
    #                             )

    #                         if (
    #                             is_accelerate_available()
    #                             and self.accelerator.distributed_type == DistributedType.DEEPSPEED
    #                         ):
    #                             grad_norm = model.get_global_grad_norm()
    #                             # In some cases the grad norm may not return a float
    #                             if hasattr(grad_norm, "item"):
    #                                 grad_norm = grad_norm.item()
    #                         else:
    #                             grad_norm = _grad_norm

    #                     self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

    #                     self.optimizer.step()

    #                     self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

    #                     optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
    #                     if optimizer_was_run:
    #                         # Delay optimizer scheduling until metrics are generated
    #                         if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #                             self.lr_scheduler.step()

    #                     model.zero_grad()
    #                     self.state.global_step += 1
    #                     self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
    #                     self.control = self.callback_handler.on_step_end(args, self.state, self.control)
    #                     self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
    #                 else:
    #                     self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

    #                 # PyTorch/XLA relies on the data loader to insert the mark_step for
    #                 # each step. Since we are breaking the loop early, we need to manually
    #                 # insert the mark_step here.
    #                 if self.control.should_epoch_stop or self.control.should_training_stop:
    #                     if is_torch_xla_available():
    #                         xm.mark_step()
    #                     break
    #             # We also need to break out of the nested loop
    #             if self.control.should_epoch_stop or self.control.should_training_stop:
    #                 if is_torch_xla_available():
    #                     xm.mark_step()
    #                 break
    #         if step < 0:
    #             logger.warning(
    #                 "There seems not to be a single sample in your epoch_iterator, stopping training at step"
    #                 f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
    #                 f" num_steps ({max_steps}) higher than the number of available samples."
    #             )
    #             self.control.should_training_stop = True

    #         self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
    #         self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

    #         if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
    #             if is_torch_xla_available():
    #                 # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
    #                 xm.master_print(met.metrics_report())
    #             else:
    #                 logger.warning(
    #                     "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
    #                     "configured. Check your training configuration if this is unexpected."
    #                 )
    #         if self.control.should_training_stop:
    #             break

    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of training
    #         delattr(self, "_past")

    #     logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    #     if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    #         # Wait for everyone to get here so we are sure the model has been saved by process 0.
    #         if is_torch_xla_available():
    #             xm.rendezvous("load_best_model_at_end")
    #         elif args.parallel_mode == ParallelMode.DISTRIBUTED:
    #             dist.barrier()
    #         elif is_sagemaker_mp_enabled():
    #             smp.barrier()

    #         self._load_best_model()

    #     # add remaining tr_loss
    #     self._total_loss_scalar += tr_loss.item()
    #     effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
    #     train_loss = self._total_loss_scalar / effective_global_step

    #     metrics = speed_metrics(
    #         "train",
    #         start_time,
    #         num_samples=num_train_samples,
    #         num_steps=self.state.max_steps,
    #         num_tokens=num_train_tokens,
    #     )
    #     self.store_flos()
    #     metrics["total_flos"] = self.state.total_flos
    #     metrics["train_loss"] = train_loss

    #     self.is_in_train = False

    #     self._memory_tracker.stop_and_update_metrics(metrics)

    #     self.log(metrics)

    #     run_dir = self._get_output_dir(trial)
    #     checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

    #     # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    #     if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
    #         for checkpoint in checkpoints_sorted:
    #             if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
    #                 logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
    #                 shutil.rmtree(checkpoint, ignore_errors=True)

    #     self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    #     # Wait for the checkpoint to be uploaded.
    #     self._finish_current_push()

    #     # After training we make sure to retrieve back the original forward pass method
    #     # for the embedding layer by removing the forward post hook.
    #     if self.neftune_noise_alpha is not None:
    #         self._deactivate_neftune(self.model)

    #     return TrainOutput(self.state.global_step, train_loss, metrics)

    # def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
    #     """
    #     Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
    #     handling potential state.
    #     """
    #     print('_prepare_inputs:', inputs.keys())
    #     inputs = self._prepare_input(inputs)
    #     if len(inputs) == 0:
    #         raise ValueError(
    #             "The batch received was empty, your model won't be able to train on it. Double-check that your "
    #             f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
    #         )
    #     if self.args.past_index >= 0 and self._past is not None:
    #         inputs["mems"] = self._past

    #     return inputs

    def compute_loss(self, model, inputs, return_outputs=None, num_items_in_batch=None):
        if not hasattr(self, '_custom_metrics'):
            self._custom_metrics = {}
        labels = None
        loss_name = self.args.loss_name
        if loss_name is None and 'loss_scale' in inputs:
            loss_name = 'loss-scale'

        loss_kwargs = {'num_items_in_batch': num_items_in_batch}
        if loss_name == 'loss-scale':
            loss_kwargs['loss_scale'] = inputs.pop('loss_scale', None)

        if loss_name is not None or self.label_smoother is not None and 'labels' in inputs:
            labels = inputs.pop('labels')

        loss_kwargs['labels'] = labels
        outputs = model(**inputs)
        # fix https://github.com/huggingface/transformers/issues/34263
        if 'labels' in inputs and num_items_in_batch is not None:
            outputs.loss = outputs.loss * (inputs['labels'][:, 1:] != -100).sum() / num_items_in_batch
        if loss_name is not None:
            loss_func = get_loss_func(loss_name)
            outputs['loss'] = loss_func(outputs, **loss_kwargs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None and loss_name is None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]

        if labels is None:
            labels = inputs['labels']

        if self.sequence_parallel_size > 1:
            from swift.trainers.xtuner import reduce_xtuner_sequence_parallel_loss
            loss = reduce_xtuner_sequence_parallel_loss(loss, labels)

        if self.is_encoder_decoder:
            preds = outputs.logits.argmax(dim=2)[..., :] if outputs.logits is not None else None
            labels = labels[..., :]
        else:
            preds = outputs.logits.argmax(dim=2)[..., :-1] if outputs.logits is not None else None
            labels = labels[..., 1:]

        masks = labels != -100
        acc_strategy = getattr(self.args, 'acc_strategy', 'token')
        acc: Optional[torch.Tensor] = None
        sft_args = getattr(self, 'sft_args', None)
        acc_steps = 1 if sft_args is None else sft_args.acc_steps
        if self.state.global_step % acc_steps == 0 and preds is not None:
            if preds.shape != labels.shape:
                pass
            elif acc_strategy == 'sentence':
                acc_list = []
                for i, m in enumerate(masks):
                    acc_list.append(torch.all(preds[i, m] == labels[i, m]).to(torch.int64).item())
                acc = torch.tensor(acc_list, device=preds.device).float().mean()
            else:
                if use_torchacc():
                    ta_trim_graph()
                    preds = preds.to('cpu')
                    masks = masks.to('cpu')
                    labels = labels.to('cpu')
                acc = (torch.masked_select(preds, masks) == torch.masked_select(labels, masks)).float().mean()
            if model.training and acc is not None:
                if 'acc' not in self._custom_metrics:
                    self._custom_metrics['acc'] = self._acc
                self._custom_metrics['acc'] = self._custom_metrics['acc'] + acc / self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss
