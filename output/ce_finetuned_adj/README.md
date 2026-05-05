---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:7080
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) <!-- at revision 1427fd652930e4ba29e8149678df786c240d8825 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'XLMRobertaForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['ถ้าผู้ให้ตาย ในช่วงที่ยังชำระหนี้เป็นครั้งคราวให้กับผู้รับ หนี้ที่ยังค้างอยู่จะมีผลเป็นยังไง', 'SECTION: [แพ่ง] มาตรา 525\nมาตรา ๕๒๕ การให้ทรัพย์สินซึ่งถ้าจะซื้อขายกันจะต้องทำเป็นหนังสือและจดทะเบียนต่อพนักงานเจ้าหน้าที่นั้น ท่านว่าย่อมสมบูรณ์ต่อเมื่อได้ทำเป็นหนังสือและจดทะเบียนต่อพนักงานเจ้าหน้าที่ ในกรณีเช่นนี้ การให้ย่อมเป็นอันสมบูรณ์โดยมิพักต้องส่งมอบ'],
    ['ในการจำนำทรัพย์สินหลายสิ่งเพื่อประกันหนี้รายเดียว ผู้รับจำนำจะเลือกเอาทรัพย์สินอย่างไร', 'SECTION: [แพ่ง] มาตรา 226\nมาตรา ๒๒๖ บุคคลผู้รับช่วงสิทธิของเจ้าหนี้ ชอบที่จะใช้สิทธิทั้งหลายบรรดาที่เจ้าหนี้มีอยู่โดยมูลหนี้ รวมทั้งประกันแห่งหนี้นั้นได้ในนามของตนเอง\n\nช่วงทรัพย์ ได้แก่เอาทรัพย์สินอันหนึ่งเข้าแทนที่ทรัพย์สินอีกอันหนึ่ง ในฐานะนิตินัยอย่างเดียวกันกับทรัพย์สินอันก่อน'],
    ['หากตั๋วเงินหายไปก่อนกำหนดใช้เงินผู้ทรงตั๋วเงินต้องทำอย่างไร', 'SECTION: [แพ่ง] มาตรา 1011\nมาตรา ๑๐๑๑ ถ้าตั๋วเงินหายไปแต่ก่อนเวลาล่วงเลยกำหนดใช้เงิน ท่านว่าบุคคลซึ่งได้เป็นผู้ทรงตั๋วเงินนั้นจะร้องขอไปยังผู้สั่งจ่ายให้ ๆ ตั๋วเงินเป็นเนื้อความเดียวกันแก่ตนใหม่อีกฉบับหนึ่งก็ได้ และในการนี้ถ้าเขาประสงค์ก็วางประกันให้ไว้แก่ผู้สั่งจ่าย เพื่อไว้ทดแทนที่เขาหากจะต้องเสียหายแก่ผู้หนึ่งผู้ใดในกรณีที่ตั๋วเงินซึ่งว่าหายนั้นจะกลับหาได้\n\nอนึ่ง ผู้สั่งจ่ายรับคำขอร้องดังว่ามานั้นแล้ว หากบอกปัดไม่ยอมให้ตั๋วเงินคู่ฉบับเช่นนั้น อาจจะถูกบังคับให้ออกให้ก็ได้'],
    ['ผ่อนเวลาชำระหนี้แล้วยังค้ำประกันได้ไหม', 'SECTION: [แพ่ง] มาตรา 698\nมาตรา ๖๙๘ อันผู้ค้ำประกันย่อมหลุดพ้นจากความรับผิด ในขณะเมื่อหนี้ของลูกหนี้ระงับสิ้นไปไม่ว่าเพราะเหตุใด ๆ'],
    ['ฟ้องหย่ากรณีโดนสามีทำร้ายร่างกายต้องใช้สิทธิภายในเมื่อไหร่', 'SECTION: [แพ่ง] มาตรา 1528\nมาตรา ๑๕๒๘ ถ้าฝ่ายที่รับค่าเลี้ยงชีพสมรสใหม่ สิทธิรับค่าเลี้ยงชีพย่อมหมดไป'],
]
scores = model.predict(pairs)
print(scores)
# [-5.8377 -3.3165  5.1927 -3.1605 -4.8271]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'ถ้าผู้ให้ตาย ในช่วงที่ยังชำระหนี้เป็นครั้งคราวให้กับผู้รับ หนี้ที่ยังค้างอยู่จะมีผลเป็นยังไง',
    [
        'SECTION: [แพ่ง] มาตรา 525\nมาตรา ๕๒๕ การให้ทรัพย์สินซึ่งถ้าจะซื้อขายกันจะต้องทำเป็นหนังสือและจดทะเบียนต่อพนักงานเจ้าหน้าที่นั้น ท่านว่าย่อมสมบูรณ์ต่อเมื่อได้ทำเป็นหนังสือและจดทะเบียนต่อพนักงานเจ้าหน้าที่ ในกรณีเช่นนี้ การให้ย่อมเป็นอันสมบูรณ์โดยมิพักต้องส่งมอบ',
        'SECTION: [แพ่ง] มาตรา 226\nมาตรา ๒๒๖ บุคคลผู้รับช่วงสิทธิของเจ้าหนี้ ชอบที่จะใช้สิทธิทั้งหลายบรรดาที่เจ้าหนี้มีอยู่โดยมูลหนี้ รวมทั้งประกันแห่งหนี้นั้นได้ในนามของตนเอง\n\nช่วงทรัพย์ ได้แก่เอาทรัพย์สินอันหนึ่งเข้าแทนที่ทรัพย์สินอีกอันหนึ่ง ในฐานะนิตินัยอย่างเดียวกันกับทรัพย์สินอันก่อน',
        'SECTION: [แพ่ง] มาตรา 1011\nมาตรา ๑๐๑๑ ถ้าตั๋วเงินหายไปแต่ก่อนเวลาล่วงเลยกำหนดใช้เงิน ท่านว่าบุคคลซึ่งได้เป็นผู้ทรงตั๋วเงินนั้นจะร้องขอไปยังผู้สั่งจ่ายให้ ๆ ตั๋วเงินเป็นเนื้อความเดียวกันแก่ตนใหม่อีกฉบับหนึ่งก็ได้ และในการนี้ถ้าเขาประสงค์ก็วางประกันให้ไว้แก่ผู้สั่งจ่าย เพื่อไว้ทดแทนที่เขาหากจะต้องเสียหายแก่ผู้หนึ่งผู้ใดในกรณีที่ตั๋วเงินซึ่งว่าหายนั้นจะกลับหาได้\n\nอนึ่ง ผู้สั่งจ่ายรับคำขอร้องดังว่ามานั้นแล้ว หากบอกปัดไม่ยอมให้ตั๋วเงินคู่ฉบับเช่นนั้น อาจจะถูกบังคับให้ออกให้ก็ได้',
        'SECTION: [แพ่ง] มาตรา 698\nมาตรา ๖๙๘ อันผู้ค้ำประกันย่อมหลุดพ้นจากความรับผิด ในขณะเมื่อหนี้ของลูกหนี้ระงับสิ้นไปไม่ว่าเพราะเหตุใด ๆ',
        'SECTION: [แพ่ง] มาตรา 1528\nมาตรา ๑๕๒๘ ถ้าฝ่ายที่รับค่าเลี้ยงชีพสมรสใหม่ สิทธิรับค่าเลี้ยงชีพย่อมหมดไป',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 7,080 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                          | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              | float                                                          |
  | details | <ul><li>min: 5 tokens</li><li>mean: 18.95 tokens</li><li>max: 124 tokens</li></ul> | <ul><li>min: 20 tokens</li><li>mean: 86.32 tokens</li><li>max: 181 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.18</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | label            |
  |:----------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>ถ้าผู้ให้ตาย ในช่วงที่ยังชำระหนี้เป็นครั้งคราวให้กับผู้รับ หนี้ที่ยังค้างอยู่จะมีผลเป็นยังไง</code> | <code>SECTION: [แพ่ง] มาตรา 525<br>มาตรา ๕๒๕ การให้ทรัพย์สินซึ่งถ้าจะซื้อขายกันจะต้องทำเป็นหนังสือและจดทะเบียนต่อพนักงานเจ้าหน้าที่นั้น ท่านว่าย่อมสมบูรณ์ต่อเมื่อได้ทำเป็นหนังสือและจดทะเบียนต่อพนักงานเจ้าหน้าที่ ในกรณีเช่นนี้ การให้ย่อมเป็นอันสมบูรณ์โดยมิพักต้องส่งมอบ</code>                                                                                                                                                                                                                                  | <code>0.0</code> |
  | <code>ในการจำนำทรัพย์สินหลายสิ่งเพื่อประกันหนี้รายเดียว ผู้รับจำนำจะเลือกเอาทรัพย์สินอย่างไร</code>       | <code>SECTION: [แพ่ง] มาตรา 226<br>มาตรา ๒๒๖ บุคคลผู้รับช่วงสิทธิของเจ้าหนี้ ชอบที่จะใช้สิทธิทั้งหลายบรรดาที่เจ้าหนี้มีอยู่โดยมูลหนี้ รวมทั้งประกันแห่งหนี้นั้นได้ในนามของตนเอง<br><br>ช่วงทรัพย์ ได้แก่เอาทรัพย์สินอันหนึ่งเข้าแทนที่ทรัพย์สินอีกอันหนึ่ง ในฐานะนิตินัยอย่างเดียวกันกับทรัพย์สินอันก่อน</code>                                                                                                                                                                                                      | <code>0.0</code> |
  | <code>หากตั๋วเงินหายไปก่อนกำหนดใช้เงินผู้ทรงตั๋วเงินต้องทำอย่างไร</code>                                  | <code>SECTION: [แพ่ง] มาตรา 1011<br>มาตรา ๑๐๑๑ ถ้าตั๋วเงินหายไปแต่ก่อนเวลาล่วงเลยกำหนดใช้เงิน ท่านว่าบุคคลซึ่งได้เป็นผู้ทรงตั๋วเงินนั้นจะร้องขอไปยังผู้สั่งจ่ายให้ ๆ ตั๋วเงินเป็นเนื้อความเดียวกันแก่ตนใหม่อีกฉบับหนึ่งก็ได้ และในการนี้ถ้าเขาประสงค์ก็วางประกันให้ไว้แก่ผู้สั่งจ่าย เพื่อไว้ทดแทนที่เขาหากจะต้องเสียหายแก่ผู้หนึ่งผู้ใดในกรณีที่ตั๋วเงินซึ่งว่าหายนั้นจะกลับหาได้<br><br>อนึ่ง ผู้สั่งจ่ายรับคำขอร้องดังว่ามานั้นแล้ว หากบอกปัดไม่ยอมให้ตั๋วเงินคู่ฉบับเช่นนั้น อาจจะถูกบังคับให้ออกให้ก็ได้</code> | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 2
- `per_device_eval_batch_size`: 16

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 2
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 1.1287 | 500  | 0.3252        |


### Training Time
- **Training**: 33.1 minutes

### Framework Versions
- Python: 3.14.0
- Sentence Transformers: 5.4.1
- Transformers: 5.5.4
- PyTorch: 2.11.0+cpu
- Accelerate: 1.13.0
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->