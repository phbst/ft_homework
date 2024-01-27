import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModel,BitsAndBytesConfig
from transformers import Trainer,TrainingArguments
from peft import get_peft_model, prepare_model_for_kbit_training, TaskType, LoraConfig
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
#配置一下全局超参数
base_model_path="/mnt/data/chatglm3-6b-model"
train_data_path="/mnt/workspace/datasets.csv"
seed=42
max_inputs=512
max_outputs=1536
lora_rank=16
lora_dropout=0.05
lora_alpha=32


#加载数据

dataset=load_dataset("csv",data_files=train_data_path)

#加载分词器


tokenizer=AutoTokenizer.from_pretrained(base_model_path,trust_remote_code=True)


#使用分词器对数据处理，进行分词，且加上特殊符号
def tokenizer_function(example, tokenizer,ignore_lable_id=-100):
    question=example["man"]
    answer=example["wemen"]
    q_ids=tokenizer.encode(question,add_special_tokens=False)
    a_ids=tokenizer.encode(answer,add_special_tokens=False)
    if len(q_ids)>max_inputs-2:
        q_ids=q_ids[:max_inputs-2]
    if len(a_ids)>max_outputs-1:
        a_ids=a_ids[:max_outputs-1]
    inputs_ids=tokenizer.build_inputs_with_special_tokens(q_ids,a_ids)
    question_length=len(q_ids)+2
    inputs_labels=[ignore_lable_id]*question_length+inputs_ids[question_length:]
    return {"input_ids":inputs_ids,"labels":inputs_labels}


tokenized_dataset=dataset["train"].map(lambda example:tokenizer_function(example,tokenizer),batched=False,remove_columns=["wemen","man"])

tokenized_dataset = tokenized_dataset.shuffle(seed=seed)
tokenized_dataset = tokenized_dataset.flatten_indices()


class Data_Collector:
    def __init__(self,pad_token_id:int,max_length:int=2048,ignore_lable_id:int=-100):
        self.pad_token_id=pad_token_id
        self.max_length=max_length
        self.ignore_lable_id=ignore_lable_id
    def __call__(self,batch_data):
        len_list=[len(i["input_ids"]) for i in batch_data]
        batch_max_len=max(len_list)
        input_ids,labels=[],[]
        for len_of_d,d in sorted(zip(len_list,batch_data),key=lambda x:-x[0]):
            pad_len=batch_max_len-len_of_d
            input_id=d["input_ids"]+[self.pad_token_id]*pad_len
            lable=d["labels"]+[self.ignore_lable_id]*pad_len
            if batch_max_len>self.max_length:
                input_id=input_ids[:self.max_length]
                label=lable[:self.max_length]
            input_ids.append(torch.LongTensor(input_id))
            labels.append(torch.LongTensor(lable))
        input_ids=torch.stack(input_ids)
        labels=torch.stack(labels)
        return {"input_ids":input_ids,"labels":labels}
data_collector=Data_Collector(pad_token_id=tokenizer.pad_token_id)



#加载模型
q_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
base_model=AutoModel.from_pretrained(base_model_path,quantization_config=q_config,device_map="auto",trust_remote_code=True)
base_model.supports_gradient_checkpointing = True
base_model.config.use_cache = False

kbit_model=prepare_model_for_kbit_training(base_model)
target_model=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']

lora_config=LoraConfig(
    target_modules=target_model,
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias='none',
    inference_mode=False,
    task_type=TaskType.CAUSAL_LM
)
qlora_model=get_peft_model(kbit_model,lora_config)



#构建训练器
output_dir="phb/chatglm3-ft"
training_args=TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    num_train_epochs=3,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    logging_steps=1,
    save_strategy="steps",
    save_steps=10,
    optim="adamw_torch",
    fp16=True
)
trainer=Trainer(
    model=qlora_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collector
)

trainer.train()
trainer.model.save_pretrained(output_dir)



import torch
from transformers import AutoModel, AutoTokenizer,BitsAndBytesConfig
from peft import PeftModel,PeftConfig
base_model_path="/mnt/data/chatglm3-6b-model"
ft_model_path="/mnt/workspace/phb/chatglm3-ft"
q_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
base_model=AutoModel.from_pretrained(base_model_path,quantization_config=q_config,device_map="auto",trust_remote_code=True)
tokenizer=AutoTokenizer.from_pretrained(base_model_path)
config=PeftConfig.from_pretrained(ft_model_path)
model=PeftModel.from_pretrained(base_model,ft_model_path)

def compare_base_to_ft(q):
    base_response,base_history=base_model.chat(tokenizer,q)

    inputs=tokenizer(q,return_tensors="pt").to(0)
    ft_outputs=model.generate(**inputs)
    ft_response=tokenizer.decode(ft_outputs[0],skip_special_tokens=True)
    print("问题：{}\n".format(q))
    print("basic：{}".format(base_response))
    print("ft:"+ft_response)