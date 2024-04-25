## 介绍
- 在当今信息爆炸的时代，人们对于食谱和烹饪技巧的需求越来越高。然而，传统的食谱查询方式往往繁琐且不够智能化。为了满足用户对食谱的快速、准确查询需求，我们开发了基于 InternLM 的智能食谱问答助手。
- InternLM 是一款在过万亿标记数据上训练的多语千亿参数基座模型。它通过多阶段的渐进式训练，具备较高的知识水平，在中英文阅读理解、推理任务等需要强思维能力的场景下表现优秀。此外，InternLM 还可以在与人类对话时响应复杂指令，并且表现出符合人类道德与价值观的回复。
- 我们的智能食谱问答助手基于 InternLM 对话模型，使用了 XiaChuFang Recipe Corpus 提供的 1,520,327 种中国食谱进行微调。这个项目正在不断开发中，提供的回答仅供参考，不作为正式菜谱的真实制作步骤。
- 通过这一项目，我们旨在为用户提供智能、便捷的食谱查询服务，让大家在烹饪美食的过程中更加得心应手。

### 安装

1. 准备 Python 虚拟环境：

   ```bash
   conda create -n xtunernew python=3.10 -y
   conda activate xtunernew
   ```

2. 克隆该仓库：

   ```shell
   git clone https://github.com/wuyue2247/Recipe_Q-A_Assistant.git
   cd ./Recipe_Q-A_Assistant
   ```

3. 安装Pytorch和依赖库：

   ```shell
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```



### 训练

​		本项目模型 使用 xtuner0.1.13 训练，在 internlm2-chat-7b 上进行微调，
(https://www.modelscope.cn/models/wuyue2247/Recipe_Q-A_Assistant/summary)

1. 微调方法如下

   ```shell
   xtuner train ${YOUR_CONFIG} --deepspeed deepspeed_zero2
   ```

   - `--deepspeed` 表示使用 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 来优化训练过程。XTuner 内置了多种策略，包括 ZeRO-1、ZeRO-2、ZeRO-3 等。如果用户期望关闭此功能，请直接移除此参数。

2. 将保存的 `.pth` 模型（如果使用的DeepSpeed，则将会是一个文件夹）转换为 LoRA 模型：

   ```shell
   export MKL_SERVICE_FORCE_INTEL=1
   xtuner convert pth_to_hf ${YOUR_CONFIG} ${PTH} ${LoRA_PATH}
   ```

   3.将LoRA模型合并入 HuggingFace 模型：

```shell
xtuner convert merge ${Base_PATH} ${LoRA_PATH} ${SAVE_PATH}
```



### 对话

```shell
xtuner chat ${SAVE_PATH} [optional arguments]
```

参数：

- `--prompt-template`: 使用  internlm2_chat。
- `--system`: 指定对话的系统字段。
- `--bits {4,8,None}`: 指定 LLM 的比特数。默认为 fp16。
- `--no-streamer`: 是否移除 streamer。
- `--top`: 建议为0.8。
- `--temperature`: 建议为0.8。
- `--repetition-penalty`: 建议为1.002。
- 更多信息，请执行 `xtuner chat -h` 查看。


```shell
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from tools.transformers.interface import GenerationConfig, generate_interactive

model_name_or_path = "wuyue2247/Recipe_Q-A_Assistant" 

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()

messages = []
generation_config = GenerationConfig(max_length=max_length, top_p=0.8, temperature=0.8, repetition_penalty=1.002)

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "酸菜鱼怎么做", history=history)
print(response)
