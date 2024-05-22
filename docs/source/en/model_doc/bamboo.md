---
license: apache-2.0
language:
- en
datasets:
- tiiuae/falcon-refinedweb
- bigcode/starcoderdata
- open-web-math/open-web-math
---
## Introducation

Sparse computing is increasingly recognized as an important direction to improve the computational efficiency (e.g., inference speed) of large language models (LLM).

Recent studies ([Zhang el al., 2021](https://arxiv.org/abs/2110.01786); [Liu et al., 2023](https://openreview.net/pdf?id=wIPIhHd00i); [Mirzadeh et al., 2023](https://arxiv.org/abs/2310.04564)) reveal that LLMs inherently exhibit properties conducive to sparse computation when employing the ReLU activation function.
This insight opens up new avenues for inference speed, akin to MoE's selective activation.
By dynamically choosing model parameters for computation, we can substantially boost inference speed.

However, the widespread adoption of ReLU-based models in the LLM field remains limited.
Here we introduce a new 7B ReLU-based LLM, Bamboo (Github link: [https://github.com/SJTU-IPADS/Bamboo](https://github.com/SJTU-IPADS/Bamboo)),
which boasts nearly 85% sparsity and performance levels on par with [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1).

## Model Architecture

To push the model's sparsity, we add a ReLU component after GLU component, called dReLU(double ReLU). So our FFN network works as follows:

```Python
class BambooMLP(nn.Module):                                                                                                                   
    def __init__(self, config):                                                                                                                
        super().__init__()                                                                                                                     
        self.config = config                                                                                                                   
        self.hidden_size = config.hidden_size                                                                                                  
        self.intermediate_size = config.intermediate_size                                                                                      
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)                                                       
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)                                                         
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)                                                       
        self.act_fn = ACT2FN[config.hidden_act]                                                                                                
                                                                                                                                               
    def forward(self, x):                                                                                                                      
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.act_fn(self.up_proj(x)))
```

## Training Details

In this section, we introduce the details of training our model, including types of data used, and hyperparameters.

We initialized the model weights to Mistral's model weights and modified the FFN structure to the dReLU structure, then continued pre-training for 200B tokens, divided into two phases:

**First phase**: For the proportion of training corpus, we followed the data mix ratio and sources of the StableLM-3B model ([link](https://stability.wandb.io/stability-llm/stable-lm/reports/StableLM-3B-4E1T--VmlldzoyMjU4?accessToken=u3zujipenkx5g7rtcj9qojjgxpconyjktjkli2po09nffrffdhhchq045vp0wyfo)), conducting a further pre-training with 150B tokens.

The following table shows the hyper-paramters we used in our training process.

| Hyper-parameters      |             |
| --------------------- | ----------- |
| GPUs                  | 64 80G-A800 |
| Learning Rate Control | Cosine      |
| Peak Learning Rate    | 5e-5        |
| Batch Size            | 4M          |
| Weight Decay          | 0.1         |
| Context Length        | 2k          |

**Second phase**: We further adjusted the training corpus ratio, incorporating more domain-specific datasets (e.g., Math, Coding), and continued training for 50B tokens.

| Hyper-parameters      |             |
| --------------------- | ----------- |
| GPUs                  | 64 80G-A800 |
| Learning Rate Control | Cosine      |
| Peak Learning Rate    | 5e-6        |
| Batch Size            | 4M          |
| Weight Decay          | 0.01        |
| Context Length        | 4k          |

## Performance Evaluation Results

Our evaluation is based on the framework lm-evaluation-harness and opencompass. The evaluation details are listed as follows:

- Huggingface LLM Leaderboard tasks.
- Other Popular Benchmarks: We report the average accuracies on Big Bench Hard (BBH) (3-shot), HumanEval.

|        | Average | MMLU   | Winogrande | TruthfulQA | Hellaswag | GSM8K  | Arc-C  | HumanEval | BBH  | 
| ------- | ------ | ---------- | ---------- | --------- | ------ | ------ | --------- | ---- | ------- |
| Bamboo  | **57.1**  | 63.89 | 76.16     | 44.06     | 82.17    | 52.84 | 62.20 | 25.6     |  50.35    |
| Mistral-v0.1 | **56.5** | 62.65 | 79.24     | 42.62     | 83.32    | 40.18 | 61.43 | 26.21    |   56.35   | 

## Inference Speed Evaluation Results

We utilize [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer), a state-of-the-art acceleration framework leveraging activation sparsity.
Here we show the inference speed compared with llama.cpp/transformers.

## Limitation & Disclaimer

- Bamboo, having undergone training with only 200B tokens, may still exhibit performance gaps in certain tasks. 
- The Bamboo model has only been trained on English-language datasets, hence its capabilities in other languages are still lacking.
- The model may produce unexpected outputs due to its size and probabilistic generation paradigm.

## License

The code is licensed under Apache-2.0, while model weights are fully open for academic research and also allow **free** commercial usage. 

## Citation

Please kindly cite using the following BibTeX:

```
@misc{bamboo,
    title={Bamboo: Harmonizing Sparsity and Performance in Large Language Models}, 
    author={Yixin Song, Haotong Xie, Zeyu Mi, Li Ma, Haibo Chen},
    year={2024}
}
```