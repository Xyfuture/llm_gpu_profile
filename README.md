Run Command

 ```bash
 python benchmark.py --model EleutherAI/gpt-j-6b static --isl 128 --osl 128 --batch 1
 ```

quantization

```bash
python benchmark.py --model EleutherAI/gpt-j-6b --quantization W8A8_SQ_PER_CHANNEL --kv-dtype float16 static --isl 128 --osl 128 --batch 1
```


使用这个，启用的是 Smooth Quant的 Kernel ， INT8运算
```bash
python benchmark.py --model EleutherAI/gpt-j-6b --quantization W8A8_SQ_PER_TENSOR_PLUGIN --kv-dtype float16 static --isl 256 --osl 256 --batch 64
```




```bash
trtllm-build 
```





hugging face access

```bash
huggingface-cli login --token my_token
```



hugging face models 

- [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [`chargoddard/llama-2-34b-uncode`](https://huggingface.co/chargoddard/llama-2-34b-uncode/)
- [`meta-llama/Llama-2-13b-hf`](https://huggingface.co/meta-llama/Llama-2-13b-hf)
- [`meta-llama/Llama-2-70b-hf`](https://huggingface.co/meta-llama/Llama-2-70b-hf)

