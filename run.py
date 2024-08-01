import os
import subprocess
import glob

batch_size_list = [16,64,128]
seq_len_list = [(128,128),(256,256)] 
model_list =['meta-llama/Llama-2-7b-hf',
             'meta-llama/Llama-2-13b-hf',
             'chargoddard/llama-2-34b-uncode',
             'meta-llama/Llama-2-70b-hf',]


tp_map = {'meta-llama/Llama-2-7b-hf':1,
             'meta-llama/Llama-2-13b-hf':1,
             'chargoddard/llama-2-34b-uncode':2,
             'meta-llama/Llama-2-70b-hf':4,}

bypass_att_config_map = {'meta-llama/Llama-2-7b-hf':"/workspaces/llm_gpu_profile/model_configs/llama_bypass_att_7b_config.json",
             'meta-llama/Llama-2-13b-hf':"/workspaces/llm_gpu_profile/model_configs/llama_bypasss_att_13b_config.json",
             'chargoddard/llama-2-34b-uncode':"/workspaces/llm_gpu_profile/model_configs/llama_bypass_att_34b_config.json",
             'meta-llama/Llama-2-70b-hf':"/workspaces/llm_gpu_profile/model_configs/llama_bypasss_att_70b_config.json",}

work_dir = "/home/wangxinyu/bench"

result_lsit = [] 



def remove_engine_file(directory):
    pattern = os.path.join(directory, "**", "*.engine")
    # 使用 glob 模块找到所有匹配的文件，递归搜索
    files = glob.glob(pattern, recursive=True)
    for file in files:
        os.remove(file)


if __name__ == "__main__":
    for bsz in batch_size_list:
        for seq in seq_len_list:
            work_path = os.path.join(work_dir, f"bsz_{bsz}/",f"seq_{'_'.join(map(str, seq))}/")

            os.makedirs(work_path,exist_ok=True)
            for model in model_list:
                input_len,output_len = seq
                
                #
                normal_commad = f"python /workspaces/llm_gpu_profile/tensorrt_llm_bench/benchmark.py --model {model} -tp {tp_map[model]} --quantization W8A8_SQ_PER_TENSOR_PLUGIN --workspace {work_path} static --isl {input_len} --osl {output_len} --batch {bsz} "
                
                result = subprocess.run(normal_commad,shell=True, text=True)

                # result_lsit.append(result.stdout)

                # 删除 engine 
                remove_engine_file(work_path)

                # 运行 bypass attention 
                bypass_att_command = f"python benchmark.py --model {model+'-bypass-att'} --workspace {work_path} -tp {tp_map[model]} --quantization W8A8_SQ_PER_TENSOR_PLUGIN static --isl {input_len} --osl {output_len} --batch {bsz} --model-config-path {bypass_att_config_map[model]}"

                result = subprocess.run(normal_commad,shell=True, text=True)

                # result_lsit.append(result.stdout)

                remove_engine_file(work_path)



    with open('./run_results','w') as f:
       for line in result_lsit:
           f.write(line+'\n\n\n')

             


                
                
