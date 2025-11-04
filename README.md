# android_qnn_npu
qnn安卓npu环境记录

<img width="1054" height="470" alt="image" src="https://github.com/user-attachments/assets/66579e70-f197-4f3b-8062-825bbfcd0980" />

```
source env.sh
sh run2.sh
```
<img width="1178" height="463" alt="image" src="https://github.com/user-attachments/assets/4e2eb863-ad70-4f96-9798-e5e7ee26e302" />
<img width="1226" height="166" alt="image" src="https://github.com/user-attachments/assets/6b448f6a-e69d-48bb-8720-d0ecfefcff34" />
<img width="1139" height="825" alt="image" src="https://github.com/user-attachments/assets/ba6d072e-34e2-4146-96c2-0e4f473475e1" />
<img width="1280" height="576" alt="image" src="https://github.com/user-attachments/assets/96bfab5d-603f-4fe0-828b-911a23aefe97" />

模型下载(官方更新后不提供so只提供dlc)
```
https://aihub.qualcomm.com/mobile/models/efficientnet_b0
```

hf的绝版so下载
```
https://huggingface.co/qualcomm/EfficientNet-B0/tree/8f26a3264b4d3046860c3ba9f54fd44875c90f39
```
手机ai天梯图
```
https://ai-benchmark.com/ranking.html?from=from_parent_mindnote
```
```
snpe-net-run推理(使用的是dlc)测试gpu最强
```
<img width="1964" height="479" alt="image" src="https://github.com/user-attachments/assets/cf1e7ce0-f220-4284-bb78-394c1ff0f983" />
<img width="1209" height="732" alt="image" src="https://github.com/user-attachments/assets/21f16fdc-2cf8-40c6-9c52-347e992f582f" />
```
qnn-net-run推理(使用的是so或bin)
```
<img width="1623" height="370" alt="image" src="https://github.com/user-attachments/assets/5c89c267-06ee-4539-92d2-eb321045c822" />
<img width="1767" height="187" alt="image" src="https://github.com/user-attachments/assets/37eb8b44-f533-4006-9ad5-6b6d15505340" />

```
云编译so(量化必须用linux)
python -m qai_hub_models.models.efficientnet_b0.export --quantize w8a16 --chipset qualcomm-snapdragon-8gen3 --target-runtime qnn
```
<img width="1211" height="686" alt="image" src="https://github.com/user-attachments/assets/a6418774-3191-4851-89af-7fb4d3293ed4" />
