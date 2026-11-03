# CBOW 四组模型的评估与词向量导出

本补充包与已有 `train.py` 的检查点格式配套。把压缩包内的文件合并到项目根目录。

| 文件 | 放置位置及职责 |
|---|---|
| `evaluate.py` | 项目根目录，与 `train.py` 同级；负责查找模型、统一运行和保存结果 |
| `evaluation_queries.json` | 项目根目录；固定中英文查询词 |
| `cbow/evaluation.py` | 已有 `cbow` 文件夹内；负责加载向量、计算近邻和导出 |
| `EVALUATION_README.md` | 使用说明 |

保留原有 `data.py`、`model.py`、`training.py`、`train.py`。这个补充包不包含训练模型或语料，也不会改写已有训练结果。

## 1. 这一步比较什么

每种语言分别比较下面四组模型；中英文齐全时共 8 个模型。

| 组别 | epoch | 词向量维度 |
|---|---:|---:|
| A | 10 | 100 |
| B | 20 | 100 |
| C | 30 | 100 |
| D | 30 | 200 |

A/B/C 用于观察训练轮数的影响，C/D 用于观察维度的影响。每个语言使用相同的查询词和相同的候选词表。脚本检查词表以及窗口、批大小、学习率、随机种子等记录是否一致；这些检查不能单独证明原始语料内容完全相同。

脚本从每个模型的 `embedding.weight` 读取输入词向量，使用余弦相似度找出最接近查询词的其他词。它不调用模型训练，也不需要重新构建 CBOW。

这属于**定性分析**。近邻可能体现同义、同类、主题相关或相似上下文，不能把“全部都是同义词”作为唯一标准。余弦相似度不是概率或准确率，不应把不同模型的平均近邻相似度直接用来评选最佳模型。

## 2. 直接运行

在项目根目录的 VSCode PowerShell 终端执行：

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

默认使用 GPU，中文、英文一起处理。程序在项目根目录的 `outputs` 下递归寻找 `model.pt`，并读取同目录中的 `config.json`。模型语言、轮数和维度会再与检查点内部记录核对，因此不需要手工填写时间戳目录，也不用猜上传 CSV 对应哪种语言。

每个模型的目录应保持 `train.py` 原来的保存方式，例如 `outputs/zh/某次运行时间/model.pt` 和同目录的 `config.json`。不要只移动模型而遗漏配置。同组有多次训练时，全部保留并分别标识，不会静默选择最新一份。缺少任一必需实验组时，程序会列出缺失项。

只评估一种语言或显式改用 CPU：

```powershell
.\.venv\Scripts\python.exe evaluate.py --lang zh
.\.venv\Scripts\python.exe evaluate.py --lang en
.\.venv\Scripts\python.exe evaluate.py --device cpu
```

除了训练时已有的 PyTorch，不需要安装新的 Python 包。计算余弦相似度只是向量运算，CPU 也可执行；默认设置仍是 CUDA。

## 3. 先查看哪些结果

每次运行都会新建 `evaluations/运行时间/`，其中包括：

| 文件 | 用途 |
|---|---|
| `comparison.md` | 先看这个：按语言、查询词排列，同一词下对比所有模型的 Top-10 近邻 |
| `model_summary.csv` | 模型编号、语言、轮数、维度、词表规模和真实检查点路径 |
| `neighbors.csv` | 每个近邻一行，适合后续汇总或写报告 |
| `query_status.csv` | 每个查询词的状态和当前语料词频 |
| `evaluation_config.json` | 查询词、Top-k、设备和参与比较的检查点清单 |

`model_summary.csv` 的 `final_epoch_train_loss` 直接来自保存的训练记录，是最后一轮训练过程中累计的平均损失，不是用最终权重重新计算的固定模型损失，也不是验证损失。

查询词不在词表时标记 `oov`；向量为零时标记 `zero_vector`，不会换成其他查询词。候选词仅排除查询词自身和零向量，不删掉不理想的结果，不自动过滤标点、数字或停用词。

固定查询词已在原始作业语料中核对存在，覆盖不同词频。中文的“污染”和英文的 `pollution` 在原始语料中分别仅出现 6 次和 8 次，可以用来观察低频词表现。英文语料中的 `china` 为小写，查询区分大小写。

如需更换查询词，修改 `evaluation_queries.json`，然后让所有实验组重新使用同一套词。保留失败和不理想的案例，避免只挑选好看的结果。

## 4. 导出词向量文件

评估时同时导出每个模型的完整词向量：

```powershell
.\.venv\Scripts\python.exe evaluate.py --export-vectors
```

输出目录下会增加 `vectors` 文件夹，每个模型对应一个 `.txt` 文件。第一行是“词数 维度”，后面每行是“词 向量分量……”，使用 UTF-8 和常见的 word2vec 文本格式。例如，格式示意为：

```text
2 3
词甲 0.1 -0.2 0.3
词乙 -0.4 0.5 0.6
```

上面的数字仅展示文件格式，不是实际模型结果。导出的是模型原始输入词向量；用于余弦计算的归一化副本不会替代它。

原始作业要求提交代码、中文词向量文件、英文词向量文件和实验报告，不要求提交训练好的模型。完成比较后，选择相应配置的中文和英文词向量文件各一份用于提交；通过 `model_summary.csv` 确认文件对应的实验组。

## 5. 如何解释结果

先固定一个查询词，在 A/B/C 中观察近邻列表随轮数的变化，再对比 C/D。记录哪些词更合理、哪些词仍不合理，以及低频词是否更不稳定。保留同一组查询词，避免用不同示例证明不同模型更好。

当前没有带人工评分的词相似度数据，因此不输出 Spearman、准确率或自动“最佳模型”结论。单个随机种子的结果只能作为初步对比；不同维度会改变随机初始化，当前训练实现中也可能影响后续样本打乱顺序。

已有模型使用了给定训练语料。不能从这些模型已见过的语料中再抽一部分，声称它是独立验证集。若后续需要验证损失，应先按文本行/文档划分语料，避免重叠窗口泄漏，再在固定划分上重新训练；也可另用独立的词相似度标注集评估词向量。

## 验证范围与参考

本包已做语法、命令行和非张量部分的检查，并核对了查询词在原始语料中的覆盖情况。交付环境没有 PyTorch，也没有你本地的 `model.pt`，尚未执行实际检查点加载和 GPU 近邻计算；实际结果由你本地运行产生。

PyTorch 文档：

- [torch.load](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
- [normalize](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html)
- [cosine_similarity](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html)
