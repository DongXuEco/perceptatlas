# PerceptAtlas：全球城市情绪宜居性

[English](README.md) · [在线平台](https://perceptatlas.com/) · [完整代码与数据包](https://github.com/DongXuEco/perceptatlas/releases/tag/v1.0.0) · [图号与代码索引](docs/FIGURES.md)

[![PerceptAtlas 在线平台](docs/assets/platform-home.png)](https://perceptatlas.com/)

本仓库对应手稿 **Quantifying and mapping global emotional livability**，提供图表的数值来源、数据分析代码与实际绘图代码，并展示交互式网页平台。

研究涵盖 **5,880 个城市、6 个感知维度，以及 4,568 对核心区—扩张区观测**。六个维度为美感、富裕、安全、活力、无聊与压抑。这些分数表示模型预测的环境感知，不代表居民的心理健康测量。网站目录中有 5,905 条地理记录，与正文有效研究样本量有所区别。

## 获取数据与复现

在 [v1.0.0 Release](https://github.com/DongXuEco/perceptatlas/releases/tag/v1.0.0) 下载附件 **perceptatlas-research-v1.0.0.zip**。其中包含完整的 `code/` 与 188 个数据文件。GitHub 自动生成的 Source code 压缩包不包含完整数据。

解压后进入 `perceptatlas/`，使用 Python 3.12：

```bash
python -m pip install -r code/requirements.txt
python code/figures/Fig_2.py --check-inputs
python code/figures/Fig_2.py --formats png pdf
python code/run_analysis.py --list
```

绘图输出位于 `outputs/`。`code/figures/` 保留按最终图号命名的实际绘图实现；`code/analysis/` 存放处理、统计、验证及环境关联模型代码；`code/_support/` 存放共用函数。

## 在线平台

| 地图与位置查询 | 同步双地球 |
|---|---|
| ![地图查询](docs/assets/platform-map.png) | ![同步双地球](docs/assets/platform-dual-globe.png) |

平台支持感知图层展示、城市比较、分布分析与数值导出。网页应用与栅格服务运行在 [perceptatlas.com](https://perceptatlas.com/)，本仓库展示其入口与截图，不包含网站后端或部署配置。

## 公开范围与限制

包含处理后的图表数据、共享分析表、分析及绘图代码。不包含原始街景影像、原始栅格产品、下载代码、感知评分与空间预测流程、服务器密码或部署配置。

已有记录中，12 个图号通过随包数据验证；其余 14 个图号依赖外部输入或配置。图 1 为人工组合的概念图，仅提供插画支持函数；补充图 S8 缺少原始步行配置。不能将本包表述为全部结果均可独立从原始数据复现。详见 [复现说明](docs/REPRODUCIBILITY.md) 和 [图号索引](docs/FIGURES.md)。

代码采用 [MIT](LICENSE)，作者生成的数据采用 [CC BY 4.0](LICENSE_DATA.md)，第三方材料遵循 [原有许可](NOTICES.md)。引用信息见 [CITATION.cff](CITATION.cff)。
