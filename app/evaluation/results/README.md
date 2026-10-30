# 实验结果目录说明

> 该目录下的 CSV/PNG 均为脚本真实输出（文件名带时间戳，不覆盖历史）。
> 实验脚本：`scripts/experiment_compare.py`、`scripts/experiment_chunk.py`。

## 有效结果（论文可用）

| 文件前缀 | 模式 | 样本 | 说明 |
|---|---|---|---|
| `compare_offline_*_full60` | offline | 60 | 哈希向量 + BM25 + RRF 的离线复现（可无模型运行），用于趋势与脚本自检 |
| `compare_service_*_c360fix` | service | 60 | **最终版**：360 卡片 / 375 切片细粒度语料；按分类检索（每库约 60 切片） |
| `compare_service_*_c360all` | service | 60 | **最终版**：同一细粒度语料的全库检索（375 切片） |
| `compare_service_*_m3fix` | service | 60 | **最终版**：bge-m3(1024) + bge-reranker-v2-m3 + DeepSeek；按分类检索（生产配置） |
| `compare_service_*_m3all` | service | 60 | **最终版**：bge-m3 + 真实重排；全库混合语料 + `RECALL_TOP_K=5`、`RERANK_TOP_N=3` |
| `compare_service_*_svc60fix` | service | 60 | 基线：bge-small-zh(512) + 词重叠代理重排；按分类检索（用于模型升级对比） |
| `compare_service_*_svc60all` | service | 60 | 基线：bge-small-zh + 代理重排；全库混合语料（用于模型升级对比） |
| `chunk_grid_*.csv` | offline | 60 | 切片参数网格实验 |

`c360fix` / `c360all` / `m3fix` / `m3all` 的 `reranker` 列为 `bge-reranker`，D 组使用真实交叉编码器；
`svc60fix` / `svc60all` 为 bge-small 基线（`lexical-proxy`），用于说明嵌入模型升级带来的变化。

> 语料说明：`c360*` 使用 `data/corpus/cards.jsonl`（60 条标准答案 × 6 张卡片）构建的细粒度语料，
> 363 个文档 / 375 个切片；`m3*` / `svc60*` 使用早期的"每类一篇知识汇编"语料（27 切片）。
> 对比表 6-3 说明了两者指标差异的成因。

## 无效结果（仅保留作排查记录，勿引用）

| 文件前缀 | 原因 |
|---|---|
| `compare_service_*_svc60_*` | 评测集 `kb_id` 与种子知识库自增 ID 不一致（股票/保险/信用三类错配），召回率被低估 |
| `compare_service_*_e2e_svc*` | 同上，且仅 4 条样本（链路 smoke） |
| `compare_service_*_timing*` / `*_fixcheck*` | 单次计时/映射自检，非实验数据 |

> 该问题已修复为「按分类名解析知识库 ID」（`app/evaluation/runner.py: resolve_category_kb_ids`），
> 修复后 `svc60fix` 的召回率由 0.56 恢复到 0.98。
