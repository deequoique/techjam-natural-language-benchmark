# TechJam natural-language benchmark

这是一个与 `techjam-conversational-search-main` 完全独立的、标准库实现的
自然语言商品识别 benchmark。它借鉴 EComAgentBench 的“先选目标、再生成用户
需求和模拟回答”的思路，但使用 TechJam 的本地 `catalog.jsonl`，并且始终只
把生成时选中的 `parent_asin` 作为正确答案。相似商品不会算对。

## 边界和数据流

```text
catalog.jsonl ──> generator ──> frozen dataset.jsonl
                                  │
                 validator <─────┘
                                  │  (only query/profile/replies)
                              Agent repo
                                  │
                            exact metrics
```

数据集中的 `target_parent_asin`、签名和候选数是评估器字段；
`project_for_agent()` 不会把它们传给 Agent。Agent 仓库只通过
`--agent-repo` 动态导入 `starter.agent.Agent`，不会写入或复制该仓库。

## 快速开始

需要 Python 3.10+，无第三方依赖：

```bash
cd /path/to/techjam-natural-language-benchmark
python3 -m unittest discover -s tests -v

python3 -m nl_benchmark generate \
  --catalog /path/to/techjam-conversational-search-main/data/catalog.jsonl \
  --samples 8 --seed 42 --output outputs/smoke.jsonl

python3 -m nl_benchmark validate \
  --catalog /path/to/techjam-conversational-search-main/data/catalog.jsonl \
  --dataset outputs/smoke.jsonl

python3 -m nl_benchmark evaluate \
  --agent-repo /path/to/techjam-conversational-search-main \
  --catalog /path/to/techjam-conversational-search-main/data/catalog.jsonl \
  --dataset outputs/smoke.jsonl --output outputs/smoke-results.json
```

`generate` 会以固定 seed 产生可审计 JSONL；普通 `validate` 和 `evaluate` 不会
联网、不重新生成题目。完整 catalog 会占用一定内存（索引和商品记录都在进程
中），但不会被复制进本项目。显式传入 `--scenario` 时，如果该场景无法构造会
直接失败，不会静默改成别的场景；默认样本数达到 7 时会强制检查全部场景覆盖。

## 样本与场景

每个样本包含 target-grounded 的事实签名，事实被分到初始查询、匿名用户画像
和隐藏 clarification slots。签名的完整合取必须在 catalog 中唯一命中目标；初始
信息通常保留多个候选。支持 `direct_search`、`multi_constraint`、
`profile_hidden`、`clarification_required`、`negative_constraint`、
`budget_rating` 和 `intent_override`（只有目录证据足够时才生成）。

模拟器 v2 会同时解释 Agent 的结构化 `ask_attribute` 和自然语言问题，并把问题
约束到 `brand`、`budget`、`feature` 等公开属性。自然语言可以在没有
`ask_attribute` 时独立路由；结构化字段与文本冲突时会返回 `ambiguous`，不会猜测
或泄露事实。它支持同义表达、语义重复、宽泛问题、无偏好、无关问题、信息耗尽及
最大轮数边界。它只能透露预先配置并由目标商品支持的事实，绝不发送 target ID。`intent_override`
会明确构造“旧的错误偏好 -> 新的目标事实”转移；新事实在 override 时更新模拟器
候选状态，override 之前的推荐不会计入 exact 指标。

明确询问已在 query/profile 中提供的属性时，v2 可以用 `reconfirmed` 重述该事实，
但不会改变候选谓词或把它计作新的 clarification；反复换一种说法问同一属性会通过
语义签名判为 `repeated`。宽泛的 `other` 问题最多披露一项匹配的隐藏事实，重复宽泛
追问不会无限吐出答案。结果中的 `protocol.simulator_version` 固定为 `2`，应与旧版
结果分开比较。

评估外部 Agent 时，CLI 使用一次性的 JSONL IPC worker 子进程。父进程不导入
`starter.agent`，目标/签名不进入 worker 的调用栈；worker 使用 `-B`、
`PYTHONDONTWRITEBYTECODE=1` 和临时工作目录，避免在 Agent checkout 写入
`__pycache__`。validator 还会重新检查 query、profile、override 和每个隐藏回复
的文字是否真的包含对应结构化事实，防止手改冻结数据绕过约束。

## 指标

评估只做 exact target scoring：

- `exact_top1`：任一轮目标位于第 1；
- `hit_at_10`：任一轮目标位于 Top-10；
- `mrr`：首次命中时目标排名倒数的平均值；
- `mttc`：首次 Top-10 命中轮数，miss 按 `max_turns + 1` 计；
- 场景分解、澄清轮数、路由/重复/耗尽诊断。

结果 JSON 同时包含每条 trace，便于检查“到底是哪一轮、哪一个自然语言问题”
改变了排序。结果中的 target 仅用于离线评分，不是 Agent 输入。

## 逐轮诊断日志

`evaluate` 会在每轮 trace 的 `diagnostics` 中旁路记录外部 Agent 的可用诊断面，
不改变 Agent 的输入或输出：

- `intent_and_policy`：规则/模型意图路径、触发原因、接受和拒绝字段、路由、追问与提交策略；
- `state`：当前品类、有效约束、查询证据、已问属性和耗尽状态；
- `stages`：原始召回、特征排序、语义排序和最终输出的商品 ID；
- `target_analysis`：由父评测器在 worker 返回后计算的目标商品阶段排名。
- `question_interpretation`：自然语言与 `ask_attribute` 分别识别出的属性、冲突、
  宽泛问题、置信度和稳定语义签名；
- `reply_reason`：披露隐藏事实、重确认画像、语义重复、冲突或真正耗尽等原因。

其中 `target_analysis` 永远不会发送给 worker。常用判断方式：

```text
target_retrieval_rank = null       -> 原始召回未捞到（也要再看 feature_input）
target_feature_input_rank 有值     -> 结构化候选池已把目标补入特征排序输入
target_feature_rank 变成 null      -> 目标被确定性特征排序淘汰
target_semantic_rank 变差           -> 语义重排损伤了目标排序
target_final_rank = null           -> 最终提交/澄清策略没有输出目标
```

意图问题可按 `intent_path -> intent_accepted/intent_rejected -> state.active_constraints`
依次检查，从而区分“模型没调用”“模型结果被过滤”和“约束已保存但检索没使用”。

需要做“每轮强制调用意图模型，并把模型及 mutation 置信度覆盖成 100%”的消融实验时，
可在 `evaluate` 后增加：

```text
--force-intent-model --intent-confidence 1.0
```

这两个开关只修改 worker 进程中的运行时对象，并会写入每轮
`diagnostics.intent_experiment`；不会修改外部 Agent 仓库。该模式用于定位问题，
不代表生产配置或正式基线。

## 设计限制

这是一个针对本地商品域的诊断 benchmark，不是 EComAgentBench 官方 662 道题的
排行榜复现。目录缺少评论字段，因此没有 review-driven 场景。确定性模板优先；
将来若加入 LLM 改写，必须先做事实校验并冻结输出。
