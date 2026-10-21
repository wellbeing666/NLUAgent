# 智能座舱 Agent

这是一个可运行的 Python 教学项目，提供命令行和本地网页，演示 **规则／本地向量 NLU → 请求路由 → LangChain Agent → 工具执行 → 结果反馈**。车辆、路线、耗电和充电站全部是模拟数据，无真实车辆操作。

项目有两条路径：明确的简单命令由本地规则或语义NLU处理；复杂出行咨询交给支持工具调用的真实模型。另附不需要密钥的固定离线案例，方便先读懂流程。原有命令默认 `--nlu rules`；新增的本地向量版本见下文。

## 本地网页：聊天与执行过程可视化

在当前已准备好模型的环境中运行：

```powershell
Set-Location F:\Agent\NLUAgent
.\.venv\Scripts\python.exe web.py
```

等待本地 BGE 模型加载完成，浏览器打开 <http://127.0.0.1:8765>。终端保持运行，按 `Ctrl+C` 停止。端口占用时可加 `--port 8766`。

网页默认 **离线演示＋规则/BGE 混合 NLU**。左侧可独立切换对话模式与 NLU；中间输入指令；右侧查看模拟车辆状态、意图、参数、匹配示例、相似度、意图分差和工具调用事件。相似度不是正确概率。界面在请求完成后一次性展示结果与事件，目前不做流式输出。

建议演示：

1. `把车内温度设置成二十四度`：本地语义匹配，工具执行后温度显示24℃。
2. `空调调到35度`：共享工具拦截，温度保持不变。
3. `电量只有25%，去机场要不要充电？帮我找个方案`：离线模式展示固定工具链；真实模型模式由配置的Agent处理。
4. `有点热`：澄清目标温度，不自动修改。

首页示例按钮只填入输入框，需要点击发送或按 Enter 才执行；Shift+Enter 换行。切换设置保留车辆和历史，“重置会话”恢复电量25%、温度26℃并清空对话。“刷新”从服务重新读取当前状态，不会再次执行指令。页面刷新后，同一浏览器的Cookie会关联进程内会话；重启服务后清空。普通标签页共享同一浏览器会话，独立浏览器或隐私窗口使用独立状态；超过1小时未使用的会话可能被回收。

没有安装本地模型时可启动纯规则网页；要默认使用已配置的真实模型则指定live：

```powershell
.\.venv\Scripts\python.exe web.py --nlu rules
.\.venv\Scripts\python.exe web.py --mode live --nlu hybrid
```

四种组合与命令行相同，网页默认值和CLI不同。模型配置仍从本地 `.env` 读取，不需要在网页输入密钥。切到真实模式后，未被本地接受的请求及最近上下文会发送给配置的模型服务；缺少配置会明确报错。此次前端验证未请求外部模型服务。

前端不需要安装Node、npm或额外Web框架：`web.py` 用Python标准库提供静态页面与JSON接口，复用 `CabinApp`、NLU和共享工具。只监听 `127.0.0.1`，只提供白名单静态文件，限制请求来源和同会话并发，浏览器不会获得 `.env` 或模型目录。定位为本机教学界面，不是公网服务。

| 文件 | 职责 |
| --- | --- |
| `web.py` | HTTP入口、浏览器会话、聊天/设置/重置接口、事件收集 |
| `frontend/index.html` | 聊天页、模式设置、状态与执行面板 |
| `frontend/app.css` | 桌面和手机响应式样式 |
| `frontend/app.js` | 页面交互、API请求、消息和事件渲染 |
| `frontend/favicon.svg` | 浏览器标签页图标 |
| `tests/test_web.py` | HTTP接口、会话隔离、参数拦截和文件访问回归测试 |

2026-10-21新增前端后，60项测试全部通过，包含10项HTTP测试和显式开启的真实本地BGE断网测试。浏览器验证了语义车控、越界拒绝、缺槽位澄清、离线机场工具链、设置切换与会话保留。

## 本地 BGE NLU：现在可以直接使用

本机已安装可选依赖、下载模型并完成校准。启动后保持进程运行，避免每次问题都重复承担约数秒至十余秒的冷启动：

```powershell
Set-Location F:\Agent\NLUAgent
.\.venv\Scripts\python.exe main.py --mode offline --nlu hybrid
```

建议依次输入：

```text
把车内温度设置成二十四度
现在还有多少电
有点热
把空调温度改成三十五度
空调调到24度？
/state
/reset
/exit
```

前两句应走本地语义识别；“有点热”要求明确目标温度；35℃被共享工具拒绝；疑问句不能触发修改。启用云端Agent时：

```powershell
.\.venv\Scripts\python.exe main.py --mode live --nlu hybrid
```

### 四种组合

| NLU | 对话模式 | 行为 |
| --- | --- | --- |
| `rules` | `offline` | 原规则＋固定离线案例，无模型权重依赖 |
| `rules` | `live` | 原规则＋云端Agent |
| `hybrid` | `offline` | 原规则＋本地BGE＋固定离线案例，正常运行不联网 |
| `hybrid` | `live` | 原规则＋本地BGE，不接受的请求再交给云端Agent |

本地模型未准备好时，显式选择 `hybrid` 会启动失败并提示准备命令，不会静默改成云端。普通启动仍默认 `rules`。

### 新机器安装与模型准备

本地NLU依赖单独锁在 `requirements-nlu.txt`，包含原有Agent锁定版本。建议使用uv，它会明确选择CPU版PyTorch：

```powershell
uv pip install -r requirements-nlu.txt --python .venv\Scripts\python.exe --torch-backend cpu --cache-dir .uv-cache
.\.venv\Scripts\python.exe -m cockpit.nlu_cli prepare
.\.venv\Scripts\python.exe -m cockpit.nlu_cli calibrate
```

也可用pip，但必须提供CPU索引才能安装锁定的 `torch==2.14.1+cpu`：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-nlu.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

模型是 `BAAI/bge-small-zh-v1.5`，固定revision为 `7999e1d3359715c523056ef9478215996d62a620`，存放于 `models/bge-small-zh-v1.5/`。`nlu-model.json` 记录版本与权重SHA256；`prepare`只下载safetensors及必要配置，不下载额外的pickle权重。模型、下载缓存不纳入版本控制。

运行时使用 `device="cpu"`、最多4个PyTorch计算线程、`local_files_only=True`、`trust_remote_code=False`，关闭HF联网与遥测；整个进程共享模型与默认示例向量。只有显式执行 `prepare` 才访问模型下载服务。

### 匹配原理与执行边界

```text
规则完整命中 → 写操作检查 → 共享工具
未命中 → 本地边界检查／缺槽位澄清
       → 温度数字替换成<温度>
       → BGE编码并归一化
       → 与32条示例向量做点积
       → 每个意图取最高分
       → 检查最高分和两个不同意图的分差
       → 温度命令再检查表达、槽位和权限
       → 本地执行／澄清／回到原Agent或离线分支
```

当前支持两个意图：`set_temperature` 和 `query_vehicle`。温度支持一个绝对目标及中文数字，例如二十四、二十四点五；“再低两度”只提示澄清，尚不维护相对调节槽位。

相似度不是概率。负面、假设、引述、疑问、多指令、延后执行、明确其他设备或未支持车辆部件，都有保守检查。规则和语义路径都不能靠高分绕过写检查；Agent工具路径仍无温度写权限。这里是有限规则集，不是能理解所有语义的安全分类器，也不保证能阻止所有自然语言攻击。

`NLU识别`日志会显示来源、候选意图、参数、相似度、不同意图分差、最高分示例、拒识原因和识别耗时。`nlu_guard`表示确定性检查；`semantic_nlu`表示使用了本地向量；规则直接匹配不会计算查询向量。

### 修改示例和校准

数据保存在 `nlu_data/`：

| 文件 | 用途 |
| --- | --- |
| `examples.json` | 2个意图各16条示例；不训练权重，只建立内存向量矩阵 |
| `calibration.json` | 54条校准数据，选择阈值和发现边界问题 |
| `test.json` | 64条保留评测数据，展示泛化与失败案例 |
| `thresholds.json` | 每个意图的similarity、margin及校准来源摘要 |

修改示例后重新执行 `calibrate`，再重启应用。模型版本或示例摘要与校准记录不匹配时，hybrid启动会要求重新校准。修改阈值也需重启，进程内不会热更新。

校准遍历similarity与margin阈值，优先最少误接受，再最多正确本地接受，最后取更严格阈值。similarity为1.01代表关闭该意图的语义接受。当前校准结果：温度 `0.85 / 0.10`，车辆查询 `0.74 / 0.04`。

校准和测试的分组ID与文本不交叉，也检查温度脱敏后的模板不重复；明确标记的数值边界回归案例允许重复规则模板。文本由人工编写，样本量很小、表达可能相近，不能据此宣称真实用户分布下的准确率。

### 评测与测试

```powershell
.\.venv\Scripts\python.exe -m cockpit.nlu_cli evaluate --rules-only
.\.venv\Scripts\python.exe -m cockpit.nlu_cli evaluate
.\.venv\Scripts\python.exe -m cockpit.nlu_cli evaluate --split calibration
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

默认单元测试使用注入的向量，不加载真实权重。单独验证真实CPU推理、两种对话模式与禁止网络连接：

```powershell
$env:RUN_REAL_NLU_TESTS='1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_semantic_nlu.py -v
Remove-Item Env:RUN_REAL_NLU_TESTS
```

`reports/nlu-test.json`包含逐条决策、工具结果、混淆矩阵和误执行次数；`nlu-test-rules.json`为原规则对照；`nlu-threshold-selection.json`记录校准选择。报告不调用云端Agent。

本次完整验证共50项测试全部通过，包含显式开启的真实CPU模型断网测试；日常默认测试会跳过其中1项真实权重测试。

2026-10-21本机64条保留集结果：

| 指标 | 规则模式 | hybrid模式 |
| --- | ---: | ---: |
| 两个业务意图Macro-F1 | 0.1916 | 0.9016 |
| 范围内意图覆盖率（含数值边界案例） | 10.71% | 82.14% |
| 四类决策准确率（含clarify/fallback） | 48.44% | 92.19% |
| 误接受次数 | 0 | 0 |
| 误执行次数 | 0 | 0 |

hybrid仍拒识了5条合法表达，包括部分口语温度设置和查询改写；报告保留原文和分数，不用这些测试样例调阈值。最终复测的预热语义P95约11ms、冷启动约14秒，数值依赖本机环境和运行负载，CLI打印耗时不计入评测推理延迟。首次测试快照保存在 `reports/nlu-test-initial.json`；校准集发现的两条查询误澄清问题修复后复测，不能把反复使用的小数据集视为外部盲测。

指标定义：意图Macro-F1取两个业务意图；四类Macro-F1另外包含clarify/fallback。本地正确覆盖率＝正确执行或澄清的条数／全部样本；范围内意图覆盖率＝两个业务意图中正确接受条数／对应样本数；误执行统计不应写入时成功执行温度工具的次数。分类正确但温度越界时，工具仍必须拒绝。

阅读新增代码时，从 `cockpit/semantic_nlu.py` 的 `NLURouter.recognize()` 看分流，再看 `SemanticNLU.score_many()`、`decide()` 和 `cockpit/nlu.py` 的写检查。`cockpit/nlu_cli.py` 提供准备、校准和评测入口。

**隐私提醒：**本地向量推理不发送文本；`live`中回退到Agent的请求，以及此前保存的本地对话，仍可能作为上下文发往配置的云端服务。本次没有增加完整的隐私网关。

## 1. 现在就运行

在 PowerShell 中执行。当前目录已经安装好独立虚拟环境，可以直接运行，无须激活环境：

```powershell
Set-Location F:\Agent\NLUAgent
.\.venv\Scripts\python.exe main.py --mode offline --demo all
```

依次演示：

1. `空调调到24度`：NLU 提取温度，共享工具校验、写入并读取状态核验。
2. `电量只有25%，去机场要不要充电？帮我找个方案`：固定脚本依次查询车辆、路线、充电站，展示方案。
3. `空调调到35度`：工具拒绝越界参数，温度保持24℃。

进入交互模式：

```powershell
.\.venv\Scripts\python.exe main.py --mode offline
```

可以直接输入 `1`、`2`、`3` 运行对应案例，也可以输入：

```text
空调调到24度
/state
演示出行
空调调到35度
/reset
/state
/exit
```

`/reset` 同时恢复电量25%、温度26℃并清空上下文。退出后所有业务状态消失。

在新机器上安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.in` 声明直接依赖范围；`requirements.txt` 锁定本次验证过的完整依赖。当前已安装的关键版本：Python 3.12.7、LangChain 1.4.6、langchain-openai 1.7.2、python-dotenv 1.2.4。离线入口本身只用 Python 标准库，也可直接 `python main.py --mode offline`。

## 2. 接上真实模型

复制配置示例，使用编辑器填写，不要把真实密钥发到聊天或提交到仓库：

```powershell
Copy-Item .env.example .env
```

`.env` 中需要填写：

```dotenv
OPENAI_API_KEY=填写你的密钥
OPENAI_MODEL=填写服务商提供的模型名称
OPENAI_BASE_URL=https://api.openai.com/v1
```

模型必须支持标准 **Chat Completions + tool calling**。兼容服务使用其对应的 API 根地址，通常包含 `/v1`，不要追加 `/chat/completions`；供应商自定义的推理字段不在本项目支持范围内。系统环境变量优先于 `.env`。不要直接运行示例中的占位值。

```powershell
.\.venv\Scripts\python.exe main.py --mode live
```

依次输入：

```text
空调调到24度
电量只有25%，去机场要不要充电？帮我找个方案
换一条耗电更少的路线
查看车辆状态
/exit
```

第一句走规则，不调用模型；第二句由模型选择工具；第三句带上最近对话与工具结果，让模型理解“路线”指机场路线。真实模型的工具顺序和措辞可能不同，不保证每次都会查询充电站。

单条调用：

```powershell
.\.venv\Scripts\python.exe main.py --mode live --query "去机场有哪些路线？先查询当前车辆状态再比较。"
```

缺少配置、认证失败或服务错误会明确提示，程序不会自动改用离线脚本。交互模式可继续输入本地车控命令，也可修复网络后重试。配置在首次创建模型时读取；如果模型已经创建后修改了 `.env`，请重启程序。

**隐私边界：**offline 不发模型请求；live 会把复杂请求、系统提示、最近10轮上下文和工具结果发送到配置的模型服务。本地 NLU 记录也可能作为后续复杂请求的上下文发出。此版本没有脱敏、个人信息识别或端云隐私网关，请只输入教学数据。

## 3. 先理解这张流程图

```text
用户文字
   ↓
app.handle：先调用规则NLU
   ├─ 完整命中 → 共享工具层校验 → 操作/查询 → 结果反馈
   └─ 未命中
       ├─ offline → 仅匹配预置出行案例 → 固定顺序调用工具
       └─ live → LangChain create_agent
                   模型选择工具 → 工具返回结构化数据
                        ↑               ↓
                        └─ 需要更多信息？ ┘
                          不再调用工具 → 最终回答
```

**离线出行案例是固定脚本，不能证明模型具备自主规划能力。** 在 `app.py` 的 `_offline_trip` 中，工具顺序由代码写死；真实模式中，`create_agent` 根据模型返回的 `tool_calls` 执行工具，直到模型给出最终回答。

### 一次完整的机场查询

本地假设：当前25%电量，到达至少预留10%。

| 方案 | 时间 | 预计到达电量 | 判断 |
| --- | --- | --- | --- |
| 机场高速 | 48分钟 | 25−18＝7% | 低于预留 |
| 机场节能路线 | 65分钟 | 25−12＝13% | 满足预留 |
| 高速＋服务区充电 | 48＋8绕路＋20充电＝76分钟 | 25＋30−18＝37% | 模拟可选备选 |

充电站C01有2个可用端口；节能路线的C02端口为0，应排除。这里的充电时间和恢复电量均为固定教学假设，忽略抵达充电站前的分段耗电与充电曲线，不能用作现实出行或充电建议。没有时限要求时，预设案例选择无需充电的节能路线。

工具查询不改变电量，不会真的导航、预约或充电。日志示意：

```text
[路由] path=langchain_agent
[模型选择工具] get_vehicle_state
[工具调用] get_vehicle_state {}
[工具结果] ok=true, simulated=true, battery_percent=25
[模型选择工具] get_route_options {destination: 机场}
[工具结果] 两条路线及到达电量
[模型选择工具] find_chargers {route_id: airport_fast}
[工具结果] 充电备选方案
助手：比较路线，说明推荐及依据。
```

程序实际输出结构化 JSON，便于核对。只展示工具选择、参数和结果；不会打印模型内部推理内容。模型最后的自然语言回答仍可能出错，面试时应与工具数值核对。

## 4. 代码阅读顺序（约60～90分钟）

| 顺序 | 文件 | 重点看什么 |
| --- | --- | --- |
| 1 | `main.py` | 参数解析、命令行循环、日志、reset/exit |
| 2 | `cockpit/app.py` | 两条路由、离线脚本、会话历史 |
| 3 | `cockpit/nlu.py` | 意图与槽位；为什么用完整匹配 |
| 4 | `cockpit/tools.py` | 四个工具、统一校验、写权限与状态核验 |
| 5 | `cockpit/data.py` | 数据假设与路线计算依据 |
| 6 | `cockpit/agent.py` | create_agent、系统提示、工具循环、错误处理 |

### 关键接口

- `parse_intent(text) -> Intent | None`：返回意图、工具名与参数，未覆盖的表达返回 `None`，没有训练模型或置信度分数。
- `ToolService.execute(name, arguments, allow_write=False) -> dict`：统一入口，返回 `ok`、`simulated` 和 `data` 或 `error`。
- `get_vehicle_state()`：读取电量、温度和预留阈值。
- `set_temperature(temperature)`：校验16～30℃，只有本地明确完整命令拥有写权限。
- `get_route_options(destination)`：只接受“机场”，计算两条路线的到达电量。
- `find_chargers(route_id="")`：支持 `airport_fast`、`airport_eco`，空字符串查询全部站点。
- `LiveAgent.run(text, history)`：运行模型—工具循环，返回最终文本和本轮完整消息。

### 为什么复杂请求不能直接写温度？

本版本故意把写权限收窄：只有本地NLU识别并通过明确表达、槽位和写操作检查的车控指令能传入 `allow_write=True`，包括规则命令和hybrid语义命令。Agent 虽能看见温度工具，但调用会被执行层拒绝。例如“不要把空调调到24度”，即便模型误调用工具，也不能修改温度。“查路线并把空调调到24度”等复合写指令，需要拆开输入。

这是一个可验证的最小权限示例，不是完备的用户授权系统。传给模型的工具参数里没有 `allow_write`，模型不能通过参数自行提权。

### 当前 Harness 做了什么？

Harness 是围绕模型的运行支撑代码。本项目包含工具定义、结构化参数、共享执行校验、状态读回、进程内上下文、日志、错误提示，以及以下边界：

- 单次网络请求超时20秒，关闭 SDK 自动重试；这不是整轮任务的总超时。
- `recursion_limit=12` 限制图执行步数，不等于12次工具调用，也不限制并行工具数量。
- 最多保留最近10个完整对话轮次，不按 token 精确计量；工具结果过长仍可能超上下文。
- 失败只记录用户请求与失败说明，不将未闭合的工具调用消息带入下一轮；保留之前成功的轮次。
- reset 清空调用方传入的历史；本版本没有独立的持久化 checkpointer。

尚未实现：ASR/TTS、训练式NLU、真实地图与车控、权限确认流程、持久化恢复、幂等操作账本、任务取消、多Agent、隐私网关、最终回答事实核验，以及生产级评测。

## 5. 五分钟面试演示与答法

**第1分钟：定位。** “这是一个规则NLU与LLM工具调用协同的座舱教学原型。简单指令本地处理，复杂咨询交给模型，执行校验由统一工具层负责。”

**第2分钟：车控。** 输入“空调调到24度”，指出意图是设置温度、槽位是24；输入“空调调到35度”，展示工具拒绝写入。

**第3分钟：多步任务。** 有密钥时用live展示机场任务；没有密钥时明确使用固定离线脚本，只说明工具链，不宣称模型自主规划。

**第4分钟：上下文和边界。** live追问“换一条耗电更少的路线”；用 `/reset` 展示上下文清空。解释否定指令即使被模型误解，也不能越过温度写权限。

**第5分钟：验证与改进。** 展示测试结果和能力边界，说明下一步会补语音、真实服务、持续评测和可恢复任务。

| 面试问题 | 可以怎样回答 |
| --- | --- |
| NLU与Agent有什么区别？ | 这里的NLU把覆盖范围内的句子映射到意图和槽位；Agent根据目标与工具观察结果动态决定下一步。 |
| 为什么不全用LLM？ | 明确高频指令可以本地执行，减少模型调用依赖；但延迟、准确率和成本收益需要实测，这个版本没有做效果对比实验。 |
| 为什么用LangChain？ | `create_agent` 提供模型—工具循环和标准接口，能专注业务规则。它底层使用LangGraph；本版本没有额外自建外层状态图。 |
| Agent自主性在哪里？ | 真实模式的工具选择与调用顺序由模型决定；边界由执行层约束。离线脚本没有自主性。 |
| 怎样减少幻觉？ | 查询工具提供事实来源，提示词要求基于结果回答，日志可回看；但目前不能保证最终自然语言完全正确。 |
| 工具报成功就够了吗？ | 温度设置后读回实际内存状态核验；真实设备还需要回执、超时和状态查询。 |
| 超时后能恢复任务吗？ | 这里只提示失败并允许发起新请求，没有断点恢复；下一步才考虑持久化状态、幂等和副作用对账。 |
| 项目做了隐私保护吗？ | 密钥不进正常日志，offline不调用模型；live会发送上下文，没有实现端云脱敏，不能声称隐私数据不出端。 |
| 怎样评测？ | 本版测试确定性规则、参数与写权限、数据一致性、上下文和失败行为；后续需要独立任务集、真实模型成功率、延迟及成本评测。 |

## 6. 测试和验证状态

运行全部测试（无密钥、不调用外部模型）：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

也可仅验证不依赖LangChain的业务核心：

```powershell
python -m unittest discover -s tests -p test_core.py -v
```

测试覆盖完整规则匹配、否定与假设不误触发、温度边界和异常类型、写权限、状态核验、路线数值、离线案例、reset、真实模式NLU无需模型、上下文、步数上限、错误信息脱敏与CLI继续输入。

**本次验证记录（2026-10-20）：**

- 24项自动测试通过；离线三个案例和CLI交互流程通过。
- 使用受控的模拟聊天模型，实际运行了 LangChain `create_agent` 的三次连续工具调用及下一轮上下文追问；也验证了模型误调用写工具被拦截。
- 该次初版验证时未配置 `OPENAI_API_KEY` 和 `OPENAI_MODEL`，没有连接真实模型服务。2026-10-21新增验证针对真实本地BGE及NLU接入，未额外请求云端模型服务。

受控模型测试的下一步响应由测试代码预设，只能验证框架与业务接线，不能作为模型推理效果证据。

参考：[LangChain架构选型](https://www.langchain.com/oss-overview)、[create_agent API](https://reference.langchain.com/python/langchain/agents/factory/create_agent)、[ChatOpenAI接口](https://reference.langchain.com/python/langchain-openai/langchain_openai)。
