# EAI6010 Module 5：ULMFiT 文本分类微服务

## 项目概述

本项目将 Assignment 3 中开发的 **ULMFiT (AWD-LSTM) 文本分类模型** 封装为 REST API 微服务并部署到云端。该模型基于 AG News 数据集，可将新闻文本自动分类为以下四个类别：

| 标签 | 类别     |
|------|----------|
| 0    | World（世界新闻）   |
| 1    | Sports（体育）      |
| 2    | Business（商业）    |
| 3    | Sci/Tech（科技）    |

---

## 执行计划

### 第一阶段：模型训练与导出

**目标**：复现 Assignment 3 中的 ULMFiT 模型，并导出为可部署的文件。

1. 安装 fastai 及相关依赖
2. 下载 AG News 数据集
3. 在 AG News 文本上微调 AWD-LSTM 语言模型
4. 使用微调后的编码器训练文本分类器
5. 通过 `learn.export()` 将训练好的分类器导出为 `.pkl` 文件

**产出物**：`models/ag_news_classifier.pkl`

---

### 第二阶段：构建微服务（FastAPI）

**目标**：创建一个 REST API，接收文本输入并返回预测的新闻类别。

1. 创建 `app.py` — FastAPI 应用，包含以下端点：
   - `GET /` — 健康检查 / 欢迎信息
   - `GET /docs` — 自动生成的 Swagger 交互文档（FastAPI 内置）
   - `POST /predict` — 接收新闻文本，返回预测类别和置信度
2. 在服务启动时加载导出的 fastai 模型
3. 使用 Pydantic 定义输入输出数据模型

**服务输入**（POST /predict）：

```json
{
  "text": "NASA launches new Mars exploration mission with advanced rover technology"
}
```

**服务输出**：

```json
{
  "prediction": "Sci/Tech",
  "confidence": 0.9523,
  "label_index": 3,
  "probabilities": {
    "World": 0.0102,
    "Sports": 0.0054,
    "Business": 0.0321,
    "Sci/Tech": 0.9523
  }
}
```

**关键文件**：
- `app.py` — FastAPI 应用入口
- `schemas.py` — Pydantic 请求/响应数据模型
- `model.py` — 模型加载与推理逻辑

---

### 第三阶段：本地测试

**目标**：在部署前验证服务的正确性。

1. 使用 `uvicorn app:app --reload` 在本地启动服务
2. 通过 Swagger UI（`http://localhost:8000/docs`）进行交互式测试
3. 通过 curl 发送测试请求
4. 验证四个类别的预测结果是否正确

---

### 第四阶段：容器化（Docker）

**目标**：将服务打包为 Docker 容器，便于部署。

1. 基于 Python 3.10 slim 镜像编写 `Dockerfile`
2. 创建 `requirements.txt`，锁定依赖版本
3. 创建 `.dockerignore` 排除不必要的文件
4. 在本地构建并测试容器

**关键文件**：
- `Dockerfile`
- `requirements.txt`
- `.dockerignore`

---

### 第五阶段：云端部署

**目标**：将容器化的服务部署到云平台，获得可公开访问的 URL。

**首选方案：Render（免费版）**
- 将 GitHub 仓库连接到 Render
- 配置为 Web Service，使用 Docker 运行时
- 根据需要设置环境变量
- 验证公开 URL 可正常访问

**备选方案：Hugging Face Spaces**
- 如果 Render 的免费版遇到限制，转用 HF Spaces 部署（支持 Gradio/Docker）
- 记录部署过程中遇到的任何错误

> **注意**：根据作业说明，如果在免费账户部署过程中遇到错误，需在作业文档中记录这些错误，并尽可能继续完成部署。

---

### 第六阶段：撰写作业文档

**目标**：完成最终提交的作业文档（`assignment5.md`）。

文档需包含以下内容：
1. **服务描述** — 微服务的功能说明
2. **通用输入/输出说明** — 请求和响应的格式与数据结构
3. **具体示例** — 至少 3 个实际的请求与响应示例（覆盖不同类别）
4. **服务 URL** — 已部署服务的公开访问地址
5. **部署说明** — 部署过程中遇到的问题及解决方案

---

## 项目结构（规划）

```
EAI6010/
├── README.md                  # 本文件 — 项目计划
├── app.py                     # FastAPI 应用入口
├── model.py                   # 模型加载与推理逻辑
├── schemas.py                 # Pydantic 请求/响应数据模型
├── train_model.py             # 模型训练与导出脚本
├── models/
│   └── ag_news_classifier.pkl # 导出的 fastai 模型
├── requirements.txt           # Python 依赖
├── Dockerfile                 # 容器定义
├── .dockerignore              # Docker 忽略文件
├── assignment3.md             # Assignment 3 报告（参考）
└── assignment5.md             # Assignment 5 报告（最终提交物）
```

---

## 技术栈

| 组件       | 技术方案               |
|-----------|------------------------|
| 机器学习框架 | fastai 2.x + PyTorch  |
| 模型       | ULMFiT (AWD-LSTM)     |
| 数据集     | AG News（4 分类）       |
| Web 框架   | FastAPI + Uvicorn      |
| 容器化     | Docker                  |
| 部署平台   | Render / HF Spaces     |
| API 文档   | Swagger UI（自动生成）   |

---

## 快速启动（本地）

```bash
# 安装依赖
pip install -r requirements.txt

# 训练并导出模型（仅需执行一次）
python train_model.py

# 启动服务
uvicorn app:app --host 0.0.0.0 --port 8000

# 测试请求
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple stock rises after strong quarterly earnings report"}'
```

---

## 进度追踪

| 阶段 | 任务               | 状态   |
|------|--------------------|--------|
| 1    | 模型训练与导出      | 待开始 |
| 2    | 构建 FastAPI 服务   | 待开始 |
| 3    | 本地测试            | 待开始 |
| 4    | Docker 容器化       | 待开始 |
| 5    | 云端部署            | 待开始 |
| 6    | 撰写作业文档        | 待开始 |
