# Vision Inference Backend

一个面向视觉模型后台开发的练手项目：用 FastAPI 把 YOLO 目标检测模型封装成可部署、可测试、可扩展的推理服务。

## 当前功能

- `GET /health`：服务健康检查
- `POST /api/v1/detect`：上传图片并返回目标检测结果
- YOLO 模型懒加载，首次请求时加载权重
- 统一的请求校验、上传大小限制和错误响应
- Pytest API 测试
- Docker / Docker Compose 部署入口

## 项目结构

```text
app/
  config.py              # 环境变量配置
  main.py                # FastAPI 路由
  schemas.py             # API 响应模型
  services/detector.py   # YOLO 推理封装
tests/
  test_api.py            # API 测试
scripts/
  smoke_request.py       # 手动请求脚本
```

## 环境隔离

本项目使用独立 Conda 环境，环境名为 `vision-inference-backend`。建议每个代码库都维护自己的 `environment.yml`，避免不同项目之间的依赖互相污染。

创建环境：

```bash
conda env create -f environment.yml
```

激活环境：

```bash
conda activate vision-inference-backend
```

如果依赖有变化，更新环境：

```bash
conda env update -f environment.yml --prune
```

## 本地运行

```bash
conda activate vision-inference-backend
uvicorn app.main:app --reload
```

打开接口文档：

```text
http://127.0.0.1:8000/docs
```

## 请求示例

```bash
python scripts/smoke_request.py path/to/image.jpg
```

或者使用 curl：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/detect" ^
  -F "image=@path/to/image.jpg"
```

## Docker 运行

```bash
docker compose up --build
```

开发模式运行，代码改动后容器内服务会自动 reload：

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

该 Compose 配置包含三个服务：

- `api`：FastAPI 接口服务
- `worker`：RQ 后台任务进程
- `redis`：任务队列

## Conda 和 Docker 怎么配合

推荐职责划分：

- Conda：本机开发、调试、跑测试。每个代码库一个独立环境，避免污染电脑里的其他项目。
- Docker：部署、交付、复现线上环境。容器里不依赖你本机 Conda 环境。
- `requirements.txt`：Conda 和 Docker 共用的 Python 依赖来源。
- `environment.yml`：只负责创建本项目的 Conda 开发环境，并通过 `-r requirements.txt` 安装依赖。
- `Dockerfile`：只负责构建容器运行环境，同样安装 `requirements.txt`。

日常开发流程：

```bash
conda activate vision-inference-backend
pytest
uvicorn app.main:app --reload
```

提交前验证容器能跑：

```bash
docker compose up --build
```

## 测试

```bash
pytest
```

## 批量图片异步检测

本地开发时可以只用 Docker 启动 Redis，然后用 Conda 环境分别启动 API 和 worker：

```bash
docker compose up -d redis
conda activate vision-inference-backend
uvicorn app.main:app --reload
```

另开一个终端启动 worker：

```bash
conda activate vision-inference-backend
python scripts/run_worker.py
```

只处理当前队列中的任务并退出：

```bash
python scripts/run_worker.py --burst
```

创建批量检测任务：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/batch-detect" ^
  -F "images=@path/to/image-1.jpg" ^
  -F "images=@path/to/image-2.jpg"
```

查询任务状态：

```bash
curl "http://127.0.0.1:8000/api/v1/tasks/{task_id}"
```

查询任务结果：

```bash
curl "http://127.0.0.1:8000/api/v1/tasks/{task_id}/result"
```

查询带框结果图片：

```bash
curl "http://127.0.0.1:8000/api/v1/tasks/{task_id}/artifacts"
```

结果会保存到：

```text
outputs/{task_id}.json
outputs/{task_id}/images/
```

`artifacts` 接口会返回图片 URL，例如：

```text
http://127.0.0.1:8000/artifacts/{task_id}/images/example.jpg
```

## 后续路线

- 增加 Redis + Celery，实现视频异步分析任务
- 增加 PostgreSQL，保存任务、图片和推理结果
- 增加 MinIO，保存上传文件和结果图
- 导出 ONNX，接入 Triton Inference Server
- 增加 Prometheus 指标：QPS、P95 延迟、错误率、模型耗时
- 增加鉴权、限流和用户维度的任务隔离
