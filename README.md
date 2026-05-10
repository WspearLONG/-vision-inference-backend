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

## 本地运行

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
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

## 测试

```bash
pytest
```

## 后续路线

- 增加 Redis + Celery，实现视频异步分析任务
- 增加 PostgreSQL，保存任务、图片和推理结果
- 增加 MinIO，保存上传文件和结果图
- 导出 ONNX，接入 Triton Inference Server
- 增加 Prometheus 指标：QPS、P95 延迟、错误率、模型耗时
- 增加鉴权、限流和用户维度的任务隔离

