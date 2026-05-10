# Project Log

## 2026-05-10

项目位置：

```text
F:\OpenAI\vision-inference-backend
```

GitHub 仓库：

```text
https://github.com/WspearLONG/-vision-inference-backend
```

Conda 环境：

```text
vision-inference-backend
```

启动方式：

```powershell
cd F:\OpenAI\vision-inference-backend
conda activate vision-inference-backend
uvicorn app.main:app --reload
```

常用入口：

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
```

### 已完成

- FastAPI 服务骨架
- Conda 独立环境
- Docker Compose：`api` / `worker` / `redis` / `postgres`
- 单张图片同步检测：`POST /api/v1/detect`
- 批量图片异步检测：`POST /api/v1/batch-detect`
- 视频异步分析：`POST /api/v1/video-tasks`
- Redis + RQ 后台任务队列
- SQLAlchemy 持久化任务、输入文件、结果图元数据
- Alembic 数据库迁移
- 结果 JSON 保存到 `outputs/{task_id}.json`
- 带框结果图保存到 `outputs/{task_id}/images/`
- 静态结果图访问：`/artifacts/{task_id}/images/{filename}`
- 服务首页：`GET /`
- GitHub Actions CI
- 模型注册表：`models/*.yaml`
- 模型列表接口：`GET /api/v1/models`
- 模型详情接口：`GET /api/v1/models/{model_id}`
- 请求级推理参数：`model_id` / `confidence` / `image_size`
- 视频任务请求级参数：`frame_stride` / `max_frames`

### 已验证

测试：

```text
10 passed
```

编译检查：

```text
python -m compileall app scripts tests
```

真实图片检测验证：

```text
POST /api/v1/detect?model_id=yolov8n&confidence=0.35&image_size=640
结果：200，model=yolov8n，confidence=0.35，image_size=640
```

真实视频任务验证：

```text
video: F:\source\dataset\GARBAGE CLASSIFICATION 3.v1-garbage-classification.yolov8\video\trash.mp4
task_id: 105867cbb7924f46b6a069b2b0267199
status: succeeded
total: 9
completed: 9
```

### 当前模型配置

```text
models/yolov8n.yaml
```

内容用途：

- `id`: 请求时使用的模型 ID
- `path`: 实际模型权重路径
- `default_confidence`: 默认置信度阈值
- `default_image_size`: 默认输入尺寸

### 明天建议继续做

优先方向：继续贴近“视觉模型后端开发工程师”。

建议下一步：

1. 给任务表增加 `model_id`、`confidence_threshold`、`image_size` 字段，并用 Alembic 新增迁移。
2. 增加任务列表接口：`GET /api/v1/tasks`
3. 增加按状态筛选任务：`pending/running/succeeded/failed`
4. 再考虑 MinIO，把上传文件和结果图从本地磁盘迁移到对象存储。

注意事项：

- 如果 `http://127.0.0.1:8000` 还是旧内容，先用 `netstat -ano | Select-String ":8000"` 检查是否有旧 Uvicorn 进程残留。
- 必须在项目目录启动服务，否则会出现 `ModuleNotFoundError: No module named 'app'`。
- 今天 `conda env update` 曾遇到源网络失败，但 PyYAML 已存在，测试通过。

