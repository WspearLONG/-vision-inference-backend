"""create task tables

Revision ID: 20260510_0001
Revises:
Create Date: 2026-05-10 20:50:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260510_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "vision_tasks",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("total", sa.Integer(), nullable=False),
        sa.Column("completed", sa.Integer(), nullable=False),
        sa.Column("result_path", sa.String(length=512), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_vision_tasks_status"), "vision_tasks", ["status"], unique=False)

    op.create_table(
        "task_images",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("path", sa.String(length=512), nullable=False),
        sa.ForeignKeyConstraint(["task_id"], ["vision_tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_task_images_task_id"), "task_images", ["task_id"], unique=False)

    op.create_table(
        "task_artifacts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("path", sa.String(length=512), nullable=False),
        sa.Column("url", sa.String(length=512), nullable=False),
        sa.ForeignKeyConstraint(["task_id"], ["vision_tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_task_artifacts_task_id"), "task_artifacts", ["task_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_task_artifacts_task_id"), table_name="task_artifacts")
    op.drop_table("task_artifacts")
    op.drop_index(op.f("ix_task_images_task_id"), table_name="task_images")
    op.drop_table("task_images")
    op.drop_index(op.f("ix_vision_tasks_status"), table_name="vision_tasks")
    op.drop_table("vision_tasks")

