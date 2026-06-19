from contextlib import contextmanager
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from ..models import Base


class SessionManager:
    """
    数据库会话管理器

    职责：
    1. 管理数据库连接和会话
    2. 提供统一的会话上下文管理
    3. 处理事务和异常回滚
    """

    def __init__(self, db_path='sqlite:///data/ticket_dispatch.db'):
        """
        初始化会话管理器

        Args:
            db_path: 数据库连接路径
        """
        # 自动创建SQLite数据库目录
        if db_path.startswith("sqlite:///"):
            db_file = db_path.replace("sqlite:///", "")
            db_dir = os.path.dirname(db_file)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

        self.engine = create_engine(db_path)
        Base.metadata.create_all(self.engine)

        # Auto-migration: add missing columns to existing tables (SQLite-friendly)
        self._auto_migrate()

        self.Session = scoped_session(sessionmaker(bind=self.engine))

    def _auto_migrate(self):
        """为已有表添加缺失的列（仅 SQLite 开发阶段，生产用 Alembic）"""
        import logging
        log = logging.getLogger("db.migration")

        migrations = {
            "tickets": [
                ("current_approver", "VARCHAR(64) DEFAULT ''"),
                ("approver_chain", "JSON DEFAULT '[]'"),
                ("history", "JSON DEFAULT '[]'"),
            ],
        }

        with self.engine.connect() as conn:
            for table, columns in migrations.items():
                # 获取现有列名
                existing = {
                    row[1] for row in
                    conn.exec_driver_sql(f"PRAGMA table_info({table})")
                }
                for col_name, col_type in columns:
                    if col_name not in existing:
                        sql = f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                        try:
                            conn.exec_driver_sql(sql)
                            conn.commit()
                            log.info(f"[Migrate] {table}.{col_name} 列已添加")
                        except Exception as e:
                            log.warning(f"[Migrate] 添加 {table}.{col_name} 失败: {e}")

    @contextmanager
    def session_scope(self):
        """
        提供会话上下文管理
        
        自动处理：
        - 会话创建和关闭
        - 事务提交和回滚
        - 异常处理
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self):
        """关闭会话管理器"""
        self.Session.remove()
        self.engine.dispose()
