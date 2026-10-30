-- ============================================================
-- 大学生理财知识问答系统 · MySQL 8.0 建表脚本
-- 与 app/models/ 下的 SQLAlchemy 模型一一对应（详见 DESIGN.md 第 4 章）
-- 执行：mysql -u<user> -p < scripts/schema.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS `finance_qa`
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_general_ci;

USE `finance_qa`;

-- ------------------------------------------------------------
-- 1. 用户表
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `user` (
    `id`            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
    `username`      VARCHAR(64)     NOT NULL COMMENT '登录名',
    `email`         VARCHAR(128)    DEFAULT NULL COMMENT '邮箱',
    `password_hash` VARCHAR(128)    NOT NULL COMMENT 'bcrypt 密码哈希',
    `role`          VARCHAR(16)     NOT NULL DEFAULT 'student' COMMENT '角色：student/admin',
    `risk_level`    VARCHAR(16)     DEFAULT NULL COMMENT '风险等级 C1~C5',
    `risk_score`    JSON            DEFAULT NULL COMMENT '风险测评五维得分',
    `status`        TINYINT         NOT NULL DEFAULT 1 COMMENT '1 正常 / 0 禁用',
    `last_login_at` DATETIME(3)     DEFAULT NULL COMMENT '最近登录时间',
    `created_at`    DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at`    DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_user_username` (`username`),
    UNIQUE KEY `uk_user_email` (`email`),
    KEY `idx_user_role` (`role`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户表';

-- ------------------------------------------------------------
-- 2. 会话表
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `conversation` (
    `id`              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`         BIGINT UNSIGNED NOT NULL COMMENT '归属用户',
    `title`           VARCHAR(128)    NOT NULL DEFAULT '新会话' COMMENT '会话标题',
    `kb_ids`          JSON            NOT NULL COMMENT '检索范围：知识库 ID 列表',
    `message_count`   INT             NOT NULL DEFAULT 0 COMMENT '消息条数',
    `last_message_at` DATETIME(3)     DEFAULT NULL COMMENT '最近消息时间',
    `is_deleted`      TINYINT         NOT NULL DEFAULT 0 COMMENT '逻辑删除标记',
    `created_at`      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at`      DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`id`),
    KEY `idx_conv_user` (`user_id`, `last_message_at`),
    CONSTRAINT `fk_conv_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话表';

-- ------------------------------------------------------------
-- 3. 消息表
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `message` (
    `id`               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
    `conversation_id`  BIGINT UNSIGNED NOT NULL COMMENT '所属会话',
    `user_id`          BIGINT UNSIGNED NOT NULL COMMENT '归属用户',
    `role`             VARCHAR(16)     NOT NULL COMMENT 'user / assistant',
    `content`          TEXT            NOT NULL COMMENT '消息正文（Markdown）',
    `citations`        JSON            DEFAULT NULL COMMENT '引用溯源列表',
    `retrieved_chunks` JSON            DEFAULT NULL COMMENT '检索切片快照',
    `latency_ms`       INT             DEFAULT NULL COMMENT '端到端耗时(ms)',
    `token_usage`      INT             DEFAULT NULL COMMENT '生成 token 数',
    `blocked`          TINYINT         NOT NULL DEFAULT 0 COMMENT '是否被风控拦截',
    `feedback`         TINYINT         DEFAULT NULL COMMENT '1 点赞 / -1 点踩',
    `created_at`       DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at`       DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`id`),
    KEY `idx_msg_conv` (`conversation_id`, `id`),
    CONSTRAINT `fk_msg_conv` FOREIGN KEY (`conversation_id`) REFERENCES `conversation` (`id`) ON DELETE CASCADE,
    CONSTRAINT `fk_msg_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='消息表';

-- ------------------------------------------------------------
-- 4. 知识库表
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `knowledge_base` (
    `id`          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
    `name`        VARCHAR(128)    NOT NULL COMMENT '知识库名称',
    `description` VARCHAR(512)    DEFAULT NULL COMMENT '描述',
    `category`    VARCHAR(64)     NOT NULL COMMENT '分类',
    `owner_id`    BIGINT UNSIGNED DEFAULT NULL COMMENT '创建者',
    `doc_count`   INT             NOT NULL DEFAULT 0 COMMENT '文档数',
    `chunk_count` INT             NOT NULL DEFAULT 0 COMMENT '切片数',
    `status`      TINYINT         NOT NULL DEFAULT 1 COMMENT '1 启用 / 0 停用',
    `created_at`  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at`  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_kb_name` (`name`),
    KEY `idx_kb_category` (`category`),
    CONSTRAINT `fk_kb_owner` FOREIGN KEY (`owner_id`) REFERENCES `user` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库表';

-- ------------------------------------------------------------
-- 5. 文档表
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `document` (
    `id`          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
    `kb_id`       BIGINT UNSIGNED NOT NULL COMMENT '所属知识库',
    `name`        VARCHAR(255)    NOT NULL COMMENT '原始文件名',
    `file_path`   VARCHAR(512)    NOT NULL COMMENT '落盘相对路径',
    `file_type`   VARCHAR(16)     NOT NULL COMMENT 'pdf/docx/txt/md/html',
    `file_size`   BIGINT          NOT NULL DEFAULT 0 COMMENT '字节数',
    `checksum`    VARCHAR(64)     DEFAULT NULL COMMENT 'MD5 校验和',
    `status`      VARCHAR(16)     NOT NULL DEFAULT 'pending' COMMENT 'pending/parsing/splitting/embedding/done/failed',
    `progress`    INT             NOT NULL DEFAULT 0 COMMENT '入库进度 0~100',
    `page_count`  INT             NOT NULL DEFAULT 0 COMMENT '解析页数',
    `chunk_count` INT             NOT NULL DEFAULT 0 COMMENT '切片数',
    `error_msg`   VARCHAR(1024)   DEFAULT NULL COMMENT '失败原因',
    `uploaded_by` BIGINT UNSIGNED DEFAULT NULL COMMENT '上传人',
    `created_at`  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at`  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_doc_kb_checksum` (`kb_id`, `checksum`),
    KEY `idx_doc_kb` (`kb_id`, `status`),
    CONSTRAINT `fk_doc_kb` FOREIGN KEY (`kb_id`) REFERENCES `knowledge_base` (`id`) ON DELETE CASCADE,
    CONSTRAINT `fk_doc_uploader` FOREIGN KEY (`uploaded_by`) REFERENCES `user` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文档表';

-- ------------------------------------------------------------
-- 6. 文档切片表
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `document_chunk` (
    `id`          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键（与 Milvus chunk_id 一致）',
    `doc_id`      BIGINT UNSIGNED NOT NULL COMMENT '所属文档',
    `kb_id`       BIGINT UNSIGNED NOT NULL COMMENT '冗余知识库',
    `chunk_index` INT             NOT NULL COMMENT '文档内切片序号',
    `content`     MEDIUMTEXT      NOT NULL COMMENT '切片原文',
    `char_count`  INT             NOT NULL DEFAULT 0 COMMENT '字符数',
    `page_no`     INT             NOT NULL DEFAULT 0 COMMENT '页码',
    `vector_id`   BIGINT UNSIGNED DEFAULT NULL COMMENT 'Milvus 主键',
    `category`    VARCHAR(64)     DEFAULT NULL COMMENT '分类',
    `risk_level`  VARCHAR(16)     NOT NULL DEFAULT 'low' COMMENT 'low/medium/high',
    `created_at`  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at`  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`id`),
    KEY `idx_chunk_doc` (`doc_id`),
    KEY `idx_chunk_kb` (`kb_id`),
    CONSTRAINT `fk_chunk_doc` FOREIGN KEY (`doc_id`) REFERENCES `document` (`id`) ON DELETE CASCADE,
    CONSTRAINT `fk_chunk_kb` FOREIGN KEY (`kb_id`) REFERENCES `knowledge_base` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文档切片表';

-- ------------------------------------------------------------
-- 7. 问答日志表（论文实验数据来源）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `qa_log` (
    `id`                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
    `user_id`             BIGINT UNSIGNED NOT NULL COMMENT '用户 ID',
    `conversation_id`     BIGINT UNSIGNED DEFAULT NULL COMMENT '会话 ID',
    `message_id`          BIGINT UNSIGNED DEFAULT NULL COMMENT '助手消息 ID',
    `question`            TEXT            NOT NULL COMMENT '原始问题',
    `rewritten_query`     TEXT            DEFAULT NULL COMMENT '改写后查询',
    `answer`              TEXT            DEFAULT NULL COMMENT '最终答案',
    `strategy`            VARCHAR(32)     NOT NULL DEFAULT 'hybrid_rerank' COMMENT 'vector/bm25/hybrid/hybrid_rerank',
    `guard_action`        VARCHAR(16)     NOT NULL DEFAULT 'PASS' COMMENT 'BLOCK/REWRITE/PASS',
    `retrieved_chunks`    JSON            DEFAULT NULL COMMENT '召回切片快照',
    `citations`           JSON            DEFAULT NULL COMMENT '引用列表',
    `recall_count`        INT             DEFAULT NULL COMMENT '融合前候选数',
    `fused_count`         INT             DEFAULT NULL COMMENT '融合后数量',
    `rerank_count`        INT             DEFAULT NULL COMMENT '重排后数量',
    `retrieve_ms`         INT             DEFAULT NULL COMMENT '检索耗时',
    `rerank_ms`           INT             DEFAULT NULL COMMENT '重排耗时',
    `generate_ms`         INT             DEFAULT NULL COMMENT '生成耗时',
    `latency_ms`          INT             DEFAULT NULL COMMENT '端到端耗时',
    `prompt_tokens`       INT             DEFAULT NULL COMMENT 'prompt token',
    `completion_tokens`   INT             DEFAULT NULL COMMENT 'completion token',
    `created_at`          DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at`          DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`id`),
    KEY `idx_qa_user_time` (`user_id`, `created_at`),
    KEY `idx_qa_strategy` (`strategy`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='问答日志表';

-- ------------------------------------------------------------
-- 8. 风控规则表
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `risk_rule` (
    `id`          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
    `rule_code`   VARCHAR(32)     NOT NULL COMMENT '规则编码',
    `name`        VARCHAR(128)    NOT NULL COMMENT '规则名称',
    `pattern`     VARCHAR(512)    NOT NULL COMMENT '正则表达式',
    `action`      VARCHAR(16)     NOT NULL COMMENT 'BLOCK / REWRITE / WARN',
    `category`    VARCHAR(32)     NOT NULL COMMENT '类别',
    `replacement` VARCHAR(255)    DEFAULT NULL COMMENT 'REWRITE 改写模板',
    `priority`    INT             NOT NULL DEFAULT 100 COMMENT '数值越小优先级越高',
    `enabled`     TINYINT         NOT NULL DEFAULT 1 COMMENT '启用开关',
    `hit_count`   INT             NOT NULL DEFAULT 0 COMMENT '命中次数',
    `remark`      VARCHAR(512)    DEFAULT NULL COMMENT '备注',
    `created_at`  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at`  DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_rule_code` (`rule_code`),
    KEY `idx_rule_enabled_action` (`enabled`, `action`, `priority`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='风控规则表';
