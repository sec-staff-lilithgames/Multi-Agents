# Keystone (API Key Storage)

This folder is reserved for your local API keys. All agents now use Qwen-Plus as the unified LLM model.

## How to use

### 必需：Qwen-Plus API Key
所有 agent 现在统一使用 qwen-plus 模型。请创建 `keystone/qwen-plus.key` 文件：

1. 复制模板文件：`cp qwen-plus.key.template qwen-plus.key`
2. 编辑 `qwen-plus.key`，将第一行替换为您的千问-plus API Key
3. 获取 API Key：https://dashscope.console.aliyun.com/

### API Key 查找顺序
1. `QWEN_API_KEY` (环境变量)
2. `DASHSCOPE_API_KEY` (环境变量) 
3. `keystone/qwen-plus.key` (文件，推荐)

### 安全注意事项
- 请勿提交您的 API Key 到版本控制
- 此目录已被 git ignore，除了 README 和模板文件
- API Key 文件应该只包含一行：您的密钥

### 旧版本兼容性
旧的 OpenAI 配置已被弃用。所有 agent 现在强制使用 qwen-plus 模型以确保一致性。

