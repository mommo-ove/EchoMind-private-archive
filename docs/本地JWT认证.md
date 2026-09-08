# EchoMind 本地 JWT 认证

本方案用于没有学校 OAuth/OIDC 服务时演示可信用户身份。它验证本地 HS256 Bearer Token，并把通过验证的 `sub` 写入 `request.state.principal_id`。它不是学校统一身份认证系统。

## 1. 配置

在 `.env` 中设置：

```dotenv
AUTH_MODE=jwt
JWT_SECRET_KEY=请替换为至少32字节的随机密钥
JWT_ISSUER=echomind-demo
JWT_AUDIENCE=echomind-api
```

不要提交真实密钥。`AUTH_MODE=disabled` 时保留匿名开发行为；`AUTH_MODE=jwt` 时，除 `/health`、`/metrics`、API 文档外的接口都要求有效 Token。

## 2. 重建服务

```powershell
docker compose up -d --build
```

## 3. 创建短期演示 Token

```powershell
$token = docker compose exec -T echomind `
  python tools/create_demo_token.py demo_user_01
```

默认有效期 15 分钟。也可以通过 `--ttl-seconds 3600` 设置为一小时，最长不超过一天。

## 4. 调用聊天接口

```powershell
$headers = @{
  Authorization = "Bearer $token"
  "Idempotency-Key" = [guid]::NewGuid().ToString()
}

$body = @{
  message = "查询我最近7天的校园卡消费"
  user_id = "网页显示用户；不用于授权"
  conv_id = $null
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/chat" `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $body
```

请求体的 `user_id` 只是兼容字段。Redis 记忆、校园卡查询和工单所有权使用 Token 中经过验证的 `sub=demo_user_01`。

## 5. 生产替换

接入学校统一认证时，不应继续由 EchoMind 自己签发 HS256 Token。应改为验证学校身份提供商签发的 JWT：固定允许的算法，通过 OIDC Discovery/JWKS 获取公钥，并校验 `iss`、`aud`、`exp` 和 `sub`。验证后的 `sub` 仍写入 `request.state.principal_id`，因此后续 ChatService、ToolManager 和 SQLite 所有权隔离无需改写。
