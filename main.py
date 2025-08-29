from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import openai
import os
import json
import uuid
import requests
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, List, Dict

# 加载环境变量
load_dotenv()

# 设置各种AI服务的API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
BAICHUAN_API_KEY = os.getenv("BAICHUAN_API_KEY")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")

# AI服务提供商配置
AI_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        "api_key_env": "OPENAI_API_KEY"
    },
    "claude": {
        "name": "Anthropic Claude",
        "models": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"],
        "api_key_env": "CLAUDE_API_KEY",
        "base_url": "https://api.anthropic.com/v1/messages"
    },
    "gemini": {
        "name": "Google Gemini",
        "models": ["gemini-pro", "gemini-pro-vision"],
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models"
    },
    "qwen": {
        "name": "阿里云通义千问",
        "models": ["qwen-turbo", "qwen-plus", "qwen-max"],
        "api_key_env": "QWEN_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    },
    "baichuan": {
        "name": "百川智能",
        "models": ["Baichuan2-Turbo", "Baichuan2-Turbo-192k"],
        "api_key_env": "BAICHUAN_API_KEY",
        "base_url": "https://api.baichuan-ai.com/v1/chat/completions"
    },
    "zhipu": {
        "name": "智谱AI",
        "models": ["glm-4", "glm-3-turbo"],
        "api_key_env": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    },
    "moonshot": {
        "name": "月之暗面 Kimi",
        "models": [
            "kimi-k2-0711-preview",
            "kimi-k2-turbo-preview",
            "moonshot-v1-8k", 
            "moonshot-v1-32k", 
            "moonshot-v1-128k",
            "moonshot-v1-8k-vision-preview",
            "moonshot-v1-32k-vision-preview",
            "moonshot-v1-128k-vision-preview",
            "kimi-latest", 
            "kimi-thinking-preview"
        ],
        "api_key_env": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.cn/v1/chat/completions"
    },
    "local": {
        "name": "本地模拟AI (测试用)",
        "models": ["local-test"],
        "api_key_env": None,
        "base_url": None
    }
}

def call_ai_api(provider: str, model: str, messages: List[Dict], temperature: float, max_tokens: int):
    """调用不同AI服务提供商的API"""
    
    if provider == "openai":
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    elif provider == "claude":
        headers = {
            "x-api-key": CLAUDE_API_KEY,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # 转换消息格式
        claude_messages = []
        system_message = ""
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                claude_messages.append({"role": msg["role"], "content": msg["content"]})
        
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_message,
            "messages": claude_messages
        }
        
        response = requests.post(AI_PROVIDERS["claude"]["base_url"], headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["content"][0]["text"]
        else:
            raise Exception(f"Claude API错误: {response.text}")
    
    elif provider == "qwen":
        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(AI_PROVIDERS["qwen"]["base_url"], headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"通义千问API错误: {response.text}")
    
    elif provider in ["baichuan", "zhipu", "moonshot"]:
        if provider == "baichuan":
            api_key = BAICHUAN_API_KEY
        elif provider == "zhipu":
            api_key = ZHIPU_API_KEY
        else:  # moonshot
            api_key = MOONSHOT_API_KEY
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(AI_PROVIDERS[provider]["base_url"], headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"{AI_PROVIDERS[provider]['name']}API错误: {response.text}")
    
    elif provider == "local":
        # 本地模拟AI响应，用于测试和备用
        import time
        time.sleep(1)  # 模拟API延迟
        user_message = messages[-1]["content"] if messages else "你好"
        return f"这是一个本地模拟的AI响应。您说：'{user_message}'。这个功能可以在外部API无法连接时使用，确保系统基本功能正常。您可以在网络问题解决后切换回其他AI服务商。"
    
    else:
        raise Exception(f"不支持的AI服务提供商: {provider}")

# 创建FastAPI应用
app = FastAPI(title="我的对话智能体")

# 添加CORS中间件（非常重要！允许网页前端调用这个API）
# CORS配置 - 根据环境变量调整
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
if allowed_origins == ["*"]:
    # 开发环境允许所有来源
    cors_origins = ["*"]
else:
    # 生产环境使用指定域名
    cors_origins = [origin.strip() for origin in allowed_origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# 定义请求体模型
class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict[str, str]] = []  # 前端传递对话历史
    system_prompt: Optional[str] = "你是一个乐于助人且幽默的AI助手。"
    model: Optional[str] = "gpt-3.5-turbo"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    session_id: Optional[str] = None
    provider: Optional[str] = "openai"  # 新增：AI服务提供商选择

class SessionRequest(BaseModel):
    session_name: Optional[str] = None

# 内存存储 - 适用于无服务器环境
# 注意：重启服务器后数据会丢失，这是无服务器环境的正常行为
SESSIONS_STORAGE = {}

def load_sessions():
    """从内存加载所有会话数据"""
    return SESSIONS_STORAGE

def save_sessions(sessions):
    """保存会话数据到内存"""
    global SESSIONS_STORAGE
    SESSIONS_STORAGE = sessions

def save_conversation(session_id, message, reply, system_prompt, model_config):
    """保存对话到会话"""
    sessions = load_sessions()
    
    if session_id not in sessions:
        sessions[session_id] = {
            'id': session_id,
            'name': f'会话 {len(sessions) + 1}',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'system_prompt': system_prompt,
            'model_config': model_config,
            'messages': []
        }
    
    sessions[session_id]['messages'].extend([
        {'role': 'user', 'content': message, 'timestamp': datetime.now().isoformat()},
        {'role': 'assistant', 'content': reply, 'timestamp': datetime.now().isoformat()}
    ])
    sessions[session_id]['updated_at'] = datetime.now().isoformat()
    
    save_sessions(sessions)
    return sessions[session_id]

# 提供静态文件服务
@app.get("/")
async def read_index():
    return FileResponse("jointown_style.html")

@app.get("/jointown_style")
async def read_jointown_style():
    return FileResponse("jointown_style.html")

@app.get("/jointown")
async def read_jointown():
    return FileResponse("jointown_style.html")

@app.get("/about_chuangmei")
async def read_about_chuangmei():
    return FileResponse("about_chuangmei.html")

@app.get("/news")
async def read_news():
    return FileResponse("news.html")

@app.get("/图片1.png")
async def read_logo_image():
    return FileResponse("图片1.png")

# API状态检查
@app.get("/api/status")
async def api_status():
    return {"message": "智能体API运行成功！", "status": "online"}

# 会话管理端点
@app.post("/api/sessions")
async def create_session(request: SessionRequest):
    """创建新会话"""
    session_id = str(uuid.uuid4())
    sessions = load_sessions()
    
    sessions[session_id] = {
        'id': session_id,
        'name': request.session_name or f'会话 {len(sessions) + 1}',
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'system_prompt': '你是一个乐于助人且幽默的AI助手。',
        'model_config': {
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'max_tokens': 1000
        },
        'messages': []
    }
    
    save_sessions(sessions)
    return {"session_id": session_id, "session": sessions[session_id]}

@app.get("/api/sessions")
async def get_sessions():
    """获取所有会话列表"""
    sessions = load_sessions()
    session_list = [{
        'id': session['id'],
        'name': session['name'],
        'created_at': session['created_at'],
        'updated_at': session['updated_at'],
        'message_count': len(session['messages'])
    } for session in sessions.values()]
    
    return {"sessions": sorted(session_list, key=lambda x: x['updated_at'], reverse=True)}

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """获取特定会话的详细信息"""
    sessions = load_sessions()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    return {"session": sessions[session_id]}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    sessions = load_sessions()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    del sessions[session_id]
    save_sessions(sessions)
    return {"message": "会话已删除"}

@app.get("/api/providers")
async def get_ai_providers():
    """获取所有支持的AI服务提供商"""
    providers_info = []
    for provider_id, config in AI_PROVIDERS.items():
        api_key_env = config.get("api_key_env")
        
        # 本地模拟AI始终可用
        if provider_id == "local":
            has_api_key = True
            status = "可用 (无需网络)"
        else:
            has_api_key = api_key_env and os.getenv(api_key_env) is not None
            status = "可用" if has_api_key else "需要API密钥"
        
        providers_info.append({
            "id": provider_id,
            "name": config["name"],
            "models": config["models"],
            "has_api_key": has_api_key,
            "status": status
        })
    
    return {"providers": providers_info}

# 定义聊天端点
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 如果没有提供session_id，创建一个新的
        session_id = request.session_id or str(uuid.uuid4())
        
        # 构建消息列表，始终以系统消息开头
        messages = [{"role": "system", "content": request.system_prompt}]
        
        # 添加历史对话
        messages.extend(request.conversation_history)
        
        # 添加最新的一条用户消息
        messages.append({"role": "user", "content": request.message})

        # 调用AI API
        assistant_reply = call_ai_api(
            provider=request.provider,
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # 保存对话到会话
        model_config = {
            'model': request.model,
            'temperature': request.temperature,
            'max_tokens': request.max_tokens
        }
        
        session = save_conversation(
            session_id, 
            request.message, 
            assistant_reply, 
            request.system_prompt, 
            model_config
        )

        # 返回响应
        return {
            "reply": assistant_reply,
            "status": "success",
            "session_id": session_id,
            "session": session
        }
        
    except Exception as e:
        print(f"Chat endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-qwen")
async def test_qwen():
    """测试通义千问API连接"""
    try:
        import requests
        
        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "qwen-plus",
            "messages": [{"role": "user", "content": "你好"}]
        }
        
        print(f"API Key: {QWEN_API_KEY[:10]}...{QWEN_API_KEY[-5:]}")
        print(f"请求URL: {AI_PROVIDERS['qwen']['base_url']}")
        print(f"请求数据: {data}")
        
        response = requests.post(
            AI_PROVIDERS["qwen"]["base_url"],
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        if response.status_code == 200:
            return {"status": "success", "response": response.json()}
        else:
            return {"status": "error", "code": response.status_code, "message": response.text}
            
    except Exception as e:
        print(f"测试异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "exception", "error": str(e)}

@app.get("/test-moonshot")
async def test_moonshot():
    """测试月之暗面API连接"""
    try:
        print(f"月之暗面API Key: {MOONSHOT_API_KEY[:10]}..." if MOONSHOT_API_KEY else "未设置")
        
        headers = {
            "Authorization": f"Bearer {MOONSHOT_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "moonshot-v1-8k",
            "messages": [
                {"role": "user", "content": "你好，请简单介绍一下你自己。"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        print(f"请求URL: https://api.moonshot.cn/v1/chat/completions")
        print(f"请求头: {headers}")
        print(f"请求数据: {data}")
        
        response = requests.post(
            "https://api.moonshot.cn/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        if response.status_code == 200:
            return {"status": "success", "response": response.json()}
        else:
            return {"status": "error", "code": response.status_code, "message": response.text}
            
    except Exception as e:
        print(f"测试异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "exception": str(e)}

# 如果要本地测试，运行：uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    
    # 从环境变量获取配置
    port = int(os.getenv("PORT", 8000))
    environment = os.getenv("ENVIRONMENT", "development")
    
    # 根据环境调整配置
    if environment == "production":
        # 生产环境配置
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info",
            access_log=True
        )
    else:
        # 开发环境配置
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            reload=True,
            log_level="debug"
        )