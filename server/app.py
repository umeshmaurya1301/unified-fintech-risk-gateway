import uvicorn
from fastapi import FastAPI
from unified_gateway import UnifiedFintechEnv

app = FastAPI()
env = UnifiedFintechEnv()

@app.post("/reset")
async def reset():
    # The grading script pings this to ensure the env is alive
    obs = env.reset()
    return {"observation": obs}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()