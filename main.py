import asyncio
from datetime import timedelta, datetime
import json
import jwt
import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, select, delete
from passlib.context import CryptContext

DATABASE_URL = "sqlite+aiosqlite:///./test.db"
SECRET_KEY = "my_secret_key"
ALGORITHM = "HS256"
API_KEY = "827b0f7b297041d271b39146e6b6e755"

engine = create_async_engine(DATABASE_URL, connect_args={"check_same_thread": False})
async_session = async_sessionmaker(engine, expire_on_commit=False)
Base = declarative_base()
redis_client = None

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=15)) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String)


class FavCity(Base):
    __tablename__ = "fav_city"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    city = Column(String, index=True)

class UserCreate(BaseModel):
    username: str
    password: str
    email: EmailStr | None = None


class FavoriteCityCreate(BaseModel):
    city: str


class Weather(BaseModel):
    city: str
    temperature: float
    description: str
    humidity: int


class UserResponse(BaseModel):
    id: int
    city: str

    model_config = {
        "from_attributes": True
    }


app = FastAPI()


async def get_db():
    async with async_session() as session:
        yield session


async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        result = await db.execute(select(User).filter(User.username == username))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="JWT decode error")

async def init_redis():
    global redis_client
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    try:
        await redis_client.ping()
    except redis.ConnectionError:
        print("Redis не запущен или недоступен")



async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.post("/register")
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    username = user.username.lower()
    result = await db.execute(select(User).filter(User.username == username))
    existing_user = result.scalars().first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    new_user = User(username=username, hashed_password=hash_password(user.password))
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return {"message": "User successfully registered"}


@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    username = form_data.username.lower()
    result = await db.execute(select(User).filter(User.username == username))
    user = result.scalars().first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/weather/current/{city}", response_model=Weather)
async def get_weather(city: str):
    if redis_client:
        cached = await redis_client.get(city.lower())
        if cached:
            return json.loads(cached)

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="City not found")
        data = await response.json()

    weather = {
        "city": data["name"],
        "temperature": data["main"]["temp"],
        "description": data["weather"][0]["description"],
        "humidity": data["main"]["humidity"]
    }

    if redis_client:
        await redis_client.set(city.lower(), json.dumps(weather), ex=600)

    return weather


@app.get("/favorites", response_model=list[UserResponse])
async def get_favorites(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    cache_key = f"favorites:{current_user.id}"
    if redis_client:
        cached = await redis_client.get(cache_key)
        if cached:
            return json.loads(cached)

    result = await db.execute(select(FavCity).filter(FavCity.user_id == current_user.id))
    cities = result.scalars().all()
    res = [{"id": c.id, "city": c.city} for c in cities]

    if redis_client:
        await redis_client.set(cache_key, json.dumps(res), ex=300)

    return res


@app.post("/favorites", response_model=FavoriteCityCreate)
async def create_favorite(city: FavoriteCityCreate, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city.city}&appid={API_KEY}&units=metric"
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="City not found")
        data = await response.json()
        valid_city = data["name"]

    result = await db.execute(select(FavCity).filter(FavCity.user_id == current_user.id, FavCity.city == valid_city))
    existing = result.scalars().first()
    if existing:
        raise HTTPException(status_code=400, detail="City already added")

    db_city = FavCity(user_id=current_user.id, city=valid_city)
    db.add(db_city)
    await db.commit()
    await db.refresh(db_city)

    if redis_client:
        await redis_client.delete(f"favorites:{current_user.id}")

    return {"city": db_city.city}


@app.delete("/favorites/{city}")
async def delete_city(city: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    result = await db.execute(delete(FavCity).where(FavCity.user_id == current_user.id).where(FavCity.city == city))
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="City not found")
    await db.commit()

    if redis_client:
        await redis_client.delete(f"favorites:{current_user.id}")

    return {"message": f"City '{city}' successfully deleted"}


@app.on_event("startup")
async def startup_event():
    await init_models()
    await init_redis()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
