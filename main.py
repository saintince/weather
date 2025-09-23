from fastapi import FastAPI, Depends, HTTPException, status, Request
from datetime import timedelta, datetime
import jwt
from sqlalchemy.orm import Session, sessionmaker, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from pydantic import BaseModel, EmailStr
from fastapi.security import OAuth2PasswordRequestForm, OAuth2AuthorizationCodeBearer, OAuth2PasswordBearer
from passlib.context import CryptContext
import httpx

DATABASE_URL = "sqlite:///weather.db"
engine = create_engine(DATABASE_URL,connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

SECRET_KEY = "my_secret_key"
ALGORITHM = "HS256"

API_KEY = "YOUR_OPENWEATHER_API_KEY"

pwd_context = CryptContext(schemes=["HS256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=15)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password):
    return pwd_context.hash(password)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401,detail="Could not verify! The username is invalid.")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401,detail="Expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401,detail="JWT error")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    city = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class FavCity(Base):
    __tablename__ = "fav_city"
    id = Column(Integer, primary_key=True, index=True)
    city = Column(String, index=True)

Base.metadata.create_all(bind=engine)

class UserCreate(BaseModel):
    username: str
    email: str = EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Weather(BaseModel):
    city: str
    temperature: float
    description: str
    humidity: int

app = FastAPI()

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    user = User(username=user.username, email= user.email, hashed_password=hash_password(user.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "User successfully registered"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect password")
    token = create_access_token(data={"sub": user.username})
    return {"token": token, "token_type": "bearer"}

@app.get("/weather/current/{city}")
async def get_weather(city: str):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()
        return {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
        }


@app.get("/favorites")
async def get_favorites(db: Session = Depends(get_db)):
    return db.query(FavCity).filter






















