from flask import Flask
from config import Config

def create_app(config=Config):
    app = Flask(__name__)
    app.config.from_object(config)
    app.static_folder = 'static'
    return app

app= create_app(Config)
from app import routes





