import logging
from datetime import timedelta


class Config:
    # DEBUG Mode
    DEBUG = True
    # session encryption key
    SECRET_KEY = "fM3PEZwSRcbLkk2Ew82yZFffdAYsNgOddWoANdQo/U3VLZ/qNsOKLsQPYXDPon2t"
    # session expiration time
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)


class DevelopmentConfig(Config):
    # Development mode configuration
    DEBUG = True
    LOG_LEVEL = logging.DEBUG


class ProductionConfig(Config):
    # Online configuration
    # Turn off debugging
    DEBUG = False
    LOG_LEVEL = logging.ERROR  # Log level


# Configuration dictionary, key: configuration
config_dict = {
    'dev': DevelopmentConfig,
    'pro': ProductionConfig
}
