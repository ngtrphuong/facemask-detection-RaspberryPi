# -*- coding: utf-8 -*-
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from config import config_dict


# Set log (the purpose is to save flask's default log and custom log to a file)
def setup_log(log_level):
    # Set the record level of the log
    logging.basicConfig(level=log_level)  # Set the log level according to the configuration type

    # Create a log recorder, specify the path where the log is saved,
    # the maximum size of each log file, and the upper limit of the number of log files saved
    file_log_handler = RotatingFileHandler("logs/log", maxBytes=1024 * 1024 * 100, backupCount=10)
    # The format of the log record creation Log level Enter the file name of the log information
    # Number of lines Log information
    formatter = logging.Formatter('%(levelname)s %(pathname)s:%(lineno)d %(message)s')
    # Set the logging format for the logger just created
    file_log_handler.setFormatter(formatter)
    # Add a logger for the global logging tool object (used by flask app)
    logging.getLogger().addHandler(file_log_handler)


# Factory function: Material is provided by the outside world,
# and the creation process of the object is encapsulated inside the function
def create_app(config_type):  # Encapsulate the creation process of web applications
    # Take out the corresponding configuration subclass according to the type
    config_class = config_dict[config_type]
    app = Flask(__name__)
    app.config.from_object(config_class)

    # If the content of the registered blueprint object is only used once in the file,
    # it is best to import it before use, which can effectively avoid import errors
    from controller.modules.home import home_blu
    app.register_blueprint(home_blu)
    from controller.modules.user import user_blu
    app.register_blueprint(user_blu)

    # Set log LEVEL
    setup_log(config_class.LOG_LEVEL)

    return app
