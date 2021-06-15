# from flask_script import Manager
from controller import create_app

# Create APP object
app = create_app('dev')
# Create script management
# mgr = Manager(app)


if __name__ == '__main__':
    # mgr.run()
    app.run(threaded=True, host="0.0.0.0")


