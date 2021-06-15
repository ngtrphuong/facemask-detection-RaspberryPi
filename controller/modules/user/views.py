from flask import session, redirect, url_for, request, render_template, jsonify

from controller.modules.user import user_blu
from controller.utils.camera import VideoCamera


# Log in
@user_blu.route("/login", methods=["GET", "POST"])
def login():
    username = session.get("username")

    if username:
        return redirect(url_for("home.index"))

    if request.method == "GET":
        return render_template("login.html")
    # Get parameters
    username = request.form.get("username")
    password = request.form.get("password")
    # Check parameters
    if not all([username, password]):
        return render_template("login.html", errmsg="Insufficient parameters")

    # Verify the corresponding administrator user data
    if username == "admin" and password == "admin":
        # Verification passed
        session["username"] = username
        return redirect(url_for("home.index"))

    return render_template("login.html", errmsg="Username or password is wrong")


# Sign Out
@user_blu.route("/logout")
def logout():
    # Delete session data
    session.pop("username", None)
    # Return to login page
    return redirect(url_for("user.login"))


# Recording status
@user_blu.route('/record_status', methods=['POST'])
def record_status():
    global video_camera
    if video_camera is None:
        video_camera = VideoCamera()

    json = request.get_json()

    status = json['status']

    if status == "true":
        video_camera.start_record()
        return jsonify(result="started")
    else:
        video_camera.stop_record()
        return jsonify(result="stopped")