from flask import Flask, render_template

app = Flask(__name__, template_folder="templateFiles")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/info_page")
def info_age():
    return render_template('info_page.html')

@app.route("/test")
def test():
    print("It works")
    return render_template('generate.html')

if __name__=='__main__':
    app.run(debug=True)
