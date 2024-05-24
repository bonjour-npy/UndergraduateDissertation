from flask import Flask, render_template, send_from_directory, request
from munch import Munch
from wsgiref.simple_server import make_server

from web_ui.models import StarGANv2, StyleGANv2, StyleGANv2_AFHQ
from web_ui.util import load_cfg, cache_path

cfg = load_cfg()
app = Flask(__name__)


@app.context_processor
def inject_config():
    return dict(cfg=cfg)


@app.route('/')
def index():
    return render_template('index.html')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.route('/model/<model_id>', methods=['GET'])
def model_page(model_id):
    if model_id in cfg.models:
        model = cfg.models[model_id]
        return render_template(f"{model_id}.html", title=model['name'], description=model['description'])
    else:
        return render_template('index.html', message=f'No such model: {model_id}.', is_warning=True)


@app.route('/api/model', methods=['POST'])
def model_inference():
    res = Munch({
        "success": False,
        "message": "default message",
        "data": None
    })

    try:
        model_name = request.form['model']
        if model_name == 'starganv2_afhq':
            res = StarGANv2.controller(request)
        if model_name == 'styleganv2_ffhq':  # request.form['model']
            res = StyleGANv2.controller(request)
        if model_name == 'styleganv2_afhq':
            res = StyleGANv2_AFHQ.controller(request)
        else:
            res.message = f"no such model: {model_name}"
    except Exception as e:
        res.message = str(e)
        print(e)
    return res


@app.route('/cache/<path:filename>')
def cached_image(filename):
    return send_from_directory(cache_path, filename)


@app.route('/api/<model_name>', methods=['POST'])
def predict(model_name):
    return {
        "success": True,
        "message": model_name
    }


# StarGANv2.init()
StyleGANv2.init()
StyleGANv2_AFHQ.init()

if __name__ == '__main__':
    httpd = make_server('0.0.0.0', 3000, app)
    print("Server on Click http://localhost:3000/ to start")
    httpd.serve_forever()
    # app.run(host='0.0.0.0', port=cfg.port, debug=True)
