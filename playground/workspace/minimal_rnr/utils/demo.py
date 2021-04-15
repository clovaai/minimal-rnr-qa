# Copied and modified from https://github.com/uwnlp/denspi

from time import time

from flask import Flask, request, jsonify
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.wsgi import WSGIContainer

from minimal_rnr.utils.logger import get_logger


def run_app(args, minimal_rnr):
    logger = get_logger("minimal-rnr-qa")

    inference_api = minimal_rnr.get_inference_api()

    app = Flask(__name__, static_url_path='/static')
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

    def _search(query, top_k, passage_score_weight):
        start = time()
        result = inference_api(query, top_k, passage_score_weight)
        return {"ret": result, "time": int((time() - start))}

    @app.route("/")
    def index():
        return app.send_static_file('index.html')

    @app.route("/files/<path:path>")
    def static_files(path):
        return app.send_static_file('files/' + path)

    @app.route("/api", methods=["GET"])
    def api():
        logger.info(request.args)

        query = request.args["query"]
        top_k = int(request.args["top_k"])

        if request.args["passage_score_weight"] == "null":
            passage_score_weight = None
        else:
            passage_score_weight = float(request.args["passage_score_weight"])

        result = _search(query, top_k, passage_score_weight)
        logger.info(result)
        return jsonify(result)

    @app.route("/get_examples", methods=["GET"])
    def get_examples():
        with open(args.examples_path, "r") as fp:
            examples = [line.strip() for line in fp.readlines()]
        return jsonify(examples)

    @app.route("/quit")
    def quit():
        raise KeyboardInterrupt

    logger.info("Warming up...")
    minimal_rnr.predict_answer("warmup", top_k=5, passage_score_weight=0.8)

    logger.info(f"Starting server at {args.demo_port}")
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(args.demo_port)
    IOLoop.instance().start()