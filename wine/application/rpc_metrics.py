from jsonrpc import dispatcher, JSONRPCResponseManager
from flask.views import View
from flask import request
from werkzeug.wrappers import Response


@dispatcher.add_method
def model_1():
    return 'ok'


@dispatcher.add_method
def model_2():
    return 'ok'


class RpcMetrics(View):
    methods = ['GET', 'POST']
    """https://flask.palletsprojects.com/en/1.1.x/api/#class-based-views
    https://json-rpc.readthedocs.io/en/latest/quickstart.html
    https://json-rpc.readthedocs.io/en/latest/flask_integration.html
    """

    def dispatch_request(self):
        # Dispatcher is dictionary {<method_name>: callable}
        dispatcher["ping"] = lambda: "pong"

        try:
            response = JSONRPCResponseManager.handle(request.data, dispatcher)
        except Exception as e:
            raise e

        return Response(response.json, mimetype='application/json')
