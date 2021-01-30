from jsonrpc import dispatcher, JSONRPCResponseManager
from flask.views import View
from flask import request
from werkzeug.wrappers import Response
from joblib import load
import os

from wine.domain.skl_logistic_regression import SklClassifier

path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)) + '/infra/models/'
model = load(path + 'modeloSKL2.joblib')
vocabulary = load(path + 'VOC_modeloSKL2.joblib')


@dispatcher.add_method
def scikit_learn_params():
    params = SklClassifier(model, vocabulary).params()
    return params


class RpcMetrics(View):
    methods = ['GET', 'POST']

    def dispatch_request(self):
        dispatcher["ping"] = lambda: "pong"

        try:
            response = JSONRPCResponseManager.handle(request.data, dispatcher)
        except Exception as e:
            raise e

        return Response(response.json, mimetype='application/json')
