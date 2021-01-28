from flask import current_app as app
from wine.application.rpc_predict import RpcPredict
from wine.application.rpc_metrics import RpcMetrics

app.add_url_rule('/predict/', view_func=RpcPredict.as_view('rpc_predict'))
app.add_url_rule('/metrics/', view_func=RpcMetrics.as_view('rpc_metrics'))
