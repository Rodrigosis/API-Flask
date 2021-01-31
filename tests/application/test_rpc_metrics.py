import json


def test_ping(client):
    response = client.post(
        "/metrics/",
        data=json.dumps({
            "method": "ping",
            "params": {},
            "jsonrpc": "2.0",
            "id": 0
        }),
        content_type='application/json'
    )

    assert response.status_code == 200

    rpc_response = json.loads(response.data)
    assert rpc_response['result'] == 'pong'

    assert rpc_response['id'] == 0
    assert rpc_response['jsonrpc'] == "2.0"


def test_scikit_learn_params(client):
    response = client.post(
        "/metrics/",
        data=payload,
        content_type='application/json'
    )

    assert response.status_code == 200

    rpc_response = json.loads(response.data)
    assert 'result' in rpc_response
    assert len(rpc_response['result']) == 2

    assert type(rpc_response['result']['accuracy']) == float

    model_params = rpc_response['result']['model_params']

    assert len(model_params) == 15

    assert 'C' in model_params
    assert 'class_weight' in model_params
    assert 'dual' in model_params
    assert 'fit_intercept' in model_params
    assert 'intercept_scaling' in model_params
    assert 'l1_ratio' in model_params
    assert 'max_iter' in model_params
    assert 'multi_class' in model_params
    assert 'n_jobs' in model_params
    assert 'penalty' in model_params
    assert 'random_state' in model_params
    assert 'solver' in model_params
    assert 'tol' in model_params
    assert 'verbose' in model_params
    assert 'warm_start' in model_params


# flake8: noqa
payload = json.dumps({"method": "scikit_learn_params",
                      "params": {},
                      "jsonrpc": "2.0",
                      "id": 0})
