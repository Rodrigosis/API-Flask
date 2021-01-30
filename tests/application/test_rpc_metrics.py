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
    assert len(rpc_response['result']) == 15

    assert 'C' in rpc_response['result']
    assert 'class_weight' in rpc_response['result']
    assert 'dual' in rpc_response['result']
    assert 'fit_intercept' in rpc_response['result']
    assert 'intercept_scaling' in rpc_response['result']
    assert 'l1_ratio' in rpc_response['result']
    assert 'max_iter' in rpc_response['result']
    assert 'multi_class' in rpc_response['result']
    assert 'n_jobs' in rpc_response['result']
    assert 'penalty' in rpc_response['result']
    assert 'random_state' in rpc_response['result']
    assert 'solver' in rpc_response['result']
    assert 'tol' in rpc_response['result']
    assert 'verbose' in rpc_response['result']
    assert 'warm_start' in rpc_response['result']


# flake8: noqa
payload = json.dumps({"method": "scikit_learn_params",
                      "params": {},
                      "jsonrpc": "2.0",
                      "id": 0})
