import json


def test_ping(client):
    response = client.post(
        "/predict/",
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
        "/predict/",
        data=payload,
        content_type='application/json'
    )

    assert response.status_code == 200

    rpc_response = json.loads(response.data)
    assert 'result' in rpc_response
    assert len(rpc_response['result']) == 1

    assert 'nota' in rpc_response['result']


# flake8: noqa
payload = json.dumps({"method": "scikit_learn",
                      "params": {"tipo": "Licoroso",
                                 "uvas": "Castas tradicionais no Douro",
                                 "regiao": "Douro",
                                 "vinicola": "Burmester",
                                 "amadurecimento": "40 anos em barricas de carvalho",
                                 "classificacao": "Suave/Doce",
                                 "visual": "Acastanhado",
                                 "aroma": "Intenso, frutas secas, especiarias,mel"},
                      "jsonrpc": "2.0",
                      "id": 0})
