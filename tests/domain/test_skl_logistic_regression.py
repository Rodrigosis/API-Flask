from joblib import load
import os

from wine.domain.skl_logistic_regression import SklClassifier

path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)) + \
       '/wine/infra/models/'
model = load(path + 'modeloSKL2.joblib')
vocabulary = load(path + 'VOC_modeloSKL2.joblib')


def test_predict():
    predict = SklClassifier(model, vocabulary).predict

    result_1 = predict(tipo='Licoroso',
                       uvas='Castas tradicionais no Douro',
                       regiao='Douro',
                       vinicola='Burmester',
                       amadurecimento='Longo amadurecimento em cascos de carvalho (média de 30 anos).',
                       classificacao='Suave/Doce',
                       visual='Amarelo dourado com reflexos âmbar',
                       aroma='Frutas secas, amêndoas, nozes, mel, tosta')

    assert result_1[0] == 4

    result_2 = predict(tipo='Tinto',
                       uvas='Tannat (100%)',
                       regiao='Champagne',
                       vinicola='Jacquart',
                       amadurecimento='Método Tradicional (Segunda fermentação em garrafa).',
                       classificacao='Brut',
                       visual='Salmão. Perlage fina e delicada.',
                       aroma='Groselha, cereja, morango e ameixa, com leves notas de pão. ')

    assert result_2[0] == 4


def test_params():
    params = SklClassifier(model, vocabulary).params()

    assert type(params['accuracy']) == float

    assert params['model_params']['C'] == 1.0
    assert params['model_params']['class_weight'] is None
    assert params['model_params']['dual'] is False
    assert params['model_params']['fit_intercept'] is True
    assert params['model_params']['intercept_scaling'] == 1
    assert params['model_params']['l1_ratio'] is None
    assert params['model_params']['max_iter'] == 100
    assert params['model_params']['multi_class'] == 'auto'
    assert params['model_params']['n_jobs'] is None
    assert params['model_params']['penalty'] == 'l2'
    assert params['model_params']['random_state'] is None
    assert params['model_params']['solver'] == 'lbfgs'
    assert params['model_params']['tol'] == 0.0001
    assert params['model_params']['verbose'] == 0
    assert params['model_params']['warm_start'] is False
