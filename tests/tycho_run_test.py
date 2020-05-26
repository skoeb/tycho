import tycho
from tycho.config as config

def test_etl():
    config.N_GENERATORS = 10
    config.RUN_PRE_EE = True
    config.RUN_BAYES_OPT = True
    config.FETCH_S3 = True
    config.BAYES_N_ITER = 1
    config.BAYES_INIT_POINTS = 1

    # todo save db as test
    tycho.etl()

def test_train():

    # todo don't save model
    tycho.train()

def test_predict():

    config.PREDICT_COUNTRIES = ['Canada']

    # todo save db as test
    tycho.predict()


def test_plot():
    tycho.plot()


def test_plot():
    tycho.package()