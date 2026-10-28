"""Prueba de humo: pytest corre y src/ está en el path."""


def test_common_se_importa():
    import common

    assert common.__name__ == "common"
