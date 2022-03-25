import logging
import torram


def test_add_scalar():
    information = []
    logging.info = lambda x: information.append(x)  # mock logging module

    logger = torram.utility.logger.LogLogger()
    logger.add_scalar("test1", 5, global_step=1)
    logger.add_scalar("test2", "2", global_step=2)
    assert len(information) == 2
    assert "test1" in information[0] and "5" in information[0]
    assert "test2" in information[1] and "2" in information[1]


def test_add_scalar_dict():
    information = []
    logging.info = lambda x: information.append(x)  # mock logging module

    logger = torram.utility.logger.LogLogger()
    scalar_dict = {"a": 1, "b": 2}
    logger.add_scalar_dict("test", scalar_dict, global_step=1)
    assert len(information) == 2
    for i, (key, value) in enumerate(scalar_dict.items()):
        assert key in information[i] and str(value) in information[i]


def test_no_error_other_function():
    logger = torram.utility.logger.LogLogger()
    logger.add_image("test", None, global_step=5)
    logger.add_histogram("test", None, global_step=5)
