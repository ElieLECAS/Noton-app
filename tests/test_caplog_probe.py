import logging

def test_probe_root(caplog):
    caplog.set_level(logging.INFO)
    logging.getLogger("app.services.page_retrieval_service").info("HELLO_ROOT")
    assert "HELLO_ROOT" in caplog.text

def test_probe_named(caplog):
    caplog.set_level(logging.INFO, logger="app.services.page_retrieval_service")
    logging.getLogger("app.services.page_retrieval_service").info("HELLO_NAMED")
    assert "HELLO_NAMED" in caplog.text

def test_probe_records(caplog):
    with caplog.at_level(logging.INFO):
        logging.getLogger("app.services.page_retrieval_service").info("HELLO_AT")
    assert any("HELLO_AT" in r.message for r in caplog.records)
