from openpi_client import websocket_client_policy


class _FakeConnection:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_close_closes_underlying_websocket() -> None:
    connection = _FakeConnection()
    policy = websocket_client_policy.WebsocketClientPolicy.__new__(websocket_client_policy.WebsocketClientPolicy)
    policy._ws = connection

    policy.close()

    assert connection.close_calls == 1
