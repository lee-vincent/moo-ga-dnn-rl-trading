import alpaca_api
import config
import unittest


class TestAlpacaApi(unittest.TestCase):

    def test_get_account_summary(self):
        """
        Test that get_account_summary() returns a JSON
        """
        # Note: Create a mock object?
        alpaca = alpaca_api.AlpacaConnect(config.ALPACA_KEY, config.ALPACA_SECRET_KEY)
        response = alpaca.get_account_summary()
        self.assertIsInstance(response, dict)
