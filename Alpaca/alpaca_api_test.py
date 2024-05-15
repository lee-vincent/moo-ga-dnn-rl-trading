import alpaca_api
import config
import unittest


class TestAlpacaApi(unittest.TestCase):
    # TODO Create mock data for API calls

    def test_get_account_summary(self):
        """
        Test that get_account_summary() returns a JSON
        """
        alpaca = alpaca_api.AlpacaConnect(config.ALPACA_KEY, config.ALPACA_SECRET_KEY)
        response = alpaca.get_account_summary()
        self.assertIsInstance(response, dict)

    def test_place_market_order(self):
        """
        Test that a market order was placed, buy side
        """
        alpaca = alpaca_api.AlpacaConnect(config.ALPACA_KEY, config.ALPACA_SECRET_KEY)
        response = alpaca.place_order("TQQQ", 10, "buy")
        self.assertIsInstance(response, dict)

    def test_place_stop_order(self):
        """
        Test that a stop order was placed, buy side
        """
        alpaca = alpaca_api.AlpacaConnect(config.ALPACA_KEY, config.ALPACA_SECRET_KEY)
        response = alpaca.place_order("TQQQ", 10, "buy", order_type='stop', stop_price=35)
        self.assertIsInstance(response, dict)

    def test_place_limit_order(self):
        """
        Test that limit order was placed, buy side
        """
        alpaca = alpaca_api.AlpacaConnect(config.ALPACA_KEY, config.ALPACA_SECRET_KEY)
        response = alpaca.place_order("TQQQ", 10, "buy", order_type='limit', stop_price=50)
        self.assertIsInstance(response, dict)

    def test_cancel_one_order(self):
        """
        Attempt to cancel one Open Order. Status code Success: 204, Fail: 422
        """
        order_id = "47dbcf72-63a5-4b9f-818b-aa6ab69fd929"
        alpaca = alpaca_api.AlpacaConnect(config.ALPACA_KEY, config.ALPACA_SECRET_KEY)
        status = alpaca.cancel_one_order(order_id)
        self.assertEqual(status, 204)


if __name__ == '__main__':
    unittest.main()
