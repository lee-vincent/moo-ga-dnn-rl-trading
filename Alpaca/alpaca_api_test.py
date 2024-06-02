import alpaca_api
import unittest
from unittest.mock import patch


class TestAlpacaApi(unittest.TestCase):

    def setUp(self):
        """
        Set up the AlpacaConnect object so it's a fresh instance
        before each test case runs.
        """
        # Initialize the AlpacaConnect object with mock credentials
        self.alpaca = alpaca_api.AlpacaConnect("mock_key", "mock_secret")

    @patch('alpaca_api.AlpacaConnect.get_account_summary')
    def test_get_account_summary(self, mock_get_account_summary):
        """
        Test that get_account_summary() returns a JSON
        """
        # Mock return value of get_account_summary
        mock_get_account_summary.return_value = {
            "id": "12345",
            "status": "ACTIVE",
            "currency": "USD",
            "cash": "10000.00",
            "portfolio_value": "15000.00"
        }

        # Call the method under test
        response = self.alpaca.get_account_summary()

        # Assertions
        self.assertIsInstance(response, dict)
        self.assertEqual(response['id'], "12345")
        self.assertEqual(response['status'], "ACTIVE")

    @patch('alpaca_api.AlpacaConnect.place_order')
    def test_place_market_order(self, mock_place_market_order):
        """
        Test that a market order was placed, buy side
        """
        mock_place_market_order.return_value = {
            "id": "order123",
            "symbol": "TQQQ",
            "qty": 10,
            "side": "buy",
            "type": "market",
            "status": "filled"
        }

        response = self.alpaca.place_order("TQQQ", 10, "buy")

        self.assertIsInstance(response, dict)
        self.assertEqual(response['id'], "order123")
        self.assertEqual(response['status'], "filled")

    @patch('alpaca_api.AlpacaConnect.place_order')
    def test_place_stop_order(self, mock_place_stop_order):
        """
        Test that a stop order was placed, buy side
        """
        mock_place_stop_order.return_value = {
            "id": "order124",
            "symbol": "TQQQ",
            "qty": 10,
            "side": "buy",
            "type": "stop",
            "status": "accepted",
            "stop_price": 35
        }

        response = self.alpaca.place_order("TQQQ", 10, "buy", order_type='stop', stop_price=35)

        self.assertIsInstance(response, dict)
        self.assertEqual(response['id'], "order124")
        self.assertEqual(response['status'], "accepted")

    @patch('alpaca_api.AlpacaConnect.place_order')
    def test_place_limit_order(self, mock_place_limit_order):
        """
        Test that limit order was placed, buy side
        """
        mock_place_limit_order.return_value = {
            "id": "order125",
            "symbol": "TQQQ",
            "qty": 10,
            "side": "buy",
            "type": "limit",
            "status": "accepted",
            "limit_price": 50
        }

        response = self.alpaca.place_order("TQQQ", 10, "buy", order_type='limit', stop_price=50)

        self.assertIsInstance(response, dict)
        self.assertEqual(response['id'], "order125")
        self.assertEqual(response['status'], "accepted")

    @patch('alpaca_api.AlpacaConnect.cancel_one_order')
    def test_cancel_one_order(self, mock_cancel_one_order):
        """
        Attempt to cancel one Open Order. Status code Success: 204, Fail: 422
        """
        mock_cancel_one_order.return_value = 204

        order_id = "abc123"

        status = self.alpaca.cancel_one_order(order_id)

        self.assertEqual(status, 204)


if __name__ == '__main__':
    unittest.main()
