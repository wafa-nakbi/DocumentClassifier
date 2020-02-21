import os
import unittest
from config import Config
from app import app

class BasicTests(unittest.TestCase):
    #setup, executed prior to each test
    def setUp(self):
        app.config['TESTING'] = True
        app.config['DEBUG'] = False
        self.app = app.test_client()

    # teardown, executed after each test
    def tearDown(self):
        pass

    def test_main_page(self):
        response = self.app.get('/index')
        self.assertEqual(response.status_code, 200)

if __name__=='__main__':
    unittest.main()

