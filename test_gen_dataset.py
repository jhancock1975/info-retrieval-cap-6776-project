import unittest
from gen_dataset import MetaDataRetriever, get_logger
from datetime import date

class TestGenDataSet(unittest.TestCase):

    def setUp(self):
        """
        constructor, initializes logger
        """
        self.logger = get_logger()
        #self.metaDataRetriever = MetaDataRetriever(self.logger)
        
    def test_get_quarters(self):
        """
        get that MetaDataRetriever gets proper list of years
        and quarters for date range
        """
        metaDataRetriever = MetaDataRetriever(self.logger)
        q_list = metaDataRetriever.get_quarters(date(2018, 10, 1),
                                        date(2019, 2, 1))
        self.assertListEqual(q_list, ['2018/QTR4', '2019/QTR1'])


if __name__ == '__main__':
    unittest.main()
