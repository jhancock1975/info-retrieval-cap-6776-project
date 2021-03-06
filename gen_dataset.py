# :author: jhancoc4@fau.edu
# Download and preprocess quarterly
# reports from the SEC edgar database

import logging
import argparse
import pandas as pd
import datetime as dt
import math

# for reading cik to ticker dictionary
import json

# for uncompressing file from SEC
# in gzip format
import zlib

# for downloading urls
import urllib.request

# for converting html to text
from bs4 import BeautifulSoup

# for pausing (sleeping) so
# we don't overwhelm the SEC
# or TD-Ameritrade
import time

# for working with samples
import random

def get_logger(name='main', level=logging.DEBUG):
    """
    create a logging object that covers most of our
    purposes, logs to console, debug log level.
    @param name: logger name, shows up on every line logged
    @param level: log level logging.info, logging.error, etc.
    @return: logging object configured to log time, level message
    """
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger
# create argument parser
parser = argparse.ArgumentParser(description='dataset generator')

class MetaDataRetriever(object):
    """
    object responsible for retrieving quarterly report
    metadata
    """
    
    def __init__(self, logger):
        """
        MetaDataRetriever constructor saves reference to logger

        :param logger: logger for the object to use
        """
        self.logger = logger
        # if set to true we will only retain
        # a random sample of companies to create
        # a dataset from
        # this can be useful for end-to-end test
        # of this code
        self.dev_mode = False

    def get_quarters(self, start_date: object, end_date: object) -> list:
        """
        gets list of strings of form YYYY/QTRX, where YYYY is the
        four digit year and literal "QTR" followed by 1,2,3,4 for
        the quarter of the respective year
        
        :param start_date: defines the quarter in which the first element
        in the result will belong to

        :param end_date: defines the quarter in which the last element
        in the result will belong to
        """
        start_year = start_date.year
        end_year = end_date.year
        start_quarter = math.ceil(start_date.month/3)
        result = []
        
        for year in range(start_year, end_year +1):
            for quarter in range(1, 5):
                result.append("{}/QTR{}".format(year, quarter))
        end_quarter = math.ceil(end_date.month/3)

        # we might have added extra quarters at the beginning and end
        # date so remove them
        for i in range(1, start_quarter):
            del result[0]

        for i in range(4, end_quarter, -1):
            del result[-1]

        return result

    def file_name_to_url(self, file_name: str) -> str:
        """
        convert file names from SEC report list
        to URL of file containing report

        :param file_name: value from file name column of
        company.gz file sec provides on a quarterly basis
        """
        report_base_url = "https://www.sec.gov/Archives/"
        
        split_name = file_name.split("/")
        dir_name = split_name[3].split(".")[0].replace("-","")
        path_name = "{}/{}/{}".format('/'.join(split_name[:3]), dir_name, split_name[3])
        return "{}{}".format(report_base_url, path_name)
                                      
    def get_filing_metadata(self, params:object) -> list:
        """
        returns list of metadata urls

        :param params: parameters for retrieving report data having
        keys 'start_date', 'end_date'
                
        :returns: list of urls for retrieval
        """
        self.logger.info("reading cik to ticker dictionary")
        with open("company_tickers.json") as cik_ticker_lookup_file:
            sec_cik_ticker_dict = json.load(cik_ticker_lookup_file)
        self.logger.info("complete reading cik to ticker dictionary")

        # we reorganize the SEC's lookup data because it is
        # keyed by integer index number
        self.logger.info("reorganizing dictionary")
        cik_ticker_dict = {}
        for key in sec_cik_ticker_dict:
            cur_val = sec_cik_ticker_dict[key]
            cik_ticker_dict[cur_val['cik_str']] = cur_val['ticker']
        
        self.logger.info("retrieving report locations")
        start_date = params['start_date']
        end_date = params['end_date']

        # reports on filings have this base url
        url_base = 'https://www.sec.gov/Archives/edgar/full-index'

        # get list of quarters to download
        quarters = self.get_quarters(start_date, end_date)
        self.logger.info("quarters = %s", quarters)

        listing_file_name = "company.gz"

        # CIK stands for central index key
        # dictionary to hold report locations
        # by quarter of year and CIK
        cik_file_dict = {}
        
        for quarter in quarters:         
            url = "{}/{}/{}".format(url_base, quarter, listing_file_name)
            logger.info("current url for downloading report list: %s", url)

            try:
                with urllib.request.urlopen(url) as res:
                    data = res.read()
                    # many thanks to StackOverflow user dnozay for the wbits
                    # answer https://stackoverflow.com/a/22310760
                    # we have to add the keyword argument
                    # wbits = zlib.MAX_WBITS | 16
                    # otherwise we get an error when attempting to decompress
                    company_list = zlib.decompress(data, wbits = zlib.MAX_WBITS | 16)
            except HTTPError:
                logger.error("HTTP Erorr attempting download company list data for %s",
                             url)
            except URLError:
                logger.error("URLlib error attempting download company list data for %s",
                             url)
                continue


            # wait 2 seconds when downloading in a loop
            # so we don't overwhelm the SEC
            self.logger.info("sleeping 1 second")
            time.sleep(1)
            self.logger.info("done sleeping")
            
            cik_file_dict[quarter] = {}
            cik_count = 0
            for line in company_list.splitlines():
                line = line.decode("utf-8")
                try:
                    if "10-Q" in line:
                        cik = int(line[74:80].rstrip())
                        ticker = cik_ticker_dict[cik]
                        company_name = line[:62].rstrip()                
                        file_name = line[98:].rstrip()
                        report_date = line[86:95].rstrip()
                        if not self.dev_mode or (self.dev_mode and random.random() > 0.95 ):
                            cik_file_dict[quarter][cik] = {}
                            cik_file_dict[quarter][cik]['company_name'] = company_name

                            cik_file_dict[quarter][cik]['file_name'] \
                                = self.file_name_to_url(file_name)

                            cik_file_dict[quarter][cik]['ticker'] = ticker
                            cik_file_dict[quarter][cik]['report_date'] = \
                                dt.datetime.strptime(args.start_date, "%Y-%m-%d")
                            cik_count = cik_count + 1
                except KeyError:
                    # not logging anything here, there are a lot of
                    # cik's not in the ticker dictionary
                    continue
        logger.info("dev mode = %s, saved %s cik's", self.dev_mode, cik_count)
        return cik_file_dict

class MetaDataLabler(object):
    """
    adds label to quarterly report data
    we follow the format from the Yelp example
    from the readme " 1 and 2 negative, and 3 and 4 positive",
    but we see in the yelp data we recover from the medium.com page
    the unique values are 1,2 when we sample the data
    """
    def __init__(self, logger):
        """
        constructor, saves reference to logger
        """
        self.logger = logger

    def label(self, filing_metadata: object) -> object:
        """
        labels quarterly report data
        we follow Yelp review polarity dataset format
        from the readme " 1 and 2 negative, and 3 and 4 positive"

        if share price is down 0% or more% we write a 1, otherwise 
        we write a 4

        :param filing_metadta: dictionary of quarterly report;
        input structure is {'quarter':
          {'cik': 
            {'company_name', 'file_name', 'ticker', 'report_date'}}}
        output structure is
        :return: dictionary with structure
        {'quarter': 
          {'cik': 
            {'company_name', 'file_name', 'ticker', 'report_date',
              label}}}
        
        we use TD-Ameritrade's REST API to get the change in share price
        """
        millis_90_days = 90*24*60*60*1000
        epoch = dt.datetime.utcfromtimestamp(0)
        
        # hold path to records
        # we need to delete
        # for http errors getting data
        # or incomplete price data
        records_to_delete = {}
        i = 1
        
        for quarter in filing_metadata:
            records_to_delete[quarter] = []
            for cik in filing_metadata[quarter]:
                if i % 100 == 0:
                    logger.info("labler: processing %s %s, record %s", quarter, cik, i)
                    i = i + 1
                ticker = filing_metadata[quarter][cik]['ticker']
                filing_date = filing_metadata[quarter][cik]['report_date']
                start_time_millis = int((filing_date - epoch).total_seconds()*1000)
                end_time_millis = start_time_millis + millis_90_days
                headers = {"Authorization" : ""}
                params = {"apikey": "GET A KEY FROM TD AMERITRADE",
                          "periodType" : "year",
                          "endDate" : end_time_millis,
                          "startDate" : start_time_millis}
                params = urllib.parse.urlencode(params)
                url = "https://api.tdameritrade.com/v1/marketdata/{}/pricehistory?{}"\
                      .format(ticker, params)
                # download the url, delte any records
                # we have trouble getting the price data for
                try:
                    with urllib.request.urlopen(url) as res:
                        price_data = json.loads(res.read().decode('utf-8'))
                except urllib.error.HTTPError:
                    logger.error("HTTP Error attempting to get price data for %s",
                                 cik)
                    records_to_delete[quarter].append(cik)
                    logger.debug("problem url: %s", url)
                    continue
                except:
                    logger.error("Error attempting to get price data for %s",
                                 cik)
                    records_to_delete[quarter].append(cik)
                    logger.error("problem url: %s", url)
                    continue

                # calculate change in price for the label
                # delete any records we do not get complete
                # price data for
                try:
                    begin_price = price_data['candles'][0]['open']
                    end_price = price_data['candles'][2]['close']
                    pct_change = (end_price - begin_price)/begin_price
                except IndexError:
                    logger.error("price data for ticker %s incomplete", ticker)
                    logger.error("data returned is: %s", price_data)
                    records_to_delete[quarter].append(cik)
                    continue
                except:
                    logger.error("exception handling %s", ticker)
                    records_to_delete[quarter].append(cik)
                    continue

                if pct_change <= 0:
                    label = 1
                else:
                    label = 2
                filing_metadata[quarter][cik]['label'] = label
                filing_metadata[quarter][cik]['pct_change'] = pct_change

                i = i + 1
                if i % 100 == 0:
                    logger.info("read %s price quotes", i)
                    
                
                # TD-Ameritrade service level agreement (SLA)
                # limits users to 120 requests per minute
                # https://developer.tdameritrade.com/content/getting-started
                # see section titled "making your first request"
                time.sleep(1)
                
        # delete any records we saved references to
        # when we encountered errors either in the http
        # "get" request to download the data, or
        # if the request returned insufficient data
        for quarter in records_to_delete:
            for cik in records_to_delete[quarter]:
                logger.info("removing record %s", filing_metadata[quarter][cik])
                del filing_metadata[quarter][cik]
                
        return filing_metadata
    
class ReportProcessor(object):
    """
    adds 10-q report text to report data
    """
    def __init__(self, logger):
        """
        constructor, saves reference to logger,
        sets
        """
        self.logger = logger

    
    def __html_to_text(self, data):
        """
        StackOverflow user Floyd
        https://stackoverflow.com/a/39899612
        remove html/javascript/css from html
        
        :param data: html source of some resource
        """
        # apply Yelp polarity dataset formatting
        # replace newlines with '\n' and double quotes
        # with double-double quotes
        # we truncate the HTML then truncate again after parsing
        text  =  ' '.join(BeautifulSoup(data[:200000], "html.parser").stripped_strings)
        # keep only first 100k characters
        text = text[:100000]
        text = text.replace("\n", "\\n")
        text = text.replace('"', '""')
        return '{}'.format(text)

    def process(self, filing_data: object, output_file_name: str) -> object:
        """
        adds report text from URL in filing data.
        write the report in the same format according to the
        description in the Yelp polarity dataset readme.txt:

        "There are 2 columns in them, corresponding to class index (1
        and 2) and review text. The review texts are escaped using
        double quotes ("), and any internal double quote is escaped by
        2 double quotes (""). New lines are escaped by a backslash
        followed with an "n" character, that is "\n"."

        :param filing_data: dictionary of quarterly report; input
        structure is:
        {'quarter': 
          {'cik': 
            {'company_name', 'file_name', 'ticker', 'filing date',
              label}}}

        :return: None, write csv file as a side-effect
        """
        records_to_remove = {}
        i = 1
        for quarter in filing_data:
            records_to_remove[quarter] = []
            for cik in filing_data[quarter]:
                logger.debug("process: %s %s", quarter, cik)
                url = filing_data[quarter][cik]['file_name']
                try:
                    with urllib.request.urlopen(url) as res:
                        data = res.read().decode('utf-8')
                        filing_data[quarter][cik]['report_text'] = self.__html_to_text(data)
                except urllib.error.HTTPError:
                    logger.error("HTTP Erorr attempting download company list data for %s",
                                 url)
                    records_to_remove[quarter].append(cik)
                    continue
                except ValueError as err:
                    logger.error("Error processing record quarter %s: cik %s, error %s",
                                 quarter, cik, err)
                    continue
                except:
                    logger.error("unknown error processing record quarter %s: cik: %s",
                                 quarter, cik)
                    continue
                
                # sleep for one second
                # so we are not overwhelming
                # the SEC's system
                time.sleep(1)
                i = i + 1
                if (i % 100 == 0):
                    logger.info("downloaded %s reports from EDGAR", i)

        for quarter in records_to_remove:
            for cik in records_to_remove[quarter]:
                logger.debug("removing record %s", filing_data[quarter][cik])
                del filing_data[quarter][cik]

        # write data to a csv file
        output_dict = {'label': [], 'report_text': [],
                       'company_name': [], 'ticker': [],
                       'report_date': [], 'pct_change': []}

        # build dictionary that we can construct
        # a pandas dataframe from
        logger.info("building output file %s", output_file_name)
        for quarter in filing_data:
            for cik in filing_data[quarter]:
                try :
                    label = filing_data[quarter][cik]['label']
                    report_text = filing_data[quarter][cik]['report_text']
                    ticker = filing_data[quarter][cik]['ticker']
                    company_name = filing_data[quarter][cik]['company_name']
                    report_date = filing_data[quarter][cik]['report_date']
                    pct_change = filing_data[quarter][cik]['pct_change']
                    output_dict['label'].append(label)
                    output_dict['report_text'].append(report_text)
                    output_dict['ticker'].append(ticker)
                    output_dict['company_name'].append(company_name)
                    output_dict['report_date'].append(report_date)
                    output_dict['pct_change'].append(pct_change)
                except KeyError:
                    logger.debug("key error for record %s %s", quarter,cik)
                    continue
        df = pd.DataFrame(output_dict)
        df.to_csv(output_file_name, index=False, header=False)
        logger.info("complete building output file %s", output_file_name)
        pass
                
if __name__ == "__main__":
    logger = get_logger(level=logging.INFO)

    logger.info('starting up')

    date_format_hint =  'in yyyy-mm-dd format, y=year, m=month number,\
    d= day of year'
    date_format_str = "%Y-%m-%d"
    
    parser.add_argument('-s', '--start_date', type=str,
                        help='start date {}'.format(date_format_hint))

    parser.add_argument('-e', '--end_date', type=str,
                        help='end date {}'.format(date_format_hint))

    parser.add_argument('-o', '--output_file_name', type=str,
                        help='fully qualified output file name')


    args = parser.parse_args()

    logger.info('read start date: %s, and end date %s',
                 args.start_date, args.end_date)

    start_date = dt.datetime.strptime(args.start_date, date_format_str)
    end_date = dt.datetime.strptime(args.end_date, date_format_str)

    metaDataRetriever = MetaDataRetriever(logger)
    params = {}
    params['start_date'] = start_date
    params['end_date'] = end_date
    filing_metadata = metaDataRetriever.get_filing_metadata(params)

    # add labels to data
    labler = MetaDataLabler(logger)
    filing_sentiments = labler.label(filing_metadata)

    # create csv files in format
    # the roBERTa example uses
    processor = ReportProcessor(logger)
    processor.process(filing_sentiments, args.output_file_name)

    logger.info("all complete")
