from sml.context import Context
from sml.extract import Extract
from sml.transform import Transform
from sml.load import Load
from datetime import datetime, timedelta


class Pipeline:
    """
    sml base class, containing context, data and ETL functions
    """

    def __init__(self):
        self.context = Context()
        self.extract = Extract()
        self.transform = Transform()
        self.load = Load()
        self.messages = []
        self.wordfreq = None

    def run_pipline(self,
                    extract=True,
                    transform=True,
                    save_output=True,
                    start_date=datetime.today(),
                    end_date=datetime.today() - timedelta(days=7),
                    queries=None):
        if extract:
            self.messages = self.extract.get_data(start_date=start_date,
                                                  end_date=end_date,
                                                  queries=queries,
                                                  context=self.context)
        else:
            self.messages = self.load.get_messages(start_date=start_date,
                                                   end_date=end_date,
                                                   context=self.context)
        if transform:
            self.transform.process_messages(messages=self.messages,
                                            context=self.context)
        if save_output:
            self.load.save_messages(messages=self.messages,
                                    context=self.context)