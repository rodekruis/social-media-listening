from sml.extract import Extract
from sml.transform import Transform
from sml.load import Load
from sml.secrets import Secrets
from datetime import datetime, timedelta


class Pipeline:
    """
    sml base class for data pipeline
    """

    def __init__(self, secrets: Secrets = None):
        self.extract = Extract(secrets=secrets)
        self.transform = Transform(secrets=secrets)
        self.load = Load(secrets=secrets)
        self.messages = []

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
                                                  queries=queries)
        else:
            self.messages = self.load.get_messages(start_date=start_date,
                                                   end_date=end_date)
        if transform:
            self.transform.process_messages(messages=self.messages)
        if save_output:
            self.load.save_messages(messages=self.messages)