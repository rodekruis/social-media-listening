from datetime import datetime, timedelta
supported_sources = ["twitter", "kobo", "telegram"]


class SocialMediaSecrets:
    """
    Secrets (API keys, tokens, etc.) for social media source
    """
    def __init__(self, api_key="", api_secret="",
                 access_token="", access_secret="",
                 asset=""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_secret = access_secret
        self.asset = asset
        # TBI automatically get secrets from environment variables (?)


class SocialMediaSource:
    """
    social media source
    """
    def __init__(self, name, secrets=None):
        if name not in supported_sources:
            raise ValueError(f"source {name} is not supported."
                             f"Supported sources are {', '.join(supported_sources)}")
        else:
            self.name = name
        self.secrets = secrets

    def set_secrets(self, secrets):
        if not isinstance(secrets, SocialMediaSecrets):
            raise TypeError(f"invalid format of secrets, use extract.SMSecrets")
        self.secrets = secrets

    def check_secrets(self):
        # TBI check that right secrets are filled for data source
        return True


class Extract:
    """
    extract data from social media
    """

    def __init__(self, source=None, secrets=None):
        if source is not None:
            self.source = SocialMediaSource(source, secrets)
        else:
            self.source = None

    def set_source(self, source, secrets):
        self.source = SocialMediaSource(source, secrets)

    def get_data(self,
                 start_date=datetime.today(),
                 end_date=datetime.today()-timedelta(days=7),
                 queries=None,
                 users=None,
                 channels=None,
                 pages=None):

        if not self.source.check_secrets():
            raise ValueError("no social media secrets found")

        if self.source.name == "twitter":
            # TBI get data from Twitter
            messages = self.get_data_twitter()
        elif self.source.name == "kobo":
            # TBI get data from Twitter
            messages = self.get_data_kobo()
        elif self.source.name == "telegram":
            # TBI get data from Telegram
            messages = self.get_data_telegram()
        else:
            raise ValueError(f"source {self.source.name} is not supported."
                             f"Supported sources are {', '.join(supported_sources)}")

        return messages

    def get_data_twitter(self):
        # TBI
        return []

    def get_data_kobo(self):
        # TBI
        return []

    def get_data_telegram(self):
        # TBI
        return []



