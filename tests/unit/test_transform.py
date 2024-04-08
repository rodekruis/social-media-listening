import os.path
from datetime import datetime, timedelta
from sml.pipeline import Pipeline
from sml.secrets import Secrets
from sml.message import Message

if not os.path.exists("credentials/.env"):
    print('credentials not found, run this test from root directory')
pipe = Pipeline(secrets=Secrets("credentials/.env"))

messages = [
        Message(
            id_="0",
            datetime_=datetime(2023, 11, 29, 17, 00, 00),
            datetime_scraped_=datetime.now(),
            country="UKR",
            source="Telegram",
            text="Доброго дня. Скажіть будь ласка :ВИ напевно знаете і ціну за сотку??? Напишіть будь ласка. ДЯКУЮ.",
            group="test",
            reply=False,
            reply_to=None,
            translations=None
        )
    ]

pipe.transform.set_translator(
    model="Microsoft",
    from_lang=["uk", "ru"],  # empty string means auto-detect language
    to_lang="en"
)
messages = pipe.transform.process_messages(messages, translate=True)
print(f"processed {len(messages)} messages!")
for message in messages:
    inputs = {'Original message': message.text}
    if message.translations:
        for translation in message.translations:
            inputs[f"Translation ({str(translation['from_lang']).upper()}-{str(translation['to_lang']).upper()})"] = translation['text']
    print(inputs)
