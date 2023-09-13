from telethon import TelegramClient
from telethon.sessions import StringSession
import asyncio

'''
Usage:
- get your own api_id and api_hash from https://my.telegram.org, under API Development
- fill them in the variables below
- run this stript (e.g. F5)
- input your telegram phone number when prompt
- input the log-in code sent by telegram
- string session is then return in your terminal, save them!
'''


async def main():
    '''
    Function to get Telegram session string
    '''
    async with TelegramClient(StringSession(), api_id, api_hash) as client:
        string_session = client.session
        print(string_session.save())


api_id = 00000000
api_hash = 'asdasdasdasdasd'
# optional 
phone = 'your-phone'
username = 'your-username'

if __name__ == "__main__":
    asyncio.run(main())