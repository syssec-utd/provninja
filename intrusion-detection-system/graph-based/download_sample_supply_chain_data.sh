#!/bin/bash

# source: https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TyoWb0R8UQcUdq3lGkWa9hwxBEeYaYYX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TyoWb0R8UQcUdq3lGkWa9hwxBEeYaYYX" -O sample_supply_chain_data.zip && rm -rf /tmp/cookies.txt

unzip sample_supply_chain_data.zip
