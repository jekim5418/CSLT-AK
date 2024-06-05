#!/usr/bin/env bash

# Download action_token_pkl
# URL: https://drive.google.com/drive/folders/1binylH2pVpu6hrri4fSJCKH1hSvAe0Ke"
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nlNjSBfwY8sPONnuFDDbB0ZdUP5GHGUZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nlNjSBfwY8sPONnuFDDbB0ZdUP5GHGUZ" -O action_token_pkl/train.pickle && rm -rf ~/cookies.txt
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1tcyRf346yRuBBpkdOk1-bK4QYBnqxZCh' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tcyRf346yRuBBpkdOk1-bK4QYBnqxZCh" -O action_token_pkl/dev.pickle && rm -rf ~/cookies.txt
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1umwjCAsM1Bz_rIZo-YhR8qMe037j09SZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1umwjCAsM1Bz_rIZo-YhR8qMe037j09SZ" -O action_token_pkl/test.pickle && rm -rf ~/cookies.txt

# Download keypoint_pkl
# URL: https://drive.google.com/drive/folders/1-4hD0K3ThFLhW8CryTe2YqglXhQlRVkL
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nh63uujqJXQu0YxtUTfELKFFEwJp82T3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nh63uujqJXQu0YxtUTfELKFFEwJp82T3" -O keypoint_pkl/train.pickle && rm -rf ~/cookies.txt
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1D0TZqZr6GjPQH7SereaQKVF3u8dw4S-l' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1D0TZqZr6GjPQH7SereaQKVF3u8dw4S-l" -O keypoint_pkl/dev.pickle && rm -rf ~/cookies.txt
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Q-Fjx5Y5urUFx9vcOCnKXZKEhZHwRWax' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Q-Fjx5Y5urUFx9vcOCnKXZKEhZHwRWax" -O keypoint_pkl/test.pickle && rm -rf ~/cookies.txt

# Download phoenix features
wget "http://cihancamgoz.com/files/cvpr2020/phoenix14t.pami0.train" -P "PHOENIX2014T"
wget "http://cihancamgoz.com/files/cvpr2020/phoenix14t.pami0.dev" -P "PHOENIX2014T"
wget "http://cihancamgoz.com/files/cvpr2020/phoenix14t.pami0.test" -P "PHOENIX2014T"