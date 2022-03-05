BASE_FOLDER=0226


python td3_fast.py --agents=16 --global_='GLOBAL' --folder=${BASE_FOLDER}/gy2/td3_16 &
python td3_fast.py --agents=32 --global_='GLOBAL' --folder=${BASE_FOLDER}/gy2/td3_32 &

python td3_fast.py --agents=16 --global_='LOCAL' --folder=${BASE_FOLDER}/ly2/td3_16 &
python td3_fast.py --agents=32 --global_='LOCAL' --folder=${BASE_FOLDER}/ly2/td3_32 &

wait
