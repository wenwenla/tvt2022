python td3_fast_curr.py --agents=64 --init=baseline_fast_3/td3_4/0/models/9201.pkl --reuse=actor --folder=curr_test/64_from_4_actor_0 &
python td3_fast_curr.py --agents=64 --init=baseline_fast_3/td3_4/0/models/9201.pkl --reuse=copy --folder=curr_test/64_from_4_copy_0 &

python td3_fast_curr.py --agents=32 --init=baseline_fast_3/td3_4/0/models/9201.pkl --reuse=actor --folder=curr_test/32_from_4_actor_0 &
python td3_fast_curr.py --agents=32 --init=baseline_fast_3/td3_4/0/models/9201.pkl --reuse=copy --folder=curr_test/32_from_4_copy_0 &

python td3_fast_curr.py --agents=128 --init=baseline_fast_3/td3_4/0/models/9201.pkl --reuse=actor --folder=curr_test/128_from_4_actor_0 &
python td3_fast_curr.py --agents=128 --init=baseline_fast_3/td3_4/0/models/9201.pkl --reuse=copy --folder=curr_test/128_from_4_copy_0 &

python td3_fast_curr.py --agents=256 --init=baseline_fast_3/td3_4/0/models/9201.pkl --reuse=actor --folder=curr_test/256_from_4_actor_0 &
python td3_fast_curr.py --agents=256 --init=baseline_fast_3/td3_4/0/models/9201.pkl --reuse=copy --folder=curr_test/256_from_4_copy_0 &
wait