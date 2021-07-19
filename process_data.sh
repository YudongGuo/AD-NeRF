python data_util/process_data.py --id=$1 --step=0 &
python data_util/process_data.py --id=$1 --step=1
python data_util/process_data.py --id=$1 --step=2
python data_util/process_data.py --id=$1 --step=6 &
python data_util/process_data.py --id=$1 --step=3
python data_util/process_data.py --id=$1 --step=4
python data_util/process_data.py --id=$1 --step=5
wait
python data_util/process_data.py --id=$1 --step=7
