# test lasher
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./RGBT_workspace/test_rgbt_mgpus.py --script_name SMSTracker --dataset_name LasHeR --yaml_name plscore_os_sigma_RGBT

# test rgbt234
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./RGBT_workspace/test_rgbt_mgpus.py --script_name SMSTracker --dataset_name RGBT234 --yaml_name plscore_os_sigma_RGBT