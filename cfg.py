gpu = 0

lt = 1.
lt_alpha = 10.
lb = 1.
lb_beta = 10.
lf = 1.
lf_theta_1 = 10.
lf_theta_2 = 1.
lf_theta_3 = 500.
epsilon = 1e-8

# train
learning_rate = 1e-4
decay_rate = 0.9
beta1 = 0.9
beta2 = 0.999
max_iter = 500000
show_loss_interval = 10
write_log_interval = 1
save_ckpt_interval = 500
gen_example_interval = 500
checkpoint_savedir = '/vtca/checkpoints/vits/swap/'
bg_checkpoint_savedir = '/vtca/checkpoints/vits/bg/'
# ckpt_path = './logs/trained_final_5M_.model'
ckpt_path = None
text_conversion_ckpt_path = None
bg_ckpt_path = "/vtca/checkpoints/vits/bg/bg-train_step-0001.model"
mask_ckpt_path = "/vtca/checkpoints/vits/mask/text-train_step-50000.model"

# data
batch_size = 32
bg_batch_size = 64
data_shape = [64, None]
data_dir = './dataset/'
i_t_dir = 'i_t'
i_t_bin_dir = 'i_t_bin'
i_s_dir = 'i_s'
t_sk_dir = 't_sk'
t_t_dir = 't_t'
t_b_dir = 't_b'
t_f_dir = 't_f'
mask_t_dir = 'mask_t'
example_data_dir = 'custom_feed/labels'
example_result_dir = 'custom_feed/gen_logs'

# predict
predict_ckpt_path = None
predict_data_dir = None
predict_result_dir = 'custom_feed/result'
