#!/bin/bash

# Parse command line arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <data_dir> <results_dir>"
    exit 1
fi

DATA_DIR="$1"
RESULTS_DIR="$2"

python examples/sonar_simple_trainer.py \
"prune_only" \
"--batch_size" "1" \
"--camera_model" "ortho" \
"--color_prior_asym" "0.08" \
"--color_prior_weight" "0.0" \
"--data_dir" "$DATA_DIR" \
"--results_dir" "$RESULTS_DIR" \
"--data_factor" "1" \
"--disable_viewer" \
"--elevate_end_step" "40000" \
"--elevate_loss_select" "50" \
"--elevate_num_samples" "5" \
"--elevate_sampling_duty_cycle" "0.2" \
"--elevate_start_step" "0" \
"--elevation_sampling_every" "500" \
"--end_at_frame" "10000" \
"--far_plane" "10" \
"--global_scale" "1.0" \
"--img_threshold" "0.2" \
"--init_extent" "1.0" \
"--init_num_pts" "10000" \
"--init_opa" "0.5" \
"--init_scale" "0.01" \
"--init_threshold" "0.4" \
"--init_type" "predefined" \
"--intermediate_azimuth_resolution" "0.05" \
"--lpips_net" "alex" \
"--max_size_prior_weight" "500.0" \
"--max_steps" "40000" \
"--near_plane" "-10" \
"--num_random_points" "0" \
"--opacity_prior_weight" "0.0" \
"--opacity_reg" "0.0" \
"--opacity_supervision_end_step" "10000" \
"--opacity_supervision_start_step" "0" \
"--opacity_supervision_thresh" "0.3" \
"--opacity_supervision_weight" "10.0" \
"--port" "8080" \
"--pose_noise" "0.0" \
"--pose_opt_lr" "1.0e-05" \
"--pose_opt_reg" "1.0e-06" \
"--range_clear_end" "200" \
"--range_clear_start" "800" \
"--render_traj_amplitude" "0.25" \
"--render_traj_freq" "10.0" \
"--render_traj_interp_val" "1" \
"--render_traj_path" "unchanged" \
"--sat_bg_prior_weight" "0.0" \
"--sat_region_prior_weight" "0.0" \
"--sat_sparsity_prior_weight" "0.0" \
"--sat_thresh" "0.09" \
"--scale_reg" "0.0" \
"--sh_degree" "3" \
"--sh_degree_interval" "1000" \
"--skip_frames" "8" \
"--skip_streak" \
"--ssim_lambda" "0.2" \
"--start_from_frame" "124" \
"--steps_scaler" "1.0" \
"--strategy.grow_grad2d" "0.0002" \
"--strategy.grow_scale2d" "0.05" \
"--strategy.grow_scale3d" "0.01" \
"--strategy.key_for_gradient" "means2d" \
"--strategy.pause_refine_after_reset" "0" \
"--strategy.prune_opa" "0.005" \
"--strategy.prune_scale2d" "0.15" \
"--strategy.prune_scale3d" "0.1" \
"--strategy.refine_every" "500" \
"--strategy.refine_scale2d_stop_iter" "0" \
"--strategy.refine_start_iter" "0" \
"--strategy.refine_stop_iter" "15000" \
"--strategy.reset_every" "3000" \
"--strategy.verbose" \
"--streak_end_step" "6000" \
"--streak_interval" "1000" \
"--streak_interval_ratio" "0.6" \
"--streak_start_step" "5000" \
"--tb_every" "100" \
"--tb_save_image" \
"--test_every" "1" \
"--train" \
"--wandb"
