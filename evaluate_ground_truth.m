%% evaluate_ground_truth.m
% Compares ICP and NSFP estimates against ground truth linear stage position.

clear; clc; close all;

%% ── Load data ────────────────────────────────────────────────────────────────
% NEW ICP data (just generated)
opts_icp = detectImportOptions('icp_results_chained.csv');
opts_icp = setvartype(opts_icp, 'outlier', 'string');
icp = readtable('icp_results_bidirectional.csv', opts_icp);

% ORIGINAL NSFP data (from before)
nsfp = readtable('nsfp_results.csv');  % Make sure this is your ORIGINAL file

% NEW ground truth data (matching the new captures)
gt = readtable('linear_20260312_114259.csv');  % UPDATE THIS to match your new capture date

% Filter outliers
outlier_logical = strcmpi(icp.outlier, 'True') | strcmpi(icp.outlier, '1') | (icp.outlier == "True");
icp = icp(~outlier_logical, :);

% Remove session gaps from NSFP
nsfp = nsfp(nsfp.dt_ms < 10000, :);

fprintf('Loaded: ICP=%d, NSFP=%d, GT=%d samples\n', height(icp), height(nsfp), height(gt));

%% ── Build time axes ──────────────────────────────────────────────────────────
icp_time = cumsum([0; icp.dt_ms(1:end-1)]) / 1000;
nsfp_time = cumsum([0; nsfp.dt_ms(1:end-1)]) / 1000;

% Apply time shift to ICP
icp_time = icp_time - 9.7;

%% ── Interpolate ground truth at frame times ─────────────────────────────────
gt_pos_icp = interp1(gt.time_s, gt.position_mm, icp_time, 'linear', 'extrap');
gt_pos_nsfp = interp1(gt.time_s, gt.position_mm, nsfp_time, 'linear', 'extrap');

%% ── Extract Z component and flip sign ───────────────────────────────────────
% Extract Z displacement from ICP and multiply by -1
icp_z = -1 * icp.tz_mm;

% Compute cumulative position from Z displacement
icp_pos = cumsum(icp_z);

%% ── Compute ground truth inter-frame displacement ───────────────────────────
gt_disp_icp = abs(diff([gt_pos_icp(1); gt_pos_icp]));
gt_disp_nsfp = abs(diff([gt_pos_nsfp(1); gt_pos_nsfp]));

%% ── Compute per-frame error ──────────────────────────────────────────────────
icp_error = abs(abs(icp_z) - gt_disp_icp);
nsfp_error = abs(nsfp.mean_displacement_mm - gt_disp_nsfp);

%% ── Summary ──────────────────────────────────────────────────────────────────
fprintf('\n═════════════════════════════════════════════════════════════\n');
fprintf('  Ground Truth Error Analysis\n');
fprintf('═════════════════════════════════════════════════════════════\n');
fprintf('  Metric                  ICP          NSFP\n');
fprintf('  ─────────────────────────────────────────────────────────\n');
fprintf('  Mean error              %.3f mm      %.3f mm\n', mean(icp_error), mean(nsfp_error));
fprintf('  Median error            %.3f mm      %.3f mm\n', median(icp_error), median(nsfp_error));
fprintf('  Std deviation           %.3f mm      %.3f mm\n', std(icp_error), std(nsfp_error));
fprintf('  Max error               %.3f mm      %.3f mm\n', max(icp_error), max(nsfp_error));
fprintf('  RMSE                    %.3f mm      %.3f mm\n', rms(icp_error), rms(nsfp_error));
fprintf('═════════════════════════════════════════════════════════════\n\n');

%% ── Figure 1: Estimated vs Ground Truth ──────────────────────────────────────
figure('Name', 'Estimated vs GT', 'Position', [100 100 1400 500]);

plot(icp_time, gt_disp_icp, 'k-', 'LineWidth', 2.0, 'DisplayName', 'Ground Truth');
hold on;
plot(icp_time, abs(icp_z), 'b-', 'LineWidth', 1.0,...
    'DisplayName', sprintf('ICP  (RMSE=%.2fmm)', rms(icp_error)));
plot(nsfp_time, nsfp.mean_displacement_mm, 'r-', 'LineWidth', 1.0,...
    'DisplayName', sprintf('NSFP (RMSE=%.2fmm)', rms(nsfp_error)));

xlabel('Time (s)', 'FontSize', 12);
ylabel('Displacement (mm)', 'FontSize', 12);
title('Frame-to-Frame Displacement: Estimated vs Ground Truth', 'FontSize', 14);
legend('Location', 'northeast');
grid on;

%% ── Figure 2: Error over time ────────────────────────────────────────────────
figure('Name', 'Error Over Time', 'Position', [100 650 1400 450]);

plot(icp_time, icp_error, 'b-', 'LineWidth', 1.0,...
    'DisplayName', sprintf('ICP  (μ=%.2fmm)', mean(icp_error)));
hold on;
plot(nsfp_time, nsfp_error, 'r-', 'LineWidth', 1.0,...
    'DisplayName', sprintf('NSFP (μ=%.2fmm)', mean(nsfp_error)));

yline(mean(icp_error), 'b--', 'LineWidth', 1.0, 'HandleVisibility', 'off');
yline(mean(nsfp_error), 'r--', 'LineWidth', 1.0, 'HandleVisibility', 'off');

xlabel('Time (s)', 'FontSize', 12);
ylabel('Absolute Error (mm)', 'FontSize', 12);
title('Per-Frame Error vs Ground Truth', 'FontSize', 14);
legend('Location', 'northeast');
grid on;

%% ── Figure 3: Position reconstruction ────────────────────────────────────────
figure('Name', 'Position Reconstruction', 'Position', [100 100 1400 500]);

% Reconstruct NSFP position from cumulative displacement
nsfp_pos = gt_pos_nsfp(1) + cumsum(nsfp.mean_displacement_mm.* sign(diff([gt_pos_nsfp(1); gt_pos_nsfp])));

plot(gt.time_s, gt.position_mm, 'k-', 'LineWidth', 2.0, 'DisplayName', 'Ground Truth');
hold on;
plot(icp_time, icp_pos+20, 'b-', 'LineWidth', 1.0, 'DisplayName', 'ICP');
plot(nsfp_time, nsfp_pos, 'r-', 'LineWidth', 1.0, 'DisplayName', 'NSFP');

yline(45, 'g--', 'LineWidth', 1.0, 'DisplayName', 'POS\_A (45mm)');
yline(105, 'm--', 'LineWidth', 1.0, 'DisplayName', 'POS\_B (105mm)');

xlabel('Time (s)', 'FontSize', 12);
ylabel('Position (mm)', 'FontSize', 12);
title('Reconstructed Position vs Ground Truth', 'FontSize', 14);
legend('Location', 'northeast');
grid on;

%% ── Figure 4: Error distribution ─────────────────────────────────────────────
figure('Name', 'Error Distribution', 'Position', [100 100 1000 500]);

histogram(icp_error, 50, 'FaceColor', 'b', 'FaceAlpha', 0.5,...
    'DisplayName', sprintf('ICP  (RMSE=%.2fmm)', rms(icp_error)));
hold on;
histogram(nsfp_error, 50, 'FaceColor', 'r', 'FaceAlpha', 0.5,...
    'DisplayName', sprintf('NSFP (RMSE=%.2fmm)', rms(nsfp_error)));

xline(mean(icp_error), 'b--', 'LineWidth', 2, 'HandleVisibility', 'off');
xline(mean(nsfp_error), 'r--', 'LineWidth', 2, 'HandleVisibility', 'off');

xlabel('Absolute Error (mm)', 'FontSize', 12);
ylabel('Count', 'FontSize', 12);
title('Error Distribution vs Ground Truth', 'FontSize', 14);
legend('Location', 'northeast');
grid on;

fprintf('Done! Generated 4 figures.\n');
